"""Gate 1: history must earn the relation graph before it buys measurements.

Scaffold kept explicit: patch correspondence is given across history and the
future frame. What is *not* given is object membership. Four parts of each
object share the same ordered motion history. All four object histories have
the same marginal set of velocity vectors, so only temporal common fate (not
mean velocity or speed) identifies the relation.

The learned relation is then carried to a future rearrangement and used as the
prior for sparse pixel reconstruction. Time-shuffled history, absolute
coordinate memory, and the ordinary spatial lattice are attackers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Iterable

import numpy as np

from what_to_look_at import (
    BLOCK_SIZE,
    COLORS,
    IMAGE_SIZE,
    N_PIXELS,
    active_component_mask,
    group_harmonic_reconstruct,
    psnr,
    spatial_harmonic_reconstruct,
)

N_OBJECTS = 4
PARTS_PER_OBJECT = 4
N_PARTS = N_OBJECTS * PARTS_PER_OBJECT
HISTORY_STEPS = 8
DEFAULT_COUNTS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


@dataclass(frozen=True)
class CurveReceipt:
    median_psnr_by_count: dict[str, float]
    success_fraction_by_count: dict[str, float]
    first_count_at_90pct_success: int | None


@dataclass(frozen=True)
class Gate1Receipt:
    worlds: int
    history_steps: int
    motion_noise: float
    median_common_fate_ari: float
    median_time_shuffled_ari: float
    active_learned_measurements_median: float
    active_learned_success_fraction: float
    learned_common_fate: CurveReceipt
    oracle_relation: CurveReceipt
    time_shuffled_history: CurveReceipt
    coordinate_memory: CurveReceipt
    spatial_lattice: CurveReceipt


def true_part_groups() -> np.ndarray:
    return np.repeat(np.arange(N_OBJECTS, dtype=np.int64), PARTS_PER_OBJECT)


def make_motion_history(seed: int, noise: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    """Return per-part motion traces with equal marginals but different order."""
    rng = np.random.default_rng(seed)
    base = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
        ],
        dtype=np.float64,
    )
    groups = true_part_groups()
    traces = []
    for group in groups:
        ordered = np.roll(base, 2 * int(group), axis=0)
        traces.append(ordered + rng.normal(scale=noise, size=ordered.shape))
    return np.asarray(traces), groups


def independently_shuffle_time(traces: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(traces).copy()
    for part in range(out.shape[0]):
        out[part] = out[part][rng.permutation(out.shape[1])]
    return out


def kmeans_farthest(x: np.ndarray, k: int, iterations: int = 64) -> np.ndarray:
    """Tiny deterministic k-means; no external ML dependency."""
    x = np.asarray(x, dtype=np.float64)
    centre = x.mean(axis=0)
    centres = [x[np.argmax(np.sum((x - centre) ** 2, axis=1))].copy()]
    while len(centres) < k:
        d2 = np.min(
            np.stack([np.sum((x - c) ** 2, axis=1) for c in centres], axis=0),
            axis=0,
        )
        centres.append(x[np.argmax(d2)].copy())
    centres = np.stack(centres)

    labels = np.full(x.shape[0], -1, dtype=np.int64)
    for _ in range(iterations):
        d2 = np.sum((x[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            member = labels == idx
            if np.any(member):
                centres[idx] = x[member].mean(axis=0)
    return labels


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    n = int(a.size)
    ua = np.unique(a)
    ub = np.unique(b)
    table = np.asarray([[np.sum((a == i) & (b == j)) for j in ub] for i in ua])
    row = table.sum(axis=1)
    col = table.sum(axis=0)

    nij = sum(comb(int(v), 2) for v in table.ravel() if v >= 2)
    ai = sum(comb(int(v), 2) for v in row if v >= 2)
    bj = sum(comb(int(v), 2) for v in col if v >= 2)
    total = comb(n, 2)
    expected = ai * bj / total
    maximum = 0.5 * (ai + bj)
    return 1.0 if maximum == expected else float((nij - expected) / (maximum - expected))


def future_patch_grid(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = np.arange(N_PARTS, dtype=np.int64)
    rng.shuffle(ids)
    return ids.reshape((N_OBJECTS, N_OBJECTS))


def expand_patch_values(patch_values: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(patch_values, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1)


def future_true_groups(patch_grid: np.ndarray) -> np.ndarray:
    return expand_patch_values((np.asarray(patch_grid) // PARTS_PER_OBJECT).astype(np.int64))


def future_learned_groups(patch_grid: np.ndarray, part_clusters: np.ndarray) -> np.ndarray:
    return expand_patch_values(np.asarray(part_clusters)[np.asarray(patch_grid)])


def coordinate_memory_groups() -> np.ndarray:
    """Absolute-coordinate attacker: assume the original row partition stayed put."""
    block_rows = np.repeat(np.arange(N_OBJECTS)[:, None], N_OBJECTS, axis=1)
    return expand_patch_values(block_rows)


def scene_from_groups(groups: np.ndarray) -> np.ndarray:
    return COLORS[np.asarray(groups)]


def random_mask(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.choice(N_PIXELS, int(count), replace=False)
    flat = np.zeros(N_PIXELS, dtype=bool)
    flat[ids] = True
    return flat.reshape((IMAGE_SIZE, IMAGE_SIZE))


def reconstruct(scene: np.ndarray, mask: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    observed = np.zeros_like(scene)
    observed[mask] = scene[mask]
    if groups is None:
        return spatial_harmonic_reconstruct(mask, observed)
    return group_harmonic_reconstruct(mask, observed, groups)


def _curve(
    worlds: list[dict],
    key: str,
    counts: Iterable[int],
    threshold_psnr_db: float,
) -> CurveReceipt:
    medians: dict[str, float] = {}
    success: dict[str, float] = {}
    first: int | None = None

    for count in counts:
        scores = []
        for world in worlds:
            mask = random_mask(int(count), world["measurement_seed"] + int(count) * 1009)
            groups = None if key == "spatial" else world[key]
            recovered = reconstruct(world["scene"], mask, groups)
            scores.append(psnr(world["scene"], recovered))
        medians[str(int(count))] = float(np.median(scores))
        rate = float(np.mean(np.asarray(scores) >= threshold_psnr_db))
        success[str(int(count))] = rate
        if first is None and rate >= 0.90:
            first = int(count)
    return CurveReceipt(medians, success, first)


def run_gate1(
    n_worlds: int = 48,
    motion_noise: float = 0.30,
    counts: Iterable[int] = DEFAULT_COUNTS,
    threshold_psnr_db: float = 40.0,
) -> Gate1Receipt:
    worlds: list[dict] = []
    ari = []
    shuffled_ari = []
    active_scores = []
    active_counts = []

    for world_idx in range(int(n_worlds)):
        traces, true_groups = make_motion_history(5000 + world_idx, motion_noise)
        learned = kmeans_farthest(traces.reshape((N_PARTS, -1)), N_OBJECTS)
        shuffled_traces = independently_shuffle_time(traces, 9000 + world_idx)
        shuffled = kmeans_farthest(shuffled_traces.reshape((N_PARTS, -1)), N_OBJECTS)
        ari.append(adjusted_rand_index(true_groups, learned))
        shuffled_ari.append(adjusted_rand_index(true_groups, shuffled))

        grid = future_patch_grid(12000 + world_idx)
        true_pixels = future_true_groups(grid)
        scene = scene_from_groups(true_pixels)
        learned_pixels = future_learned_groups(grid, learned)
        shuffled_pixels = future_learned_groups(grid, shuffled)
        coordinate_pixels = coordinate_memory_groups()

        active_mask = active_component_mask(learned_pixels)
        active_scores.append(psnr(scene, reconstruct(scene, active_mask, learned_pixels)))
        active_counts.append(int(active_mask.sum()))

        worlds.append(
            {
                "scene": scene,
                "oracle": true_pixels,
                "learned": learned_pixels,
                "shuffled": shuffled_pixels,
                "coordinate": coordinate_pixels,
                "spatial": None,
                "measurement_seed": 20000 + world_idx,
            }
        )

    return Gate1Receipt(
        worlds=int(n_worlds),
        history_steps=HISTORY_STEPS,
        motion_noise=float(motion_noise),
        median_common_fate_ari=float(np.median(ari)),
        median_time_shuffled_ari=float(np.median(shuffled_ari)),
        active_learned_measurements_median=float(np.median(active_counts)),
        active_learned_success_fraction=float(np.mean(np.asarray(active_scores) >= threshold_psnr_db)),
        learned_common_fate=_curve(worlds, "learned", counts, threshold_psnr_db),
        oracle_relation=_curve(worlds, "oracle", counts, threshold_psnr_db),
        time_shuffled_history=_curve(worlds, "shuffled", counts, threshold_psnr_db),
        coordinate_memory=_curve(worlds, "coordinate", counts, threshold_psnr_db),
        spatial_lattice=_curve(worlds, "spatial", counts, threshold_psnr_db),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("results/gate1_summary.json"))
    args = parser.parse_args()
    result = run_gate1(n_worlds=args.worlds)
    payload = asdict(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
