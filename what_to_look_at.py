"""Gate 0 for WhatToLookAt.

Question: can a relation prior replace physical measurements?

The world is intentionally synthetic and the relation graph is oracle in Gate 0.
Four objects are broken into spatially disconnected patches. A relation graph
knows which pixels belong to the same object; controls know only the spatial
lattice or receive a size-matched shuffled relation graph.

Reconstruction is deliberately simple: harmonic interpolation on each graph.
For a complete relation component, that is exactly the mean of observed values
in that component. The spatial control uses lazy Jacobi harmonic inpainting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_SIZE = 16
BLOCK_SIZE = 4
N_OBJECTS = 4
N_PIXELS = IMAGE_SIZE * IMAGE_SIZE

COLORS = np.asarray(
    [
        [0.10, 0.20, 0.90],
        [0.90, 0.20, 0.10],
        [0.20, 0.85, 0.25],
        [0.90, 0.80, 0.10],
    ],
    dtype=np.float64,
)

DEFAULT_COUNTS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


@dataclass(frozen=True)
class MethodReceipt:
    median_psnr_by_count: dict[str, float]
    success_fraction_by_count: dict[str, float]
    first_count_at_90pct_success: int | None


@dataclass(frozen=True)
class Gate0Receipt:
    image_size: int
    pixels: int
    objects: int
    threshold_psnr_db: float
    active_relation_measurements: int
    active_relation_fraction: float
    active_relation_psnr_db: float
    random_relation: MethodReceipt
    spatial_lattice: MethodReceipt
    shuffled_relation: MethodReceipt


def make_labels() -> np.ndarray:
    """Return a Latin-square arrangement of disconnected object patches."""
    nblocks = IMAGE_SIZE // BLOCK_SIZE
    if nblocks != N_OBJECTS:
        raise ValueError("Gate 0 assumes one block per object per block-row/column")
    latin = np.fromfunction(
        lambda r, c: (r + c) % N_OBJECTS,
        (nblocks, nblocks),
        dtype=int,
    ).astype(np.int64)
    return np.repeat(np.repeat(latin, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1)


def make_scene(labels: np.ndarray | None = None) -> np.ndarray:
    labels = make_labels() if labels is None else np.asarray(labels)
    return COLORS[labels]


def shuffled_groups(labels: np.ndarray, seed: int = 99173) -> np.ndarray:
    """Preserve group sizes while destroying alignment with the true objects."""
    rng = np.random.default_rng(seed)
    flat = labels.ravel().copy()
    return flat[rng.permutation(flat.size)].reshape(labels.shape)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(reference) - np.asarray(estimate)) ** 2))
    if mse <= 1e-14:
        return 120.0
    return float(10.0 * np.log10(1.0 / mse))


def group_harmonic_reconstruct(
    observed_mask: np.ndarray,
    observed_image: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Harmonic interpolation for disconnected complete relation components.

    The minimum-energy value on an unobserved node in a complete constant-value
    component is the mean of the clamped observations in that component. An
    entirely unobserved component is intentionally *not guessed*: it receives
    the global observed mean, exposing the information boundary.
    """
    mask = np.asarray(observed_mask, dtype=bool)
    obs = np.asarray(observed_image, dtype=np.float64)
    groups = np.asarray(groups)
    if not np.any(mask):
        raise ValueError("at least one measurement is required")

    out = np.empty_like(obs)
    global_mean = obs[mask].mean(axis=0)
    for group in np.unique(groups):
        member = groups == group
        known = member & mask
        value = obs[known].mean(axis=0) if np.any(known) else global_mean
        out[member] = value
    out[mask] = obs[mask]
    return out


def spatial_harmonic_reconstruct(
    observed_mask: np.ndarray,
    observed_image: np.ndarray,
    iterations: int = 350,
) -> np.ndarray:
    """Lazy Jacobi harmonic inpainting on the ordinary 4-neighbour lattice."""
    mask = np.asarray(observed_mask, dtype=bool)
    obs = np.asarray(observed_image, dtype=np.float64)
    if not np.any(mask):
        raise ValueError("at least one measurement is required")

    h, w, channels = obs.shape
    global_mean = obs[mask].mean(axis=0)
    x = np.broadcast_to(global_mean, (h, w, channels)).copy()
    x[mask] = obs[mask]

    degree = np.zeros((h, w), dtype=np.float64)
    degree[:-1, :] += 1
    degree[1:, :] += 1
    degree[:, :-1] += 1
    degree[:, 1:] += 1

    for _ in range(iterations):
        total = np.zeros_like(x)
        total[:-1, :] += x[1:, :]
        total[1:, :] += x[:-1, :]
        total[:, :-1] += x[:, 1:]
        total[:, 1:] += x[:, :-1]
        neighbour_mean = total / degree[..., None]
        x = 0.5 * x + 0.5 * neighbour_mean
        x[mask] = obs[mask]
    return x


def random_mask(count: int, seed: int) -> np.ndarray:
    if count < 1 or count > N_PIXELS:
        raise ValueError("measurement count must be in [1, N_PIXELS]")
    rng = np.random.default_rng(seed)
    ids = rng.choice(N_PIXELS, int(count), replace=False)
    mask = np.zeros(N_PIXELS, dtype=bool)
    mask[ids] = True
    return mask.reshape((IMAGE_SIZE, IMAGE_SIZE))


def active_component_mask(groups: np.ndarray) -> np.ndarray:
    """Spend exactly one measurement on every known relation component."""
    flat = np.asarray(groups).ravel()
    mask = np.zeros(flat.size, dtype=bool)
    for group in np.unique(flat):
        mask[np.flatnonzero(flat == group)[0]] = True
    return mask.reshape(groups.shape)


def _evaluate_method(
    scene: np.ndarray,
    groups: np.ndarray | None,
    counts: Iterable[int],
    seeds: Iterable[int],
    threshold_psnr_db: float,
) -> MethodReceipt:
    median: dict[str, float] = {}
    success: dict[str, float] = {}
    first: int | None = None

    for count in counts:
        scores: list[float] = []
        for seed in seeds:
            mask = random_mask(int(count), int(seed))
            observed = np.zeros_like(scene)
            observed[mask] = scene[mask]
            if groups is None:
                recovered = spatial_harmonic_reconstruct(mask, observed)
            else:
                recovered = group_harmonic_reconstruct(mask, observed, groups)
            scores.append(psnr(scene, recovered))

        med = float(np.median(scores))
        frac = float(np.mean(np.asarray(scores) >= threshold_psnr_db))
        median[str(int(count))] = med
        success[str(int(count))] = frac
        if first is None and frac >= 0.90:
            first = int(count)

    return MethodReceipt(median, success, first)


def run_gate0(
    seeds: int = 48,
    counts: Iterable[int] = DEFAULT_COUNTS,
    threshold_psnr_db: float = 40.0,
) -> Gate0Receipt:
    labels = make_labels()
    scene = make_scene(labels)
    wrong = shuffled_groups(labels)
    seed_values = range(1000, 1000 + int(seeds))

    relation = _evaluate_method(scene, labels, counts, seed_values, threshold_psnr_db)
    spatial = _evaluate_method(scene, None, counts, seed_values, threshold_psnr_db)
    shuffled = _evaluate_method(scene, wrong, counts, seed_values, threshold_psnr_db)

    active_mask = active_component_mask(labels)
    observed = np.zeros_like(scene)
    observed[active_mask] = scene[active_mask]
    active = group_harmonic_reconstruct(active_mask, observed, labels)
    active_count = int(active_mask.sum())

    return Gate0Receipt(
        image_size=IMAGE_SIZE,
        pixels=N_PIXELS,
        objects=N_OBJECTS,
        threshold_psnr_db=float(threshold_psnr_db),
        active_relation_measurements=active_count,
        active_relation_fraction=active_count / N_PIXELS,
        active_relation_psnr_db=psnr(scene, active),
        random_relation=relation,
        spatial_lattice=spatial,
        shuffled_relation=shuffled,
    )


def receipt_to_dict(receipt: Gate0Receipt) -> dict:
    return asdict(receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("results/gate0_summary.json"))
    args = parser.parse_args()

    receipt = run_gate0(seeds=args.seeds)
    payload = receipt_to_dict(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
