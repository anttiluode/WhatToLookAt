"""Gate 5: measure the price of a sparse sensing operator.

Gate 4 used a dense cheap projection: every guide measurement mixed all 192
high-resolution coordinates. This gate makes the guide rows sparse.

The relation world is deliberately support-recovery-like:
- four stable relation anchors;
- twelve non-anchor patches;
- a local innovation changes only 8 of 192 high-resolution coordinates;
- a guide row touches only d coordinates.

If a sparse row never overlaps the innovation support, no threshold can detect
that relation failure from that row. More guide rows increase the chance of
overlap, but they also increase scalar sample count.

For each row support d and guide-row count m:
1. train a residual threshold on mixed innovation worlds;
2. measure simple classification BA on a separate validation set;
3. run the actual adaptive reconstruction on another validation panel;
4. select the smallest m whose *minimum* validation success across innovation
   levels is at least 95%;
5. freeze that configuration and evaluate on untouched test worlds.

Two costs are reported separately:
- guide scalar samples = 16 * m;
- guide coordinate touches / multiply-add opportunities = 16 * m * d.

That is the empirical price curve: sparse rows can save mixing computation, but
may require more measurement rows before they reliably intersect a sparse
innovation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

N_OBJECTS = 4
PARTS_PER_OBJECT = 4
N_PARTS = N_OBJECTS * PARTS_PER_OBJECT
CONTENT_D = 192

INNOVATION_SUPPORT = 8
INNOVATION_AMPLITUDE = 0.35
BASE_NOISE = 0.002
QUALITY_DB = 40.0
LOCAL_MSE_LIMIT = 10.0 ** (-QUALITY_DB / 10.0)

ROW_SUPPORTS = (1, 2, 4, 8, 16, 32, 64, 96, 192)
GUIDE_ROW_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)
INNOVATION_LEVELS = (0, 2, 4, 6, 8)
VALIDATION_SUCCESS_TARGET = 0.95

GROUPS = np.repeat(
    np.arange(N_OBJECTS, dtype=np.int64),
    PARTS_PER_OBJECT,
)
ANCHORS = np.asarray([0, 4, 8, 12], dtype=np.int64)
NON_ANCHORS = np.asarray(
    [index for index in range(N_PARTS) if index not in set(ANCHORS.tolist())],
    dtype=np.int64,
)
FULL_SCAN_COST = N_PARTS * CONTENT_D


@dataclass(frozen=True)
class HeldOutReceipt:
    innovation_patches: int
    success_fraction: float
    mean_expensive_patch_reads: float
    mean_total_scalar_cost: float
    mean_total_cost_fraction_of_full_scan: float
    median_psnr_db: float


@dataclass(frozen=True)
class SparseOperatorReceipt:
    row_support: int
    feasible: bool
    selected_guide_rows: int | None
    best_tested_guide_rows: int | None
    min_validation_success: float
    validation_balanced_accuracy: float
    threshold: float | None
    guide_coordinate_touches_total: int
    guide_scalar_samples_total: int
    held_out: list[HeldOutReceipt]


@dataclass(frozen=True)
class Gate5Receipt:
    training_worlds: int
    threshold_validation_worlds: int
    policy_validation_worlds_per_level: int
    test_worlds_per_level: int
    signal_dimension: int
    innovation_support: int
    validation_min_success_target: float
    row_supports: list[int]
    guide_row_candidates: list[int]
    selected: list[SparseOperatorReceipt]


def make_base(seed: int = 222) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, size=(N_OBJECTS, CONTENT_D))


BASE_CONTENT = make_base()


def make_world(seed: int, innovation_patches: int) -> dict:
    rng = np.random.default_rng(seed)
    content = BASE_CONTENT[GROUPS].copy()
    content += rng.normal(scale=BASE_NOISE, size=content.shape)
    content = np.clip(content, 0.0, 1.0)

    innovations = np.zeros(N_PARTS, dtype=bool)
    if innovation_patches:
        chosen = rng.choice(
            NON_ANCHORS,
            int(innovation_patches),
            replace=False,
        )
        innovations[chosen] = True
        for patch in chosen:
            support = rng.choice(
                CONTENT_D,
                INNOVATION_SUPPORT,
                replace=False,
            )
            signs = rng.choice(
                np.asarray([-1.0, 1.0]),
                size=INNOVATION_SUPPORT,
            )
            content[patch, support] = np.clip(
                content[patch, support] + INNOVATION_AMPLITUDE * signs,
                0.0,
                1.0,
            )

    return {
        "content": content,
        "innovations": innovations,
    }


def sparse_projection(rows: int, row_support: int) -> np.ndarray:
    rng = np.random.default_rng(777 + 10000 * int(row_support) + int(rows))
    matrix = np.zeros((int(rows), CONTENT_D), dtype=np.float64)
    for row in range(int(rows)):
        support = rng.choice(
            CONTENT_D,
            int(row_support),
            replace=False,
        )
        matrix[row, support] = (
            rng.choice(np.asarray([-1.0, 1.0]), size=int(row_support))
            / np.sqrt(float(row_support))
        )
    return matrix


def residual_features(
    world: dict,
    projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    content = np.asarray(world["content"])
    guide = content @ projection.T

    score = np.zeros(N_PARTS, dtype=np.float64)
    needs_read = np.zeros(N_PARTS, dtype=bool)

    for group, anchor in enumerate(ANCHORS):
        members = np.flatnonzero(GROUPS == group)
        score[members] = np.linalg.norm(
            guide[members] - guide[int(anchor)][None, :],
            axis=1,
        )
        for patch in members:
            mse = float(
                np.mean(
                    (content[patch] - content[int(anchor)]) ** 2
                )
            )
            needs_read[patch] = mse > LOCAL_MSE_LIMIT

    return score, needs_read


def balanced_accuracy(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> float:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    positive = (
        float(np.mean(predicted[truth])) if np.any(truth) else 1.0
    )
    negative = (
        float(np.mean(~predicted[~truth])) if np.any(~truth) else 1.0
    )
    return 0.5 * (positive + negative)


def fit_threshold(
    worlds: list[dict],
    projection: np.ndarray,
) -> tuple[float, float]:
    values = []
    labels = []
    for world in worlds:
        score, need = residual_features(world, projection)
        values.append(score)
        labels.append(need)

    value = np.concatenate(values)
    truth = np.concatenate(labels)
    candidates = np.linspace(
        float(value.min()),
        float(value.max()),
        101,
    )

    best_threshold = float(candidates[0])
    best_score = -1.0
    for threshold in candidates:
        score = balanced_accuracy(
            truth,
            value >= threshold,
        )
        if score > best_score + 1e-12:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def threshold_accuracy(
    worlds: list[dict],
    projection: np.ndarray,
    threshold: float,
) -> float:
    values = []
    labels = []
    for world in worlds:
        score, need = residual_features(world, projection)
        values.append(score)
        labels.append(need)
    value = np.concatenate(values)
    truth = np.concatenate(labels)
    return balanced_accuracy(truth, value >= threshold)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(reference) - np.asarray(estimate)) ** 2))
    if mse <= 1e-14:
        return 120.0
    return float(10.0 * np.log10(1.0 / mse))


def adaptive_reconstruct(
    world: dict,
    projection: np.ndarray,
    threshold: float,
) -> tuple[float, int]:
    content = np.asarray(world["content"])
    score, _ = residual_features(world, projection)

    estimate = np.empty_like(content)
    reads: set[int] = set()

    for group, anchor in enumerate(ANCHORS):
        members = np.flatnonzero(GROUPS == group)
        anchor = int(anchor)
        reads.add(anchor)
        estimate[members] = content[anchor]

    for patch in np.flatnonzero(score >= threshold):
        reads.add(int(patch))
        estimate[patch] = content[patch]

    return psnr(content, estimate), len(reads)


def policy_validation_success(
    worlds_by_level: dict[int, list[dict]],
    projection: np.ndarray,
    threshold: float,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for level, worlds in worlds_by_level.items():
        scores = [
            adaptive_reconstruct(world, projection, threshold)[0]
            for world in worlds
        ]
        result[int(level)] = float(
            np.mean(np.asarray(scores) >= QUALITY_DB)
        )
    return result


def evaluate_test_level(
    worlds: list[dict],
    projection: np.ndarray,
    threshold: float,
    guide_rows: int,
    row_support: int,
) -> HeldOutReceipt:
    results = [
        adaptive_reconstruct(world, projection, threshold)
        for world in worlds
    ]
    guide_scalar_samples = N_PARTS * int(guide_rows)
    costs = [
        guide_scalar_samples + reads * CONTENT_D
        for _, reads in results
    ]
    return HeldOutReceipt(
        innovation_patches=int(
            np.sum(np.asarray(worlds[0]["innovations"], dtype=bool))
        ),
        success_fraction=float(
            np.mean([score >= QUALITY_DB for score, _ in results])
        ),
        mean_expensive_patch_reads=float(
            np.mean([reads for _, reads in results])
        ),
        mean_total_scalar_cost=float(np.mean(costs)),
        mean_total_cost_fraction_of_full_scan=float(
            np.mean(costs) / FULL_SCAN_COST
        ),
        median_psnr_db=float(
            np.median([score for score, _ in results])
        ),
    )


def run_gate5(
    training_worlds: int = 100,
    threshold_validation_worlds: int = 50,
    policy_validation_worlds_per_level: int = 40,
    test_worlds_per_level: int = 80,
) -> Gate5Receipt:
    train = [
        make_world(
            1000 + index,
            INNOVATION_LEVELS[index % len(INNOVATION_LEVELS)],
        )
        for index in range(int(training_worlds))
    ]
    threshold_validation = [
        make_world(
            3000 + index,
            INNOVATION_LEVELS[index % len(INNOVATION_LEVELS)],
        )
        for index in range(int(threshold_validation_worlds))
    ]
    policy_validation = {
        int(level): [
            make_world(
                4000 + 1000 * int(level) + index,
                int(level),
            )
            for index in range(int(policy_validation_worlds_per_level))
        ]
        for level in INNOVATION_LEVELS
    }
    test = {
        int(level): [
            make_world(
                10000 + 1000 * int(level) + index,
                int(level),
            )
            for index in range(int(test_worlds_per_level))
        ]
        for level in INNOVATION_LEVELS
    }

    selected: list[SparseOperatorReceipt] = []

    for row_support in ROW_SUPPORTS:
        candidates = []
        for guide_rows in GUIDE_ROW_CANDIDATES:
            projection = sparse_projection(guide_rows, row_support)
            threshold, training_ba = fit_threshold(train, projection)
            validation_ba = threshold_accuracy(
                threshold_validation,
                projection,
                threshold,
            )
            validation_success = policy_validation_success(
                policy_validation,
                projection,
                threshold,
            )
            candidates.append(
                {
                    "guide_rows": int(guide_rows),
                    "threshold": float(threshold),
                    "training_ba": float(training_ba),
                    "validation_ba": float(validation_ba),
                    "min_validation_success": float(
                        min(validation_success.values())
                    ),
                }
            )

        feasible = next(
            (
                candidate
                for candidate in candidates
                if candidate["min_validation_success"]
                >= VALIDATION_SUCCESS_TARGET
            ),
            None,
        )

        if feasible is None:
            best = max(
                candidates,
                key=lambda candidate: (
                    candidate["min_validation_success"],
                    candidate["validation_ba"],
                ),
            )
            guide_rows = int(best["guide_rows"])
            selected.append(
                SparseOperatorReceipt(
                    row_support=int(row_support),
                    feasible=False,
                    selected_guide_rows=None,
                    best_tested_guide_rows=guide_rows,
                    min_validation_success=float(
                        best["min_validation_success"]
                    ),
                    validation_balanced_accuracy=float(
                        best["validation_ba"]
                    ),
                    threshold=None,
                    guide_coordinate_touches_total=(
                        N_PARTS * guide_rows * int(row_support)
                    ),
                    guide_scalar_samples_total=N_PARTS * guide_rows,
                    held_out=[],
                )
            )
            continue

        guide_rows = int(feasible["guide_rows"])
        projection = sparse_projection(guide_rows, row_support)
        threshold = float(feasible["threshold"])
        held_out = [
            evaluate_test_level(
                test[int(level)],
                projection,
                threshold,
                guide_rows,
                int(row_support),
            )
            for level in INNOVATION_LEVELS
        ]
        selected.append(
            SparseOperatorReceipt(
                row_support=int(row_support),
                feasible=True,
                selected_guide_rows=guide_rows,
                best_tested_guide_rows=None,
                min_validation_success=float(
                    feasible["min_validation_success"]
                ),
                validation_balanced_accuracy=float(
                    feasible["validation_ba"]
                ),
                threshold=threshold,
                guide_coordinate_touches_total=(
                    N_PARTS * guide_rows * int(row_support)
                ),
                guide_scalar_samples_total=N_PARTS * guide_rows,
                held_out=held_out,
            )
        )

    return Gate5Receipt(
        training_worlds=int(training_worlds),
        threshold_validation_worlds=int(threshold_validation_worlds),
        policy_validation_worlds_per_level=int(
            policy_validation_worlds_per_level
        ),
        test_worlds_per_level=int(test_worlds_per_level),
        signal_dimension=CONTENT_D,
        innovation_support=INNOVATION_SUPPORT,
        validation_min_success_target=VALIDATION_SUCCESS_TARGET,
        row_supports=[int(value) for value in ROW_SUPPORTS],
        guide_row_candidates=[
            int(value) for value in GUIDE_ROW_CANDIDATES
        ],
        selected=selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-worlds", type=int, default=100)
    parser.add_argument("--threshold-validation-worlds", type=int, default=50)
    parser.add_argument("--policy-validation-worlds", type=int, default=40)
    parser.add_argument("--test-worlds", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate5_summary.json"),
    )
    args = parser.parse_args()
    receipt = run_gate5(
        training_worlds=args.train_worlds,
        threshold_validation_worlds=args.threshold_validation_worlds,
        policy_validation_worlds_per_level=args.policy_validation_worlds,
        test_worlds_per_level=args.test_worlds,
    )
    payload = asdict(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
