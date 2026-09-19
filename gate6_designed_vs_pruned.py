"""Gate 6: designed sparse sensing versus post-hoc sparsification.

Gate 5 showed that sparse sensing rows save fan-in work but can miss sparse
innovations. Gate 6 asks whether the sparse operator should be designed *as a
sparse operator* rather than produced by pruning a dense one after the fact.

The attacker is deliberately sharp: every changed patch modifies only one of
192 high-resolution coordinates. A sparse guide row touches 64 coordinates.

Four sensing families compete:
- coverage_designed: sparse rows maximize union coverage before repeating;
- posthoc_pruned: start from dense Gaussian rows, then keep only the 64 largest
  absolute coefficients in each row;
- random_sparse: independent random 64-coordinate supports;
- dense: unpruned Gaussian rows.

For each family, the number of guide rows m is swept over
1,2,3,4,6,8,12,16. Each (family,m) learns its own residual threshold on
training worlds. The smallest m whose *worst* validation innovation level
reaches 95% reconstruction success is frozen and evaluated on untouched test
worlds.

Two costs stay separate:
- guide scalar samples = 16 * m;
- coordinate touches / multiply-add opportunities = 16 * m * row_support.

This gate therefore asks whether support geometry itself is useful, not merely
whether sparse matrices are cheaper.
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

ROW_SUPPORT = 64
INNOVATION_SUPPORT = 1
INNOVATION_AMPLITUDE = 0.65
BASE_NOISE = 0.002
QUALITY_DB = 40.0
LOCAL_MSE_LIMIT = 10.0 ** (-QUALITY_DB / 10.0)

GUIDE_ROW_CANDIDATES = (1, 2, 3, 4, 6, 8, 12, 16)
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
class CandidateReceipt:
    guide_rows: int
    coordinate_coverage: int
    min_validation_success: float
    validation_success_by_level: dict[str, float]


@dataclass(frozen=True)
class StrategyReceipt:
    strategy: str
    row_support: int
    feasible: bool
    selected_guide_rows: int | None
    validation_min_success: float
    training_balanced_accuracy: float | None
    threshold: float | None
    coordinate_coverage: int
    guide_scalar_samples_total: int
    guide_coordinate_touches_total: int
    candidates: list[CandidateReceipt]
    held_out: list[HeldOutReceipt]


@dataclass(frozen=True)
class Gate6Receipt:
    training_worlds: int
    validation_worlds_per_level: int
    test_worlds_per_level: int
    signal_dimension: int
    innovation_support: int
    sparse_row_support: int
    validation_min_success_target: float
    guide_row_candidates: list[int]
    strategies: list[StrategyReceipt]


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
                content[patch, support]
                + INNOVATION_AMPLITUDE * signs,
                0.0,
                1.0,
            )

    return {
        "content": content,
        "innovations": innovations,
    }


def dense_projection(rows: int) -> np.ndarray:
    rng = np.random.default_rng(6060 + int(rows) * 17)
    matrix = rng.normal(size=(int(rows), CONTENT_D))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def posthoc_pruned_projection(rows: int) -> np.ndarray:
    dense = dense_projection(rows)
    sparse = np.zeros_like(dense)
    for row_index, row in enumerate(dense):
        support = np.argpartition(
            np.abs(row),
            -ROW_SUPPORT,
        )[-ROW_SUPPORT:]
        values = row[support]
        values = values / np.linalg.norm(values)
        sparse[row_index, support] = values
    return sparse


def random_sparse_projection(rows: int) -> np.ndarray:
    rng = np.random.default_rng(6061 + int(rows) * 17)
    matrix = np.zeros((int(rows), CONTENT_D), dtype=np.float64)
    for row in range(int(rows)):
        support = rng.choice(
            CONTENT_D,
            ROW_SUPPORT,
            replace=False,
        )
        matrix[row, support] = (
            rng.choice(
                np.asarray([-1.0, 1.0]),
                size=ROW_SUPPORT,
            )
            / np.sqrt(float(ROW_SUPPORT))
        )
    return matrix


def coverage_designed_projection(rows: int) -> np.ndarray:
    """Maximize coordinate union before allowing support repetition."""
    rng = np.random.default_rng(6062 + int(rows) * 17)
    matrix = np.zeros((int(rows), CONTENT_D), dtype=np.float64)
    permutation = rng.permutation(CONTENT_D)
    cursor = 0

    for row in range(int(rows)):
        if cursor < CONTENT_D:
            take = min(ROW_SUPPORT, CONTENT_D - cursor)
            support = list(permutation[cursor : cursor + take])
            cursor += take

            if take < ROW_SUPPORT:
                used = set(support)
                pool = np.asarray(
                    [
                        index
                        for index in range(CONTENT_D)
                        if index not in used
                    ],
                    dtype=np.int64,
                )
                support.extend(
                    rng.choice(
                        pool,
                        ROW_SUPPORT - take,
                        replace=False,
                    ).tolist()
                )
        else:
            support = rng.choice(
                CONTENT_D,
                ROW_SUPPORT,
                replace=False,
            ).tolist()

        support_array = np.asarray(support, dtype=np.int64)
        matrix[row, support_array] = (
            rng.choice(
                np.asarray([-1.0, 1.0]),
                size=ROW_SUPPORT,
            )
            / np.sqrt(float(ROW_SUPPORT))
        )

    return matrix


def projection_for(strategy: str, rows: int) -> np.ndarray:
    if strategy == "coverage_designed":
        return coverage_designed_projection(rows)
    if strategy == "posthoc_pruned":
        return posthoc_pruned_projection(rows)
    if strategy == "random_sparse":
        return random_sparse_projection(rows)
    if strategy == "dense":
        return dense_projection(rows)
    raise ValueError(strategy)


def row_support_for(strategy: str) -> int:
    return CONTENT_D if strategy == "dense" else ROW_SUPPORT


def coordinate_coverage(projection: np.ndarray) -> int:
    return int(np.sum(np.any(projection != 0.0, axis=0)))


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
        anchor = int(anchor)
        score[members] = np.linalg.norm(
            guide[members] - guide[anchor][None, :],
            axis=1,
        )
        for patch in members:
            mse = float(
                np.mean(
                    (content[patch] - content[anchor]) ** 2
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
    thresholds = np.linspace(
        float(value.min()),
        float(value.max()),
        201,
    )

    best_threshold = float(thresholds[0])
    best_score = -1.0
    for threshold in thresholds:
        score = balanced_accuracy(
            truth,
            value >= threshold,
        )
        if score > best_score + 1e-12:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(
        np.mean(
            (np.asarray(reference) - np.asarray(estimate)) ** 2
        )
    )
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
        patch = int(patch)
        reads.add(patch)
        estimate[patch] = content[patch]

    return psnr(content, estimate), len(reads)


def policy_success(
    worlds: list[dict],
    projection: np.ndarray,
    threshold: float,
) -> float:
    scores = [
        adaptive_reconstruct(
            world,
            projection,
            threshold,
        )[0]
        for world in worlds
    ]
    return float(
        np.mean(
            np.asarray(scores) >= QUALITY_DB
        )
    )


def evaluate_test_level(
    worlds: list[dict],
    projection: np.ndarray,
    threshold: float,
    strategy: str,
    guide_rows: int,
) -> HeldOutReceipt:
    results = [
        adaptive_reconstruct(
            world,
            projection,
            threshold,
        )
        for world in worlds
    ]

    guide_scalar_samples = N_PARTS * int(guide_rows)
    costs = [
        guide_scalar_samples + reads * CONTENT_D
        for _, reads in results
    ]

    return HeldOutReceipt(
        innovation_patches=int(
            np.sum(
                np.asarray(
                    worlds[0]["innovations"],
                    dtype=bool,
                )
            )
        ),
        success_fraction=float(
            np.mean(
                [score >= QUALITY_DB for score, _ in results]
            )
        ),
        mean_expensive_patch_reads=float(
            np.mean(
                [reads for _, reads in results]
            )
        ),
        mean_total_scalar_cost=float(np.mean(costs)),
        mean_total_cost_fraction_of_full_scan=float(
            np.mean(costs) / FULL_SCAN_COST
        ),
        median_psnr_db=float(
            np.median(
                [score for score, _ in results]
            )
        ),
    )


def run_strategy(
    strategy: str,
    train: list[dict],
    validation: dict[int, list[dict]],
    test: dict[int, list[dict]],
) -> StrategyReceipt:
    candidate_receipts: list[CandidateReceipt] = []
    selected = None

    for guide_rows in GUIDE_ROW_CANDIDATES:
        projection = projection_for(strategy, guide_rows)
        threshold, training_ba = fit_threshold(
            train,
            projection,
        )
        success_by_level = {
            int(level): policy_success(
                validation[int(level)],
                projection,
                threshold,
            )
            for level in INNOVATION_LEVELS
        }
        minimum = float(min(success_by_level.values()))

        candidate_receipts.append(
            CandidateReceipt(
                guide_rows=int(guide_rows),
                coordinate_coverage=coordinate_coverage(
                    projection
                ),
                min_validation_success=minimum,
                validation_success_by_level={
                    str(level): float(value)
                    for level, value in success_by_level.items()
                },
            )
        )

        if (
            selected is None
            and minimum >= VALIDATION_SUCCESS_TARGET
        ):
            selected = {
                "guide_rows": int(guide_rows),
                "projection": projection,
                "threshold": float(threshold),
                "training_ba": float(training_ba),
                "min_validation_success": minimum,
            }

    if selected is None:
        best = max(
            candidate_receipts,
            key=lambda item: (
                item.min_validation_success,
                -item.guide_rows,
            ),
        )
        support = row_support_for(strategy)
        return StrategyReceipt(
            strategy=strategy,
            row_support=int(support),
            feasible=False,
            selected_guide_rows=None,
            validation_min_success=float(
                best.min_validation_success
            ),
            training_balanced_accuracy=None,
            threshold=None,
            coordinate_coverage=int(
                best.coordinate_coverage
            ),
            guide_scalar_samples_total=(
                N_PARTS * best.guide_rows
            ),
            guide_coordinate_touches_total=(
                N_PARTS * best.guide_rows * support
            ),
            candidates=candidate_receipts,
            held_out=[],
        )

    guide_rows = int(selected["guide_rows"])
    projection = selected["projection"]
    support = row_support_for(strategy)
    held_out = [
        evaluate_test_level(
            test[int(level)],
            projection,
            float(selected["threshold"]),
            strategy,
            guide_rows,
        )
        for level in INNOVATION_LEVELS
    ]

    return StrategyReceipt(
        strategy=strategy,
        row_support=int(support),
        feasible=True,
        selected_guide_rows=guide_rows,
        validation_min_success=float(
            selected["min_validation_success"]
        ),
        training_balanced_accuracy=float(
            selected["training_ba"]
        ),
        threshold=float(selected["threshold"]),
        coordinate_coverage=coordinate_coverage(
            projection
        ),
        guide_scalar_samples_total=N_PARTS * guide_rows,
        guide_coordinate_touches_total=(
            N_PARTS * guide_rows * support
        ),
        candidates=candidate_receipts,
        held_out=held_out,
    )


def run_gate6(
    training_worlds: int = 150,
    validation_worlds_per_level: int = 80,
    test_worlds_per_level: int = 120,
) -> Gate6Receipt:
    train = [
        make_world(
            11000 + index,
            INNOVATION_LEVELS[
                index % len(INNOVATION_LEVELS)
            ],
        )
        for index in range(int(training_worlds))
    ]

    validation = {
        int(level): [
            make_world(
                21000 + 1000 * int(level) + index,
                int(level),
            )
            for index in range(
                int(validation_worlds_per_level)
            )
        ]
        for level in INNOVATION_LEVELS
    }

    test = {
        int(level): [
            make_world(
                41000 + 1000 * int(level) + index,
                int(level),
            )
            for index in range(
                int(test_worlds_per_level)
            )
        ]
        for level in INNOVATION_LEVELS
    }

    strategies = [
        run_strategy(
            strategy,
            train,
            validation,
            test,
        )
        for strategy in (
            "coverage_designed",
            "posthoc_pruned",
            "random_sparse",
            "dense",
        )
    ]

    return Gate6Receipt(
        training_worlds=int(training_worlds),
        validation_worlds_per_level=int(
            validation_worlds_per_level
        ),
        test_worlds_per_level=int(
            test_worlds_per_level
        ),
        signal_dimension=CONTENT_D,
        innovation_support=INNOVATION_SUPPORT,
        sparse_row_support=ROW_SUPPORT,
        validation_min_success_target=(
            VALIDATION_SUCCESS_TARGET
        ),
        guide_row_candidates=[
            int(value)
            for value in GUIDE_ROW_CANDIDATES
        ],
        strategies=strategies,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-worlds",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--validation-worlds",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--test-worlds",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate6_summary.json"),
    )
    args = parser.parse_args()

    receipt = run_gate6(
        training_worlds=args.train_worlds,
        validation_worlds_per_level=args.validation_worlds,
        test_worlds_per_level=args.test_worlds,
    )
    payload = asdict(receipt)
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
