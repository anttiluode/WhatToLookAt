"""Gate 7: coverage is not enough once innovations have internal structure.

Gate 6 earned a strong but narrow result: in a one-coordinate innovation world,
three 64-wide sparse rows can deliberately cover all 192 coordinates and match
the post-hoc-pruned quality frontier with four times fewer coordinate touches.

Gate 7 freezes those Gate-6-selected sensing geometries and their K=1-trained
thresholds, then changes only one thing: how many coordinates inside each
innovation change.

Innovation support K is swept over:
    1, 2, 4, 8, 16, 32, 64

Total innovation energy is held approximately constant by scaling per-coordinate
amplitude as 0.65 / sqrt(K). No sensing geometry is redesigned and no threshold
is recalibrated after K=1.

This is the attacker Gate 6 explicitly invited. If support coverage is the whole
mechanism, the designed sparse sensor should remain competitive or improve as
innovations become more diffuse. If it fails, then mere coverage is not enough:
multiple changed coordinates can cancel inside the same sparse measurement, and
the sensor needs measurement diversity / overlapping codes / incoherence rather
than one-hit coverage alone.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gate6_designed_vs_pruned import (
    ANCHORS,
    BASE_CONTENT,
    BASE_NOISE,
    CONTENT_D,
    FULL_SCAN_COST,
    GROUPS,
    INNOVATION_LEVELS,
    NON_ANCHORS,
    N_PARTS,
    QUALITY_DB,
    adaptive_reconstruct,
    coordinate_coverage,
    fit_threshold,
    projection_for,
    row_support_for,
)

INNOVATION_SUPPORTS = (1, 2, 4, 8, 16, 32, 64)
BASE_TOTAL_AMPLITUDE = 0.65

SELECTED_ROWS = {
    "coverage_designed": 3,
    "posthoc_pruned": 12,
    "random_sparse": 12,
    "dense": 2,
}


@dataclass(frozen=True)
class SupportReceipt:
    innovation_support: int
    min_success_across_patch_levels: float
    success_by_innovation_patch_count: dict[str, float]
    mean_reads_by_innovation_patch_count: dict[str, float]


@dataclass(frozen=True)
class FrozenSensorReceipt:
    strategy: str
    guide_rows: int
    row_support: int
    coordinate_coverage: int
    threshold_trained_at_support_1: float
    training_balanced_accuracy: float
    guide_coordinate_touches_total: int
    supports: list[SupportReceipt]


@dataclass(frozen=True)
class Gate7Receipt:
    training_worlds: int
    test_worlds_per_cell: int
    signal_dimension: int
    innovation_supports: list[int]
    innovation_patch_levels: list[int]
    total_innovation_energy_reference_amplitude: float
    sensors: list[FrozenSensorReceipt]


def make_world(
    seed: int,
    innovation_patches: int,
    innovation_support: int,
) -> dict:
    rng = np.random.default_rng(seed)

    content = BASE_CONTENT[GROUPS].copy()
    content += rng.normal(
        scale=BASE_NOISE,
        size=content.shape,
    )
    content = np.clip(content, 0.0, 1.0)

    innovations = np.zeros(N_PARTS, dtype=bool)
    if innovation_patches:
        chosen = rng.choice(
            NON_ANCHORS,
            int(innovation_patches),
            replace=False,
        )
        innovations[chosen] = True

        amplitude = (
            BASE_TOTAL_AMPLITUDE
            / np.sqrt(float(innovation_support))
        )

        for patch in chosen:
            support = rng.choice(
                CONTENT_D,
                int(innovation_support),
                replace=False,
            )
            signs = rng.choice(
                np.asarray([-1.0, 1.0]),
                size=int(innovation_support),
            )
            content[patch, support] = np.clip(
                content[patch, support]
                + amplitude * signs,
                0.0,
                1.0,
            )

    return {
        "content": content,
        "innovations": innovations,
    }


def evaluate_support(
    strategy: str,
    rows: int,
    projection: np.ndarray,
    threshold: float,
    support_size: int,
    test_worlds_per_cell: int,
) -> SupportReceipt:
    success: dict[str, float] = {}
    reads: dict[str, float] = {}

    for innovation_patches in INNOVATION_LEVELS:
        worlds = [
            make_world(
                71000
                + 10000 * int(support_size)
                + 1000 * int(innovation_patches)
                + index,
                int(innovation_patches),
                int(support_size),
            )
            for index in range(
                int(test_worlds_per_cell)
            )
        ]

        results = [
            adaptive_reconstruct(
                world,
                projection,
                threshold,
            )
            for world in worlds
        ]

        success[str(int(innovation_patches))] = float(
            np.mean(
                [
                    score >= QUALITY_DB
                    for score, _ in results
                ]
            )
        )
        reads[str(int(innovation_patches))] = float(
            np.mean(
                [
                    read_count
                    for _, read_count in results
                ]
            )
        )

    return SupportReceipt(
        innovation_support=int(support_size),
        min_success_across_patch_levels=float(
            min(success.values())
        ),
        success_by_innovation_patch_count=success,
        mean_reads_by_innovation_patch_count=reads,
    )


def run_gate7(
    training_worlds: int = 150,
    test_worlds_per_cell: int = 120,
) -> Gate7Receipt:
    # Recreate the Gate-6 K=1 calibration set exactly, then freeze thresholds.
    train = [
        make_world(
            11000 + index,
            int(
                INNOVATION_LEVELS[
                    index % len(INNOVATION_LEVELS)
                ]
            ),
            1,
        )
        for index in range(int(training_worlds))
    ]

    sensors: list[FrozenSensorReceipt] = []

    for strategy, rows in SELECTED_ROWS.items():
        projection = projection_for(
            strategy,
            int(rows),
        )
        threshold, training_ba = fit_threshold(
            train,
            projection,
        )
        support = row_support_for(strategy)

        receipts = [
            evaluate_support(
                strategy,
                int(rows),
                projection,
                float(threshold),
                int(support_size),
                int(test_worlds_per_cell),
            )
            for support_size in INNOVATION_SUPPORTS
        ]

        sensors.append(
            FrozenSensorReceipt(
                strategy=strategy,
                guide_rows=int(rows),
                row_support=int(support),
                coordinate_coverage=coordinate_coverage(
                    projection
                ),
                threshold_trained_at_support_1=float(
                    threshold
                ),
                training_balanced_accuracy=float(
                    training_ba
                ),
                guide_coordinate_touches_total=(
                    N_PARTS * int(rows) * int(support)
                ),
                supports=receipts,
            )
        )

    return Gate7Receipt(
        training_worlds=int(training_worlds),
        test_worlds_per_cell=int(
            test_worlds_per_cell
        ),
        signal_dimension=CONTENT_D,
        innovation_supports=[
            int(value)
            for value in INNOVATION_SUPPORTS
        ],
        innovation_patch_levels=[
            int(value)
            for value in INNOVATION_LEVELS
        ],
        total_innovation_energy_reference_amplitude=(
            BASE_TOTAL_AMPLITUDE
        ),
        sensors=sensors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-worlds",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--test-worlds",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate7_summary.json"),
    )
    args = parser.parse_args()

    receipt = run_gate7(
        training_worlds=args.train_worlds,
        test_worlds_per_cell=args.test_worlds,
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
