"""Gate 8: buy measurement diversity deliberately.

Gate 7 showed that full coordinate coverage is not enough: a three-row
coverage-designed sparse sensor touches all 192 coordinates but signed
multi-coordinate innovations can cancel inside one row.

Gate 8 asks whether sparse overlap can be designed explicitly rather than paid
for with twelve generic rows.

Structured regular designs use 64-wide sparse rows arranged in partition
layers:
- degree 1: 3 rows, every coordinate appears once;
- degree 2: 6 rows, every coordinate appears twice;
- degree 3: 9 rows, every coordinate appears three times;
- degree 4: 12 rows, every coordinate appears four times.

Each layer is an independent partition of the 192 coordinates into three
64-coordinate rows with random +/- coefficients. Thus coordinate degree and
total touch cost are exact.

Thresholds are trained on a mixture of innovation supports
K = 1,2,4,8,16,32,64. A separate validation panel selects the smallest
structured degree whose worst support/patch-count reconstruction success reaches
95%. That choice is then frozen for held-out testing.

At the selected touch budget, three controls compete:
- independent random sparse rows;
- post-hoc top-weight pruning of dense Gaussian rows;
- dense Gaussian rows with the same total coordinate-touch count.

The older robust 12-row sparse controls are also retained as a cost ceiling.

This gate therefore asks a narrow engineering question:
can deliberate sparse overlap buy the diversity Gate 7 needed with fewer
measurement interactions than simply adding more generic sparse rows?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gate6_designed_vs_pruned import (
    CONTENT_D,
    INNOVATION_LEVELS,
    N_PARTS,
    QUALITY_DB,
    adaptive_reconstruct,
    dense_projection,
    posthoc_pruned_projection,
    random_sparse_projection,
    fit_threshold,
)
from gate7_coverage_not_enough import (
    INNOVATION_SUPPORTS,
    make_world,
)

ROW_SUPPORT = 64
VALIDATION_SUCCESS_TARGET = 0.95
STRUCTURED_DEGREES = (1, 2, 3, 4)


@dataclass(frozen=True)
class SupportReceipt:
    innovation_support: int
    min_success_across_patch_levels: float
    success_by_innovation_patch_count: dict[str, float]


@dataclass(frozen=True)
class SensorReceipt:
    name: str
    family: str
    guide_rows: int
    row_support: int
    coordinate_degree: int | None
    coordinate_coverage: int
    guide_scalar_samples_total: int
    guide_coordinate_touches_total: int
    threshold: float
    training_balanced_accuracy: float
    validation_min_success: float
    validation_by_support: dict[str, float]
    held_out_min_success: float
    held_out_by_support: list[SupportReceipt]


@dataclass(frozen=True)
class Gate8Receipt:
    training_worlds: int
    validation_worlds_per_cell: int
    test_worlds_per_cell: int
    signal_dimension: int
    row_support: int
    validation_success_target: float
    innovation_supports: list[int]
    structured_degrees: list[int]
    selected_structured_degree: int | None
    sensors: list[SensorReceipt]


def regular_overlap_projection(
    degree: int,
    seed: int = 8800,
) -> np.ndarray:
    """Return a row-regular sparse code with exact coordinate degree.

    Each degree layer is an independent partition of all 192 coordinates into
    three rows of width 64. Every coordinate therefore appears exactly once per
    layer and exactly degree times overall.
    """
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be positive")

    rows = 3 * degree
    rng = np.random.default_rng(int(seed) + degree)
    matrix = np.zeros((rows, CONTENT_D), dtype=np.float64)

    for layer in range(degree):
        permutation = rng.permutation(CONTENT_D)
        for local_row in range(3):
            row = layer * 3 + local_row
            support = permutation[
                local_row * ROW_SUPPORT :
                (local_row + 1) * ROW_SUPPORT
            ]
            matrix[row, support] = (
                rng.choice(
                    np.asarray([-1.0, 1.0]),
                    size=ROW_SUPPORT,
                )
                / np.sqrt(float(ROW_SUPPORT))
            )

    return matrix


def coordinate_coverage(
    projection: np.ndarray,
) -> int:
    return int(
        np.sum(
            np.any(
                np.asarray(projection) != 0.0,
                axis=0,
            )
        )
    )


def train_worlds(
    count: int,
) -> list[dict]:
    worlds: list[dict] = []
    supports = tuple(
        int(value)
        for value in INNOVATION_SUPPORTS
    )
    levels = tuple(
        int(value)
        for value in INNOVATION_LEVELS
    )

    for index in range(int(count)):
        support = supports[
            index % len(supports)
        ]
        level = levels[
            (index // len(supports))
            % len(levels)
        ]
        worlds.append(
            make_world(
                31000 + index,
                level,
                support,
            )
        )

    return worlds


def cell_worlds(
    base_seed: int,
    support: int,
    innovation_patches: int,
    count: int,
) -> list[dict]:
    return [
        make_world(
            int(base_seed)
            + 10000 * int(support)
            + 1000 * int(innovation_patches)
            + index,
            int(innovation_patches),
            int(support),
        )
        for index in range(int(count))
    ]


def evaluate_support(
    projection: np.ndarray,
    threshold: float,
    support: int,
    count: int,
    base_seed: int,
) -> SupportReceipt:
    success_by_level: dict[str, float] = {}

    for innovation_patches in INNOVATION_LEVELS:
        worlds = cell_worlds(
            base_seed,
            int(support),
            int(innovation_patches),
            int(count),
        )
        results = [
            adaptive_reconstruct(
                world,
                projection,
                threshold,
            )[0]
            for world in worlds
        ]
        success_by_level[
            str(int(innovation_patches))
        ] = float(
            np.mean(
                np.asarray(results)
                >= QUALITY_DB
            )
        )

    return SupportReceipt(
        innovation_support=int(support),
        min_success_across_patch_levels=float(
            min(success_by_level.values())
        ),
        success_by_innovation_patch_count=(
            success_by_level
        ),
    )


def make_sensor_receipt(
    name: str,
    family: str,
    projection: np.ndarray,
    row_support: int,
    coordinate_degree: int | None,
    training: list[dict],
    validation_worlds_per_cell: int,
    test_worlds_per_cell: int,
) -> SensorReceipt:
    threshold, training_ba = fit_threshold(
        training,
        projection,
    )

    validation = [
        evaluate_support(
            projection,
            float(threshold),
            int(support),
            int(validation_worlds_per_cell),
            51000,
        )
        for support in INNOVATION_SUPPORTS
    ]
    held_out = [
        evaluate_support(
            projection,
            float(threshold),
            int(support),
            int(test_worlds_per_cell),
            91000,
        )
        for support in INNOVATION_SUPPORTS
    ]

    rows = int(projection.shape[0])
    return SensorReceipt(
        name=name,
        family=family,
        guide_rows=rows,
        row_support=int(row_support),
        coordinate_degree=(
            None
            if coordinate_degree is None
            else int(coordinate_degree)
        ),
        coordinate_coverage=coordinate_coverage(
            projection
        ),
        guide_scalar_samples_total=(
            N_PARTS * rows
        ),
        guide_coordinate_touches_total=(
            N_PARTS
            * int(np.count_nonzero(projection))
        ),
        threshold=float(threshold),
        training_balanced_accuracy=float(
            training_ba
        ),
        validation_min_success=float(
            min(
                receipt.min_success_across_patch_levels
                for receipt in validation
            )
        ),
        validation_by_support={
            str(receipt.innovation_support): float(
                receipt.min_success_across_patch_levels
            )
            for receipt in validation
        },
        held_out_min_success=float(
            min(
                receipt.min_success_across_patch_levels
                for receipt in held_out
            )
        ),
        held_out_by_support=held_out,
    )


def run_gate8(
    training_world_count: int = 210,
    validation_worlds_per_cell: int = 80,
    test_worlds_per_cell: int = 120,
) -> Gate8Receipt:
    training = train_worlds(
        int(training_world_count)
    )

    structured: list[SensorReceipt] = []
    for degree in STRUCTURED_DEGREES:
        projection = regular_overlap_projection(
            int(degree)
        )
        structured.append(
            make_sensor_receipt(
                name=f"regular_degree_{int(degree)}",
                family="structured_overlap",
                projection=projection,
                row_support=ROW_SUPPORT,
                coordinate_degree=int(degree),
                training=training,
                validation_worlds_per_cell=(
                    validation_worlds_per_cell
                ),
                test_worlds_per_cell=(
                    test_worlds_per_cell
                ),
            )
        )

    passing = [
        sensor
        for sensor in structured
        if sensor.validation_min_success
        >= VALIDATION_SUCCESS_TARGET
    ]
    selected_degree = (
        min(
            int(sensor.coordinate_degree)
            for sensor in passing
            if sensor.coordinate_degree is not None
        )
        if passing
        else None
    )

    controls: list[SensorReceipt] = []
    for name, family, projection, row_support in (
        (
            "random_sparse_9",
            "same_touch_random",
            random_sparse_projection(9),
            ROW_SUPPORT,
        ),
        (
            "posthoc_pruned_9",
            "same_touch_posthoc",
            posthoc_pruned_projection(9),
            ROW_SUPPORT,
        ),
        (
            "dense_3",
            "same_touch_dense",
            dense_projection(3),
            CONTENT_D,
        ),
        (
            "random_sparse_12",
            "robust_12row_ceiling",
            random_sparse_projection(12),
            ROW_SUPPORT,
        ),
        (
            "posthoc_pruned_12",
            "robust_12row_ceiling",
            posthoc_pruned_projection(12),
            ROW_SUPPORT,
        ),
    ):
        controls.append(
            make_sensor_receipt(
                name=name,
                family=family,
                projection=projection,
                row_support=int(row_support),
                coordinate_degree=None,
                training=training,
                validation_worlds_per_cell=(
                    validation_worlds_per_cell
                ),
                test_worlds_per_cell=(
                    test_worlds_per_cell
                ),
            )
        )

    return Gate8Receipt(
        training_worlds=int(
            training_world_count
        ),
        validation_worlds_per_cell=int(
            validation_worlds_per_cell
        ),
        test_worlds_per_cell=int(
            test_worlds_per_cell
        ),
        signal_dimension=CONTENT_D,
        row_support=ROW_SUPPORT,
        validation_success_target=(
            VALIDATION_SUCCESS_TARGET
        ),
        innovation_supports=[
            int(value)
            for value in INNOVATION_SUPPORTS
        ],
        structured_degrees=[
            int(value)
            for value in STRUCTURED_DEGREES
        ],
        selected_structured_degree=(
            selected_degree
        ),
        sensors=structured + controls,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--train-worlds",
        type=int,
        default=210,
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
        default=Path(
            "results/gate8_summary.json"
        ),
    )
    args = parser.parse_args()

    receipt = run_gate8(
        training_world_count=args.train_worlds,
        validation_worlds_per_cell=(
            args.validation_worlds
        ),
        test_worlds_per_cell=(
            args.test_worlds
        ),
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
