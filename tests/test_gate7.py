from gate7_coverage_not_enough import run_gate7


def _sensor(receipt, name):
    return next(
        sensor
        for sensor in receipt.sensors
        if sensor.strategy == name
    )


def _support(sensor, support):
    return next(
        receipt
        for receipt in sensor.supports
        if receipt.innovation_support == support
    )


def test_gate7_reproduces_gate6_one_sparse_advantage():
    receipt = run_gate7(
        training_worlds=60,
        test_worlds_per_cell=24,
    )
    designed = _sensor(
        receipt,
        "coverage_designed",
    )
    one_sparse = _support(designed, 1)

    assert one_sparse.min_success_across_patch_levels >= 0.95


def test_gate7_coverage_only_breaks_for_multi_coordinate_change():
    receipt = run_gate7(
        training_worlds=60,
        test_worlds_per_cell=24,
    )
    designed = _sensor(
        receipt,
        "coverage_designed",
    )
    pruned = _sensor(
        receipt,
        "posthoc_pruned",
    )

    designed_two = _support(designed, 2)
    pruned_two = _support(pruned, 2)

    assert designed_two.min_success_across_patch_levels < 0.90
    assert pruned_two.min_success_across_patch_levels >= 0.95


def test_gate7_exposes_measurement_diversity_not_missing_coverage():
    receipt = run_gate7(
        training_worlds=60,
        test_worlds_per_cell=24,
    )
    designed = _sensor(
        receipt,
        "coverage_designed",
    )

    assert designed.coordinate_coverage == 192
    assert _support(
        designed,
        2,
    ).min_success_across_patch_levels < 0.90
