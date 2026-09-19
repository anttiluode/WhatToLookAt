from gate6_designed_vs_pruned import run_gate6


def _by_name(receipt):
    return {
        strategy.strategy: strategy
        for strategy in receipt.strategies
    }


def test_gate6_coverage_design_reaches_target_with_fewer_touches():
    receipt = run_gate6(
        training_worlds=60,
        validation_worlds_per_level=30,
        test_worlds_per_level=32,
    )
    strategies = _by_name(receipt)

    designed = strategies["coverage_designed"]
    pruned = strategies["posthoc_pruned"]
    dense = strategies["dense"]

    assert designed.feasible
    assert pruned.feasible
    assert dense.feasible

    assert designed.coordinate_coverage == 192
    assert designed.selected_guide_rows == 3
    assert pruned.selected_guide_rows >= 8

    assert (
        designed.guide_coordinate_touches_total
        < pruned.guide_coordinate_touches_total
    )
    assert (
        designed.guide_coordinate_touches_total
        < dense.guide_coordinate_touches_total
    )


def test_gate6_designed_sparse_holds_quality_on_untouched_worlds():
    receipt = run_gate6(
        training_worlds=60,
        validation_worlds_per_level=30,
        test_worlds_per_level=32,
    )
    designed = _by_name(receipt)["coverage_designed"]

    assert designed.feasible
    assert min(
        level.success_fraction
        for level in designed.held_out
    ) >= 0.95
