from gate4_relation_residual_budget import FULL_SCAN_COST, run_gate4


def test_gate4_guide_residual_predicts_relation_mismatch():
    receipt = run_gate4(
        training_worlds=50,
        validation_worlds=25,
        test_worlds_per_level=24,
    )
    assert receipt.training_balanced_accuracy >= 0.95
    assert receipt.validation_balanced_accuracy >= 0.95


def test_gate4_residual_buys_only_needed_expensive_reads():
    receipt = run_gate4(
        training_worlds=50,
        validation_worlds=25,
        test_worlds_per_level=24,
    )

    clean = receipt.residual_adaptive[0]
    medium = receipt.residual_adaptive[2]
    hard = receipt.residual_adaptive[4]

    assert clean.success_fraction == 1.0
    assert clean.mean_total_scalar_cost < 0.35 * FULL_SCAN_COST

    # One fixed read per relation is no longer enough once local innovations exist.
    assert receipt.fixed_one_per_relation[1].success_fraction == 0.0

    # Residual-guided sensing should keep the quality target while remaining
    # below a full scan even when half the patches contain large innovations.
    assert medium.success_fraction >= 0.95
    assert hard.success_fraction >= 0.95
    assert hard.mean_total_scalar_cost < FULL_SCAN_COST

    # Same number of extra expensive reads at random locations must fail.
    assert receipt.equal_cost_random_extra[2].success_fraction < 0.25
    assert receipt.equal_cost_random_extra[4].success_fraction < 0.25

    # Adaptive cost should stay close to the oracle residual policy.
    assert (
        hard.mean_total_scalar_cost
        - receipt.oracle_residual[4].mean_total_scalar_cost
        <= 0.02 * FULL_SCAN_COST
    )
