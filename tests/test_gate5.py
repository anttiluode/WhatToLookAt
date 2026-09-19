from gate5_price_of_sparse_measurements import run_gate5


def _by_support(receipt):
    return {item.row_support: item for item in receipt.selected}


def test_gate5_extreme_measurement_sparsity_has_a_price():
    receipt = run_gate5(
        training_worlds=50,
        threshold_validation_worlds=25,
        policy_validation_worlds_per_level=20,
        test_worlds_per_level=24,
    )
    by_support = _by_support(receipt)

    # With the tested guide-row ceiling, touching one coordinate per row is
    # not enough to meet the end-to-end validation target.
    assert by_support[1].feasible is False

    # Moderately sparse operators should become feasible.
    assert by_support[32].feasible is True
    assert by_support[64].feasible is True


def test_gate5_sparse_operator_can_save_mixing_work():
    receipt = run_gate5(
        training_worlds=50,
        threshold_validation_worlds=25,
        policy_validation_worlds_per_level=20,
        test_worlds_per_level=24,
    )
    by_support = _by_support(receipt)

    sparse = by_support[64]
    dense = by_support[192]

    assert sparse.feasible and dense.feasible

    # The selected sparse guide should use fewer coordinate touches than the
    # selected dense guide while keeping strong held-out reconstruction.
    assert (
        sparse.guide_coordinate_touches_total
        < dense.guide_coordinate_touches_total
    )
    assert min(item.success_fraction for item in sparse.held_out) >= 0.90

    # The sparse and dense selected guides should require the same order of
    # scalar guide samples; the saving is in how much each row touches.
    assert sparse.guide_scalar_samples_total <= 2 * dense.guide_scalar_samples_total
