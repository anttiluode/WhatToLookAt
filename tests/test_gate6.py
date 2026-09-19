from gate6_posthoc_sparsification import run_gate6


def _by_support(receipt):
    return {item.row_support: item for item in receipt.results}


def test_gate6_designed_sparse_recovery_can_work():
    receipt = run_gate6(trials=60)
    by_support = _by_support(receipt)

    assert by_support[64].designed_sparse.mean_support_recall >= 0.95
    assert by_support[96].designed_sparse.exact_support_fraction >= 0.90


def test_gate6_posthoc_sparsification_is_not_equivalent():
    receipt = run_gate6(trials=60)
    by_support = _by_support(receipt)

    # Same sparse operator works when it generated the response, but not when
    # it is substituted after a dense measurement has already been made.
    assert (
        by_support[64].designed_sparse.mean_support_recall
        - by_support[64].posthoc_sparse.mean_support_recall
        >= 0.50
    )
    assert by_support[64].posthoc_sparse.exact_support_fraction <= 0.05

    # At density 1 the distinction disappears, as it should.
    assert (
        abs(
            by_support[192].dense.mean_support_recall
            - by_support[192].posthoc_sparse.mean_support_recall
        )
        < 1e-12
    )
