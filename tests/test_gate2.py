from gate2_confidence_budget import run_gate2


def test_gate2_confidence_buys_local_measurements():
    receipt = run_gate2(training_worlds=40, test_worlds_per_level=24)

    assert 0.4 <= receipt.learned_confidence_threshold <= 0.65
    assert receipt.training_balanced_accuracy >= 0.95

    # No ambiguity: adaptive sensing should stay at the cheap four-measurement budget.
    assert receipt.confidence_adaptive[0].success_fraction == 1.0
    assert receipt.confidence_adaptive[0].mean_measurements <= 4.5

    # With correspondence errors, blind trust is cheap and wrong.
    assert receipt.always_trust[2].mean_measurements == 4.0
    assert receipt.always_trust[2].success_fraction <= 0.10

    # Global caution is safe but always pays for every patch.
    assert receipt.global_cautious[2].success_fraction == 1.0
    assert receipt.global_cautious[2].mean_measurements == 16.0

    # Local confidence should recover most worlds for substantially less cost.
    assert receipt.confidence_adaptive[2].success_fraction >= 0.85
    assert receipt.confidence_adaptive[2].mean_measurements < 12.0

    # Same number of low-confidence flags at the wrong locations should not work.
    assert receipt.shuffled_confidence[2].success_fraction < 0.25
