from gate3_image_confidence import FULL_SCAN_COST, run_gate3


def test_gate3_selects_an_image_derived_confidence_cue():
    receipt = run_gate3(
        training_worlds=40,
        validation_worlds=24,
        test_worlds_per_level=24,
    )

    assert receipt.selected_cue in {"margin", "neg_error", "cycle"}
    selected = next(
        cue for cue in receipt.candidate_cues if cue.name == receipt.selected_cue
    )
    assert selected.validation_balanced_accuracy >= 0.90


def test_gate3_image_confidence_localizes_extra_sensing():
    receipt = run_gate3(
        training_worlds=40,
        validation_worlds=24,
        test_worlds_per_level=24,
    )

    stable = receipt.image_confidence_adaptive[0]
    medium = receipt.image_confidence_adaptive[2]
    hard = receipt.image_confidence_adaptive[3]

    assert stable.success_fraction == 1.0
    assert stable.mean_total_scalar_cost < 0.35 * FULL_SCAN_COST

    # Blind trust stays cheap but fails under real image-matching ambiguity.
    assert receipt.always_trust[2].success_fraction <= 0.10

    # Image-derived confidence should keep most hard worlds correct while
    # remaining below a full high-resolution scan.
    assert medium.success_fraction >= 0.85
    assert hard.success_fraction >= 0.85
    assert hard.mean_total_scalar_cost < FULL_SCAN_COST

    # Same confidence values on the wrong patches must not earn the mechanism.
    assert receipt.shuffled_confidence[2].success_fraction < 0.25
    assert receipt.shuffled_confidence[3].success_fraction < 0.25
