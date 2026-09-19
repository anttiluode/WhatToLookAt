from gate8_coded_overlap import run_gate8


def _sensor(receipt, name):
    return next(
        sensor
        for sensor in receipt.sensors
        if sensor.name == name
    )


def test_gate8_coded_overlap_receipt():
    # One shared CI-sized receipt keeps this gate fast while checking selection,
    # matched-touch attackers, held-out quality, and cost.
    receipt = run_gate8(
        training_world_count=84,
        validation_worlds_per_cell=24,
        test_worlds_per_cell=48,
    )

    degree_two = _sensor(receipt, "regular_degree_2")
    degree_three = _sensor(receipt, "regular_degree_3")
    random_nine = _sensor(receipt, "random_sparse_9")
    pruned_nine = _sensor(receipt, "posthoc_pruned_9")
    pruned_twelve = _sensor(receipt, "posthoc_pruned_12")

    assert receipt.selected_structured_degree == 3
    assert degree_two.validation_min_success < 0.95
    assert degree_three.validation_min_success >= 0.95

    assert degree_three.guide_coordinate_touches_total == 9216
    assert random_nine.guide_coordinate_touches_total == 9216
    assert pruned_nine.guide_coordinate_touches_total == 9216

    assert random_nine.validation_min_success < 0.95
    assert pruned_nine.validation_min_success < 0.95

    assert degree_three.held_out_min_success >= 0.95
    assert (
        degree_three.guide_coordinate_touches_total
        < pruned_twelve.guide_coordinate_touches_total
    )
