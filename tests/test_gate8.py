from gate8_coded_overlap import run_gate8


def _sensor(receipt, name):
    return next(
        sensor
        for sensor in receipt.sensors
        if sensor.name == name
    )


def test_gate8_degree_three_is_first_structured_validation_winner():
    receipt = run_gate8(
        training_world_count=84,
        validation_worlds_per_cell=24,
        test_worlds_per_cell=24,
    )

    assert receipt.selected_structured_degree == 3

    degree_two = _sensor(
        receipt,
        "regular_degree_2",
    )
    degree_three = _sensor(
        receipt,
        "regular_degree_3",
    )

    assert degree_two.validation_min_success < 0.95
    assert degree_three.validation_min_success >= 0.95


def test_gate8_degree_three_beats_same_touch_controls_on_validation():
    receipt = run_gate8(
        training_world_count=84,
        validation_worlds_per_cell=24,
        test_worlds_per_cell=24,
    )

    degree_three = _sensor(
        receipt,
        "regular_degree_3",
    )
    random_nine = _sensor(
        receipt,
        "random_sparse_9",
    )
    pruned_nine = _sensor(
        receipt,
        "posthoc_pruned_9",
    )

    assert degree_three.guide_coordinate_touches_total == 9216
    assert random_nine.guide_coordinate_touches_total == 9216
    assert pruned_nine.guide_coordinate_touches_total == 9216

    assert degree_three.validation_min_success >= 0.95
    assert random_nine.validation_min_success < 0.95
    assert pruned_nine.validation_min_success < 0.95


def test_gate8_selected_overlap_survives_held_out_and_costs_less_than_12_rows():
    receipt = run_gate8(
        training_world_count=84,
        validation_worlds_per_cell=24,
        test_worlds_per_cell=32,
    )

    degree_three = _sensor(
        receipt,
        "regular_degree_3",
    )
    pruned_twelve = _sensor(
        receipt,
        "posthoc_pruned_12",
    )

    assert degree_three.held_out_min_success >= 0.95
    assert (
        degree_three.guide_coordinate_touches_total
        < pruned_twelve.guide_coordinate_touches_total
    )
