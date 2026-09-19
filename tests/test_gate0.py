import numpy as np

from what_to_look_at import (
    active_component_mask,
    group_harmonic_reconstruct,
    make_labels,
    make_scene,
    psnr,
    run_gate0,
    shuffled_groups,
)


def test_active_relation_one_measurement_per_object_is_exact():
    labels = make_labels()
    scene = make_scene(labels)
    mask = active_component_mask(labels)
    observed = np.zeros_like(scene)
    observed[mask] = scene[mask]
    recovered = group_harmonic_reconstruct(mask, observed, labels)

    assert int(mask.sum()) == 4
    assert psnr(scene, recovered) >= 100.0


def test_shuffled_graph_destroys_the_relation_advantage():
    labels = make_labels()
    wrong = shuffled_groups(labels)
    scene = make_scene(labels)
    mask = active_component_mask(labels)
    observed = np.zeros_like(scene)
    observed[mask] = scene[mask]

    right = group_harmonic_reconstruct(mask, observed, labels)
    bad = group_harmonic_reconstruct(mask, observed, wrong)
    assert psnr(scene, right) >= 100.0
    assert psnr(scene, bad) < 20.0


def test_gate0_reference_receipt():
    # Small CI receipt; README/result generation uses the larger 48-seed run.
    receipt = run_gate0(seeds=16)
    assert receipt.active_relation_fraction == 4 / 256
    assert receipt.active_relation_psnr_db >= 100.0
    assert receipt.random_relation.first_count_at_90pct_success is not None
    assert receipt.random_relation.first_count_at_90pct_success <= 24
    assert receipt.spatial_lattice.first_count_at_90pct_success is None
    assert receipt.shuffled_relation.first_count_at_90pct_success is None
