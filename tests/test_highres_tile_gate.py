import numpy as np

from highres_tile_utils import (
    candidate_pass,
    choose_smallest_passing,
    feather_window,
    gram_recovery,
    grid_size,
    partition_sum,
    tile_origins,
    top_indices,
)


def test_overlap_geometry_is_five_by_five():
    origins = tile_origins(2048, 512, 128)
    assert len(origins) == 25
    assert grid_size(2048, 512, 128) == 5
    assert origins[0] == (0, 0)
    assert origins[-1] == (1536, 1536)


def test_feather_windows_form_partition_of_unity():
    total = partition_sum(2048, 512, 128)
    assert total.shape == (2048, 2048)
    assert np.max(np.abs(total - 1.0)) < 1e-5


def test_interior_window_fades_on_all_sides():
    w = feather_window(2, 2, 5, 512, 128)
    assert w[0, 0] < 0.01
    assert w[256, 256] > 0.99
    assert w[-1, -1] < 0.01


def test_gram_recovery_is_exact_for_simple_orthogonal_case():
    g = np.diag([1.0, 2.0, 3.0])
    assert abs(gram_recovery(g, [2]) - 0.5) < 1e-12
    assert abs(gram_recovery(g, [0, 1, 2]) - 1.0) < 1e-12


def _row(count, recovery, margin, speed, wins=6, seam=1.1):
    return {
        "mean_recovery": recovery,
        "min_layout_psnr_db": 35.0,
        "mean_edge_correlation": 0.97,
        "mean_edge_minus_random_recovery": margin,
        "edge_case_wins_vs_random_median": wins,
        "mean_measured_speedup": speed,
        "refine_fraction": count / 25,
        "max_teacher_seam_ratio": seam,
    }


def test_main_claim_rejects_more_than_half_even_if_quality_is_good():
    row = _row(18, 0.95, 0.2, 1.9)
    assert not candidate_pass(
        row,
        min_recovery=0.8,
        min_layout_psnr_db=30.0,
        min_edge_correlation=0.9,
        min_edge_random_margin=0.1,
        min_speedup=1.75,
        required_wins=4,
        max_refine_fraction=0.5,
        max_teacher_seam_ratio=2.0,
    )


def test_seamy_teacher_invalidates_gate():
    row = _row(12, 0.95, 0.2, 2.0, seam=2.5)
    assert not candidate_pass(
        row,
        min_recovery=0.8,
        min_layout_psnr_db=30.0,
        min_edge_correlation=0.9,
        min_edge_random_margin=0.1,
        min_speedup=1.75,
        required_wins=4,
        max_refine_fraction=0.5,
        max_teacher_seam_ratio=2.0,
    )


def test_choose_smallest_passing_budget():
    rows = {
        "refine_6": _row(6, 0.72, 0.15, 3.8),
        "refine_12": _row(12, 0.88, 0.14, 1.95),
        "refine_18": _row(18, 0.97, 0.08, 1.3),
    }
    thresholds = dict(
        min_recovery=0.8,
        min_layout_psnr_db=30.0,
        min_edge_correlation=0.9,
        min_edge_random_margin=0.1,
        min_speedup=1.75,
        required_wins=4,
        max_refine_fraction=0.5,
        max_teacher_seam_ratio=2.0,
    )
    assert choose_smallest_passing(rows, **thresholds) == "refine_12"


def test_top_indices():
    assert top_indices(np.asarray([1, 4, 2, 3]), 2) == [1, 3]
