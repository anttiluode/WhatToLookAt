import numpy as np

from highres_tile_utils import (
    candidate_pass,
    choose_smallest_passing,
    correction_capture,
    edge_energy_scores,
    top_indices,
)


def test_edge_scores_find_sharp_half():
    x = np.zeros((32, 32, 3), dtype=float)
    x[:, 16::2] = 1.0
    scores = edge_energy_scores(x, grid=2)
    assert scores.shape == (4,)
    assert scores[1] > scores[0]
    assert scores[3] > scores[2]


def test_correction_capture_is_energy_fraction():
    e = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert abs(correction_capture(e, [3, 2]) - 0.7) < 1e-12


def _row(count, recovery, margin, speed, wins=6):
    return {
        "mean_recovery": recovery,
        "min_layout_psnr_db": 35.0,
        "mean_edge_correlation": 0.97,
        "mean_edge_minus_random_recovery": margin,
        "edge_case_wins_vs_random_median": wins,
        "mean_measured_speedup": speed,
        "refine_fraction": count / 16,
    }


def test_main_claim_rejects_twelve_of_sixteen_even_if_quality_is_good():
    row = _row(12, 0.95, 0.2, 1.3)
    assert not candidate_pass(
        row,
        min_recovery=0.8,
        min_layout_psnr_db=30.0,
        min_edge_correlation=0.9,
        min_edge_random_margin=0.1,
        min_speedup=1.75,
        required_wins=4,
        max_refine_fraction=0.5,
    )


def test_choose_smallest_passing_budget():
    rows = {
        "refine_4": _row(4, 0.72, 0.15, 3.8),
        "refine_8": _row(8, 0.88, 0.14, 1.95),
        "refine_12": _row(12, 0.97, 0.08, 1.3),
    }
    thresholds = dict(
        min_recovery=0.8,
        min_layout_psnr_db=30.0,
        min_edge_correlation=0.9,
        min_edge_random_margin=0.1,
        min_speedup=1.75,
        required_wins=4,
        max_refine_fraction=0.5,
    )
    assert choose_smallest_passing(rows, **thresholds) == "refine_8"


def test_top_indices():
    assert top_indices(np.asarray([1, 4, 2, 3]), 2) == [1, 3]
