from feature_cache_gate_utils import (
    cache_candidate_pass,
    select_fastest_passing,
)


def _row(recovery, psnr, speedup, faster):
    return {
        "mean_recovery": recovery,
        "min_psnr_to_teacher_db": psnr,
        "mean_stage_speedup": speedup,
        "faster_case_count": faster,
    }


def test_cache_candidate_requires_quality_and_speed():
    good = _row(0.9, 31.0, 1.2, 5)
    bad_quality = _row(0.7, 31.0, 1.3, 6)
    bad_speed = _row(0.95, 35.0, 1.05, 6)

    assert cache_candidate_pass(good, 0.8, 28.0, 1.1, 4)
    assert not cache_candidate_pass(bad_quality, 0.8, 28.0, 1.1, 4)
    assert not cache_candidate_pass(bad_speed, 0.8, 28.0, 1.1, 4)


def test_fastest_passing_candidate_is_selected():
    rows = {
        "branch_0": _row(0.91, 31.0, 1.18, 5),
        "branch_1": _row(0.93, 32.0, 1.25, 6),
        "branch_2": _row(0.97, 34.0, 1.12, 6),
    }
    assert select_fastest_passing(
        rows, 0.8, 28.0, 1.1, 4
    ) == "branch_1"
