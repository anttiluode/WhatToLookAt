from vae_closeout_utils import (
    select_fastest_vae,
    vae_candidate_pass,
)


def _row(recovery, layout, edge, speedup, faster):
    return {
        "mean_recovery": recovery,
        "min_layout_psnr_to_teacher_db": layout,
        "mean_edge_correlation_to_teacher": edge,
        "mean_stage_speedup": speedup,
        "faster_case_count": faster,
    }


def test_vae_candidate_requires_fidelity_and_speed():
    good = _row(0.92, 31.0, 0.96, 1.5, 6)
    bad_quality = _row(0.70, 31.0, 0.96, 2.0, 6)
    bad_speed = _row(0.95, 32.0, 0.97, 1.2, 6)

    assert vae_candidate_pass(good, 0.85, 28.0, 0.90, 1.30, 4)
    assert not vae_candidate_pass(bad_quality, 0.85, 28.0, 0.90, 1.30, 4)
    assert not vae_candidate_pass(bad_speed, 0.85, 28.0, 0.90, 1.30, 4)


def test_fastest_passing_vae_is_selected():
    rows = {
        "fp16_fix": _row(0.98, 35.0, 0.99, 1.45, 6),
        "taesdxl": _row(0.90, 29.0, 0.94, 1.85, 6),
    }
    assert select_fastest_vae(
        rows, 0.85, 28.0, 0.90, 1.30, 4
    ) == "taesdxl"
