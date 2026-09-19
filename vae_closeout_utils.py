"""Pure selection helpers for the SDXL VAE engineering closeout."""

from __future__ import annotations


def vae_candidate_pass(
    summary: dict,
    min_mean_recovery: float,
    min_layout_psnr_db: float,
    min_edge_correlation: float,
    min_stage_speedup: float,
    required_faster_cases: int,
) -> bool:
    return bool(
        summary["mean_recovery"] >= float(min_mean_recovery)
        and summary["min_layout_psnr_to_teacher_db"] >= float(min_layout_psnr_db)
        and summary["mean_edge_correlation_to_teacher"] >= float(min_edge_correlation)
        and summary["mean_stage_speedup"] >= float(min_stage_speedup)
        and summary["faster_case_count"] >= int(required_faster_cases)
    )


def select_fastest_vae(
    summaries: dict[str, dict],
    min_mean_recovery: float,
    min_layout_psnr_db: float,
    min_edge_correlation: float,
    min_stage_speedup: float,
    required_faster_cases: int,
) -> str | None:
    passing = [
        (name, summary)
        for name, summary in summaries.items()
        if vae_candidate_pass(
            summary,
            min_mean_recovery,
            min_layout_psnr_db,
            min_edge_correlation,
            min_stage_speedup,
            required_faster_cases,
        )
    ]
    if not passing:
        return None

    passing.sort(
        key=lambda item: (
            -float(item[1]["mean_stage_speedup"]),
            -float(item[1]["mean_recovery"]),
            item[0],
        )
    )
    return passing[0][0]
