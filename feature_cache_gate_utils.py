"""Pure helpers for the practical feature-cache gate."""

from __future__ import annotations


def cache_candidate_pass(
    summary: dict,
    min_mean_recovery: float,
    min_psnr_db: float,
    min_speedup: float,
    required_faster_cases: int,
) -> bool:
    return bool(
        summary["mean_recovery"] >= float(min_mean_recovery)
        and summary["min_psnr_to_teacher_db"] >= float(min_psnr_db)
        and summary["mean_stage_speedup"] >= float(min_speedup)
        and summary["faster_case_count"] >= int(required_faster_cases)
    )


def select_fastest_passing(
    summaries: dict[str, dict],
    min_mean_recovery: float,
    min_psnr_db: float,
    min_speedup: float,
    required_faster_cases: int,
) -> str | None:
    passing = [
        (name, summary)
        for name, summary in summaries.items()
        if cache_candidate_pass(
            summary,
            min_mean_recovery,
            min_psnr_db,
            min_speedup,
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
