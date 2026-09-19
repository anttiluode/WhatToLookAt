"""Pure helpers for the high-resolution whole-tile omission gate."""

from __future__ import annotations

import numpy as np


def tile_reduce(field: np.ndarray, grid: int) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    h, w = field.shape
    if h % grid or w % grid:
        raise ValueError("field shape must be divisible by grid")
    th, tw = h // grid, w // grid
    return field.reshape(grid, th, grid, tw).mean(axis=(1, 3)).ravel()


def edge_energy_scores(rgb: np.ndarray, grid: int) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    return tile_reduce(gx * gx + gy * gy, grid)


def top_indices(scores: np.ndarray, count: int) -> list[int]:
    values = np.asarray(scores, dtype=np.float64).ravel()
    if count < 1 or count > values.size:
        raise ValueError("invalid count")
    return [int(i) for i in np.argsort(values)[-int(count):][::-1]]


def correction_capture(correction_energy: np.ndarray, selected: list[int]) -> float:
    energy = np.asarray(correction_energy, dtype=np.float64).ravel()
    total = float(energy.sum())
    if total <= 1e-20:
        return float(len(selected) / len(energy))
    return float(energy[[int(i) for i in selected]].sum() / total)


def candidate_pass(
    summary: dict,
    min_recovery: float,
    min_layout_psnr_db: float,
    min_edge_correlation: float,
    min_edge_random_margin: float,
    min_speedup: float,
    required_wins: int,
    max_refine_fraction: float,
) -> bool:
    return bool(
        summary["mean_recovery"] >= float(min_recovery)
        and summary["min_layout_psnr_db"] >= float(min_layout_psnr_db)
        and summary["mean_edge_correlation"] >= float(min_edge_correlation)
        and summary["mean_edge_minus_random_recovery"] >= float(min_edge_random_margin)
        and summary["edge_case_wins_vs_random_median"] >= int(required_wins)
        and summary["mean_measured_speedup"] >= float(min_speedup)
        and summary["refine_fraction"] <= float(max_refine_fraction)
    )


def choose_smallest_passing(
    summaries: dict[str, dict],
    **thresholds,
) -> str | None:
    passing = [
        (int(name.split("_")[-1]), name)
        for name, summary in summaries.items()
        if candidate_pass(summary, **thresholds)
    ]
    if not passing:
        return None
    passing.sort()
    return passing[0][1]
