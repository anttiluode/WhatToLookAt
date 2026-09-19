"""Pure helpers for overlap-blended high-resolution tile omission."""

from __future__ import annotations

import numpy as np


def tile_origins(
    output_size: int = 2048,
    tile_size: int = 512,
    overlap: int = 128,
) -> list[tuple[int, int]]:
    stride = int(tile_size) - int(overlap)
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    span = int(output_size) - int(tile_size)
    if span < 0 or span % stride:
        raise ValueError("tile geometry must cover output exactly")
    grid = span // stride + 1
    return [
        (row * stride, col * stride)
        for row in range(grid)
        for col in range(grid)
    ]


def grid_size(
    output_size: int = 2048,
    tile_size: int = 512,
    overlap: int = 128,
) -> int:
    origins = tile_origins(output_size, tile_size, overlap)
    return int(round(len(origins) ** 0.5))


def _fade_in(overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.empty((0,), dtype=np.float32)
    t = (np.arange(overlap, dtype=np.float32) + 0.5) / float(overlap)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def feather_window(
    row: int,
    col: int,
    grid: int,
    tile_size: int = 512,
    overlap: int = 128,
) -> np.ndarray:
    """2-D partition-of-unity feather for one overlapping tile."""
    wx = np.ones(tile_size, dtype=np.float32)
    wy = np.ones(tile_size, dtype=np.float32)
    fade = _fade_in(overlap)

    if col > 0:
        wx[:overlap] = fade
    if col < grid - 1:
        wx[-overlap:] = 1.0 - fade
    if row > 0:
        wy[:overlap] = fade
    if row < grid - 1:
        wy[-overlap:] = 1.0 - fade

    return wy[:, None] * wx[None, :]


def partition_sum(
    output_size: int = 2048,
    tile_size: int = 512,
    overlap: int = 128,
) -> np.ndarray:
    origins = tile_origins(output_size, tile_size, overlap)
    grid = grid_size(output_size, tile_size, overlap)
    total = np.zeros((output_size, output_size), dtype=np.float32)
    for index, (y0, x0) in enumerate(origins):
        row, col = divmod(index, grid)
        total[y0:y0+tile_size, x0:x0+tile_size] += feather_window(
            row, col, grid, tile_size, overlap
        )
    return total


def edge_energy_scores(
    rgb: np.ndarray,
    output_size: int = 2048,
    tile_size: int = 512,
    overlap: int = 128,
) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    energy = gx * gx + gy * gy

    scores = []
    for y0, x0 in tile_origins(output_size, tile_size, overlap):
        scores.append(float(np.mean(energy[y0:y0+tile_size, x0:x0+tile_size])))
    return np.asarray(scores, dtype=np.float64)


def top_indices(scores: np.ndarray, count: int) -> list[int]:
    values = np.asarray(scores, dtype=np.float64).ravel()
    if count < 1 or count > values.size:
        raise ValueError("invalid count")
    return [int(i) for i in np.argsort(values)[-int(count):][::-1]]


def gram_recovery(gram: np.ndarray, selected: list[int]) -> float:
    """Exact correction recovery for a subset of weighted tile deltas."""
    g = np.asarray(gram, dtype=np.float64)
    total = float(g.sum())
    if total <= 1e-20:
        return 1.0
    sel = np.asarray([int(i) for i in selected], dtype=int)
    if sel.size == 0:
        return 0.0
    inner_teacher_selected = float(g[:, sel].sum())
    selected_energy = float(g[np.ix_(sel, sel)].sum())
    return float((2.0 * inner_teacher_selected - selected_energy) / total)


def greedy_oracle_indices(gram: np.ndarray, count: int) -> list[int]:
    """Teacher-aware greedy recovery ceiling/reference."""
    n = int(np.asarray(gram).shape[0])
    chosen: list[int] = []
    remaining = set(range(n))
    for _ in range(int(count)):
        best = max(
            remaining,
            key=lambda index: gram_recovery(gram, chosen + [int(index)]),
        )
        chosen.append(int(best))
        remaining.remove(best)
    return chosen


def candidate_pass(
    summary: dict,
    min_recovery: float,
    min_layout_psnr_db: float,
    min_edge_correlation: float,
    min_edge_random_margin: float,
    min_speedup: float,
    required_wins: int,
    max_refine_fraction: float,
    max_teacher_seam_ratio: float,
) -> bool:
    return bool(
        summary["mean_recovery"] >= float(min_recovery)
        and summary["min_layout_psnr_db"] >= float(min_layout_psnr_db)
        and summary["mean_edge_correlation"] >= float(min_edge_correlation)
        and summary["mean_edge_minus_random_recovery"] >= float(min_edge_random_margin)
        and summary["edge_case_wins_vs_random_median"] >= int(required_wins)
        and summary["mean_measured_speedup"] >= float(min_speedup)
        and summary["refine_fraction"] <= float(max_refine_fraction)
        and summary["max_teacher_seam_ratio"] <= float(max_teacher_seam_ratio)
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
