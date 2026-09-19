"""Pure CPU helpers for the practical diffusion compute-routing experiment."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def sihti_core_cpu(
    draft: np.ndarray,
    steps: int = 8,
    sigma: float = 0.12,
) -> np.ndarray:
    """Reproduce the small edge-aware Sihti AI sieve on an RGB float image."""
    img = np.asarray(draft, dtype=np.float64)
    curr = img.copy()
    inv_2s2 = 1.0 / (2.0 * float(sigma) ** 2)

    for _ in range(int(steps)):
        pc = np.pad(curr, ((1, 1), (1, 1), (0, 0)), mode="edge")
        po = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")

        uo = po[:-2, 1:-1]
        do = po[2:, 1:-1]
        lo = po[1:-1, :-2]
        ro = po[1:-1, 2:]

        wu = np.exp(-np.sum((img - uo) ** 2, axis=2, keepdims=True) * inv_2s2)
        wd = np.exp(-np.sum((img - do) ** 2, axis=2, keepdims=True) * inv_2s2)
        wl = np.exp(-np.sum((img - lo) ** 2, axis=2, keepdims=True) * inv_2s2)
        wr = np.exp(-np.sum((img - ro) ** 2, axis=2, keepdims=True) * inv_2s2)

        uv = pc[:-2, 1:-1]
        dv = pc[2:, 1:-1]
        lv = pc[1:-1, :-2]
        rv = pc[1:-1, 2:]

        wsum = 1.0 + 0.25 * (wu + wd + wl + wr)
        curr = (
            img + 0.25 * (wu * uv + wd * dv + wl * lv + wr * rv)
        ) / wsum

    return curr


def tile_reduce(field: np.ndarray, grid: int) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    h, w = field.shape
    if h % grid or w % grid:
        raise ValueError(f"shape {field.shape} not divisible by grid={grid}")
    th, tw = h // grid, w // grid
    return field.reshape(grid, th, grid, tw).mean(axis=(1, 3)).ravel()


def edge_scores(draft: np.ndarray, grid: int = 4) -> np.ndarray:
    x = np.asarray(draft, dtype=np.float64)
    gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    return tile_reduce(gx * gx + gy * gy, grid)


def variance_scores(draft: np.ndarray, grid: int = 4) -> np.ndarray:
    x = np.asarray(draft, dtype=np.float64)
    h, w, _ = x.shape
    if h % grid or w % grid:
        raise ValueError("draft shape must be divisible by grid")
    th, tw = h // grid, w // grid
    blocks = x.reshape(grid, th, grid, tw, 3)
    return blocks.var(axis=(1, 3, 4)).ravel()


def sihti_residue_scores(
    draft: np.ndarray,
    grid: int = 4,
    steps: int = 8,
    sigma: float = 0.12,
) -> np.ndarray:
    x = np.asarray(draft, dtype=np.float64)
    core = sihti_core_cpu(x, steps=steps, sigma=sigma)
    residue = np.mean((x - core) ** 2, axis=2)
    return tile_reduce(residue, grid)


def matched_gaussian_residue_scores(
    draft: np.ndarray,
    grid: int = 4,
    steps: int = 8,
    sigma: float = 0.12,
) -> tuple[np.ndarray, float]:
    x = np.asarray(draft, dtype=np.float64)
    core = sihti_core_cpu(x, steps=steps, sigma=sigma)
    target = float(np.mean((x - core) ** 2))

    best = None
    for candidate in (0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        smooth = gaussian_filter(
            x,
            sigma=(candidate, candidate, 0.0),
            mode="reflect",
        )
        mse = float(np.mean((x - smooth) ** 2))
        item = (abs(mse - target), candidate, smooth)
        if best is None or item[0] < best[0]:
            best = item

    _, chosen_sigma, smooth = best
    residue = np.mean((x - smooth) ** 2, axis=2)
    return tile_reduce(residue, grid), float(chosen_sigma)


def top_tiles(scores: np.ndarray, count: int) -> list[int]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if count < 1 or count > scores.size:
        raise ValueError("invalid tile count")
    return [int(i) for i in np.argsort(scores)[-int(count):][::-1]]


def random_tiles(total: int, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    return [int(i) for i in rng.choice(int(total), int(count), replace=False)]


def context_crops(
    image: Image.Image,
    tile_indices: list[int],
    grid: int = 4,
    context: int = 32,
) -> tuple[list[Image.Image], dict]:
    """Return fixed-size reflected-context crops around selected grid tiles."""
    rgb = np.asarray(image.convert("RGB"))
    h, w, _ = rgb.shape
    if h != w or h % grid:
        raise ValueError("expected square image divisible by grid")
    tile = h // grid

    padded = np.pad(
        rgb,
        ((context, context), (context, context), (0, 0)),
        mode="reflect",
    )
    crop_size = tile + 2 * int(context)

    crops = []
    coords = []
    for index in tile_indices:
        row, col = divmod(int(index), grid)
        y0 = row * tile
        x0 = col * tile
        crop = padded[y0 : y0 + crop_size, x0 : x0 + crop_size]
        crops.append(Image.fromarray(crop, mode="RGB"))
        coords.append((row, col))

    return crops, {
        "tile_size": int(tile),
        "context": int(context),
        "grid": int(grid),
        "coords": coords,
    }


def paste_refined_centers(
    base: Image.Image,
    refined_crops: list[Image.Image],
    meta: dict,
) -> Image.Image:
    out = base.convert("RGB").copy()
    tile = int(meta["tile_size"])
    context = int(meta["context"])

    for crop, (row, col) in zip(refined_crops, meta["coords"]):
        crop = crop.convert("RGB")
        center = crop.crop(
            (context, context, context + tile, context + tile)
        )
        out.paste(center, (col * tile, row * tile))

    return out
