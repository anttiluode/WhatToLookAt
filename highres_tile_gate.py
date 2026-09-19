#!/usr/bin/env python3
"""High-resolution Gate H0b: overlap-blended whole-tile omission at 2048px.

H0's first implementation is invalid as a visual-quality test: it pasted
independently refined 512px blocks edge-to-edge, so the all-tile "teacher" was
itself visibly blocky. H0b fixes the geometry before any claim is scored.

Geometry:
    2048x2048 output
    512x512 diffusion tiles
    128px overlap
    384px stride
    5x5 = 25 independent tile calls

Each tile contributes through a cosine feather window. The 25 windows form a
partition of unity, so the all-tile teacher is a smooth weighted blend. A
selective route starts from the smooth Lanczos base and adds only the selected
feathered tile corrections. Skipping a tile still deletes one complete
VAE+UNet+VAE diffusion call, but it no longer creates a hard rectangular paste.

Candidate budgets:
    refine 6/25  (~24%, skip 76%)
    refine 12/25 (~48%, skip 52%)
    refine 18/25 (~72%, diagnostic only)

Validation seeds 100/101 choose the smallest eligible passing budget. Seed 102
is held out.

A teacher-seam validity control is included. If overlap blending still creates
boundary jumps more than 2x nearby ordinary gradients, the gate is invalid
regardless of recovery.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    AutoencoderKL,
)

from highres_tile_utils import (
    candidate_pass,
    choose_smallest_passing,
    edge_energy_scores,
    feather_window,
    gram_recovery,
    greedy_oracle_indices,
    grid_size,
    partition_sum,
    tile_origins,
    top_indices,
)

MODEL_ID = "stabilityai/sdxl-turbo"
FP16_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

PROMPTS = (
    "Wide landscape photograph of a calm Nordic lake beneath a large clear blue sky, distant pine forest and mountains, natural daylight",
    "Minimalist architectural photograph of a white concrete museum under a clear sky, broad smooth walls, a few sharp geometric details",
    "Macro photograph of a vivid red flower with crisp petals against a large soft creamy bokeh background, natural photographic lighting",
)

SOURCE_SIZE = 512
OUTPUT_SIZE = 2048
TILE_SIZE = 512
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP
GRID = grid_size(OUTPUT_SIZE, TILE_SIZE, OVERLAP)
TOTAL_TILES = GRID * GRID

REFINE_COUNTS = (6, 12, 18)
RANDOM_REPEATS = 64

VALIDATION_SEEDS = (100, 101)
HELD_OUT_SEEDS = (102,)

MIN_RECOVERY = 0.80
MIN_LAYOUT_PSNR_DB = 30.0
MIN_EDGE_CORRELATION = 0.90
MIN_EDGE_RANDOM_MARGIN = 0.10
MIN_SPEEDUP = 1.75
MAX_REFINE_FRACTION = 0.50
MAX_TEACHER_SEAM_RATIO = 2.0


def sync():
    torch.cuda.synchronize()


def gen(seed: int):
    return torch.Generator(device="cuda").manual_seed(int(seed))


def timed(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t0


def arr(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def to_image(x: np.ndarray) -> Image.Image:
    y = np.clip(np.asarray(x) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(y, mode="RGB")


def mse(a: Image.Image, b: Image.Image) -> float:
    aa, bb = arr(a), arr(b)
    return float(np.mean((aa - bb) ** 2))


def psnr(a: Image.Image, b: Image.Image) -> float:
    value = mse(a, b)
    return 120.0 if value <= 1e-14 else float(10.0 * np.log10(1.0 / value))


def layout_psnr(a: Image.Image, b: Image.Image) -> float:
    return psnr(
        a.resize((512, 512), Image.Resampling.BILINEAR),
        b.resize((512, 512), Image.Resampling.BILINEAR),
    )


def edges(image: Image.Image) -> np.ndarray:
    x = arr(image).astype(np.float64)
    gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    return np.sqrt(gx * gx + gy * gy).ravel()


def edge_corr(a: Image.Image, b: Image.Image) -> float:
    x, y = edges(a), edges(b)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float(np.allclose(x, y))
    return float(np.corrcoef(x, y)[0, 1])


def recovery(base: Image.Image, candidate: Image.Image, teacher: Image.Image) -> float:
    base_err = mse(base, teacher)
    if base_err <= 1e-20:
        return 0.0
    return float(1.0 - mse(candidate, teacher) / base_err)


def boundary_gradient_values(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    x = arr(image).astype(np.float64)
    gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    vertical = np.mean(np.abs(gray[:, 1:] - gray[:, :-1]), axis=0)
    horizontal = np.mean(np.abs(gray[1:, :] - gray[:-1, :]), axis=1)
    return vertical, horizontal


def teacher_seam_ratio(image: Image.Image) -> float:
    """Boundary jump divided by nearby non-boundary jump."""
    vertical, horizontal = boundary_gradient_values(image)
    boundaries = [STRIDE * i for i in range(1, GRID)]

    ratios = []
    radius = 12
    exclusion = 2

    for boundary in boundaries:
        idx = boundary - 1
        lo = max(0, idx - radius)
        hi = min(len(vertical), idx + radius + 1)
        neighborhood = np.concatenate([
            vertical[lo:max(lo, idx - exclusion)],
            vertical[min(hi, idx + exclusion + 1):hi],
        ])
        if neighborhood.size:
            ratios.append(float(vertical[idx] / max(float(neighborhood.mean()), 1e-8)))

    for boundary in boundaries:
        idx = boundary - 1
        lo = max(0, idx - radius)
        hi = min(len(horizontal), idx + radius + 1)
        neighborhood = np.concatenate([
            horizontal[lo:max(lo, idx - exclusion)],
            horizontal[min(hi, idx + exclusion + 1):hi],
        ])
        if neighborhood.size:
            ratios.append(float(horizontal[idx] / max(float(neighborhood.mean()), 1e-8)))

    return 1.0 if not ratios else float(np.mean(ratios))


def make_windows() -> list[np.ndarray]:
    windows = []
    for index in range(TOTAL_TILES):
        row, col = divmod(index, GRID)
        windows.append(
            feather_window(row, col, GRID, TILE_SIZE, OVERLAP).astype(np.float32)
        )
    return windows


def weighted_deltas(
    base_np: np.ndarray,
    refined_tiles: list[Image.Image],
    origins: list[tuple[int, int]],
    windows: list[np.ndarray],
) -> list[np.ndarray]:
    deltas = []
    for index, (y0, x0) in enumerate(origins):
        base_crop = base_np[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE]
        refined = arr(refined_tiles[index])
        delta = (refined - base_crop) * windows[index][:, :, None]
        deltas.append(delta.astype(np.float32))
    return deltas


def build_gram(
    deltas: list[np.ndarray],
    origins: list[tuple[int, int]],
) -> np.ndarray:
    n = len(deltas)
    g = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        yi, xi = origins[i]
        g[i, i] = float(np.sum(deltas[i].astype(np.float64) ** 2))
        for j in range(i + 1, n):
            yj, xj = origins[j]

            y0 = max(yi, yj)
            x0 = max(xi, xj)
            y1 = min(yi + TILE_SIZE, yj + TILE_SIZE)
            x1 = min(xi + TILE_SIZE, xj + TILE_SIZE)
            if y1 <= y0 or x1 <= x0:
                continue

            ai = deltas[i][
                y0 - yi:y1 - yi,
                x0 - xi:x1 - xi,
            ].astype(np.float64)
            bj = deltas[j][
                y0 - yj:y1 - yj,
                x0 - xj:x1 - xj,
            ].astype(np.float64)
            dot = float(np.sum(ai * bj))
            g[i, j] = dot
            g[j, i] = dot

    return g


def assemble(
    base_np: np.ndarray,
    refined_tiles: list[Image.Image],
    selected: list[int],
    origins: list[tuple[int, int]],
    windows: list[np.ndarray],
) -> Image.Image:
    accum = np.zeros_like(base_np, dtype=np.float32)
    weight = np.zeros(base_np.shape[:2], dtype=np.float32)

    selected_set = {int(i) for i in selected}
    for index in selected_set:
        y0, x0 = origins[index]
        w = windows[index]
        refined = arr(refined_tiles[index])
        accum[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] += refined * w[:, :, None]
        weight[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] += w

    # All-tile windows form a partition of unity. For a selected subset, the
    # missing weight is explicitly assigned back to the smooth Lanczos base.
    weight = np.clip(weight, 0.0, 1.0)
    out = base_np * (1.0 - weight[:, :, None]) + accum
    return to_image(out)


def random_recoveries(
    gram: np.ndarray,
    count: int,
    seed: int,
    repeats: int = RANDOM_REPEATS,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    values = []
    n = gram.shape[0]
    for _ in range(int(repeats)):
        selected = [int(i) for i in rng.choice(n, int(count), replace=False)]
        values.append(gram_recovery(gram, selected))
    return np.asarray(values, dtype=np.float64)


def make_contact(
    base: Image.Image,
    teacher: Image.Image,
    variants: list[tuple[str, Image.Image]],
    path: Path,
):
    items = [("2048 base", base), ("all 25 overlap tiles", teacher)] + variants
    thumb = 384
    label_h = 26
    cols = 3
    rows = math.ceil(len(items) / cols)
    canvas = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(items):
        row, col = divmod(i, cols)
        x0, y0 = col * thumb, row * (thumb + label_h)
        canvas.paste(
            image.resize((thumb, thumb), Image.Resampling.LANCZOS),
            (x0, y0),
        )
        draw.text((x0 + 4, y0 + thumb + 4), label, fill="black")
    canvas.save(path, quality=92)


def summarize(cases: list[dict], seed_set: set[int], count: int) -> dict:
    rows = [
        case["budgets"][str(int(count))]
        for case in cases
        if int(case["seed"]) in seed_set
    ]
    recoveries = np.asarray([row["edge_recovery"] for row in rows])
    random_median = np.asarray([row["random_median_recovery"] for row in rows])
    layout = np.asarray([row["layout_psnr_db"] for row in rows])
    edgec = np.asarray([row["edge_correlation"] for row in rows])
    speed = np.asarray([row["measured_speedup"] for row in rows])
    seam = np.asarray([
        case["teacher_seam_ratio"]
        for case in cases
        if int(case["seed"]) in seed_set
    ])

    return {
        "cases": len(rows),
        "refine_count": int(count),
        "total_tiles": TOTAL_TILES,
        "refine_fraction": float(count / TOTAL_TILES),
        "skip_fraction": float(1.0 - count / TOTAL_TILES),
        "mean_recovery": float(recoveries.mean()),
        "min_recovery": float(recoveries.min()),
        "mean_random_median_recovery": float(random_median.mean()),
        "mean_edge_minus_random_recovery": float(
            (recoveries - random_median).mean()
        ),
        "edge_case_wins_vs_random_median": int(np.sum(recoveries > random_median)),
        "mean_layout_psnr_db": float(layout.mean()),
        "min_layout_psnr_db": float(layout.min()),
        "mean_edge_correlation": float(edgec.mean()),
        "min_edge_correlation": float(edgec.min()),
        "mean_measured_speedup": float(speed.mean()),
        "min_measured_speedup": float(speed.min()),
        "mean_greedy_oracle_recovery": float(np.mean([
            row["greedy_oracle_recovery"] for row in rows
        ])),
        "mean_teacher_seam_ratio": float(seam.mean()),
        "max_teacher_seam_ratio": float(seam.max()),
    }


def required_wins(n: int) -> int:
    return int(math.ceil(2 * int(n) / 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--strength", type=float, default=0.60)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/highres_tile_h0b"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("H0b requires CUDA.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = tuple(args.prompt) if args.prompt else PROMPTS

    origins = tile_origins(OUTPUT_SIZE, TILE_SIZE, OVERLAP)
    windows = make_windows()

    part = partition_sum(OUTPUT_SIZE, TILE_SIZE, OVERLAP)
    partition_error = float(np.max(np.abs(part - 1.0)))
    if partition_error > 1e-5:
        raise RuntimeError(f"feather windows do not partition unity: {partition_error}")

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        f"Geometry: {GRID}x{GRID}={TOTAL_TILES} tiles, "
        f"{TILE_SIZE}px, overlap {OVERLAP}px, stride {STRIDE}px"
    )
    print("Loading SDXL-Turbo and fp16-fix VAE...")

    pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        variant="fp16",
        local_files_only=args.local_only,
    ).to("cuda")
    pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i).to("cuda")
    fp16_vae = AutoencoderKL.from_pretrained(
        FP16_VAE_ID,
        dtype=torch.float16,
        local_files_only=args.local_only,
    ).to("cuda")
    pipe_i2i.vae = fp16_vae
    pipe_t2i.set_progress_bar_config(disable=True)
    pipe_i2i.set_progress_bar_config(disable=True)

    print("Warmup...")
    _ = pipe_t2i(
        prompt="warmup",
        width=SOURCE_SIZE,
        height=SOURCE_SIZE,
        num_inference_steps=1,
        guidance_scale=0.0,
        generator=gen(1),
    ).images[0]
    _ = pipe_i2i(
        prompt="warmup",
        image=Image.new("RGB", (TILE_SIZE, TILE_SIZE), "gray"),
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=0.0,
        generator=gen(2),
    ).images[0]
    sync()

    cases = []

    for pidx, prompt in enumerate(prompts, start=1):
        for seed in args.seeds:
            stem = f"p{pidx:02d}_s{seed}"
            print("\n", stem, prompt)

            source, source_s = timed(lambda: pipe_t2i(
                prompt=prompt,
                width=SOURCE_SIZE,
                height=SOURCE_SIZE,
                num_inference_steps=1,
                guidance_scale=0.0,
                generator=gen(seed),
            ).images[0])
            base = source.resize(
                (OUTPUT_SIZE, OUTPUT_SIZE),
                Image.Resampling.LANCZOS,
            )
            base_np = arr(base)

            refined_tiles = []
            tile_seconds = []
            for tile_index, (y0, x0) in enumerate(origins):
                tile = base.crop((x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE))
                refined, dt = timed(
                    lambda tile=tile, tile_index=tile_index: pipe_i2i(
                        prompt=prompt,
                        image=tile,
                        num_inference_steps=args.steps,
                        strength=args.strength,
                        guidance_scale=0.0,
                        generator=gen(seed + 100000 + tile_index),
                    ).images[0]
                )
                refined_tiles.append(refined)
                tile_seconds.append(float(dt))

            teacher = assemble(
                base_np,
                refined_tiles,
                list(range(TOTAL_TILES)),
                origins,
                windows,
            )
            teacher_seconds = float(sum(tile_seconds))
            seam_ratio = teacher_seam_ratio(teacher)

            deltas = weighted_deltas(base_np, refined_tiles, origins, windows)
            gram = build_gram(deltas, origins)

            # Cross-check the Gram recovery against the actual image assembly.
            gram_all = gram_recovery(gram, list(range(TOTAL_TILES)))
            if abs(gram_all - 1.0) > 1e-6:
                raise RuntimeError(f"full Gram recovery is not 1: {gram_all}")

            scores = edge_energy_scores(
                base_np,
                OUTPUT_SIZE,
                TILE_SIZE,
                OVERLAP,
            )

            budgets = {}
            contact_variants = []

            print(
                f"teacher seam ratio {seam_ratio:.3f}  "
                f"all-tile time {teacher_seconds:.2f}s"
            )

            for count in REFINE_COUNTS:
                selector_t0 = time.perf_counter()
                edge_selected = top_indices(scores, count)
                selector_seconds = time.perf_counter() - selector_t0

                edge_image = assemble(
                    base_np,
                    refined_tiles,
                    edge_selected,
                    origins,
                    windows,
                )
                edge_recovery = recovery(base, edge_image, teacher)
                gram_edge = gram_recovery(gram, edge_selected)

                random_values = random_recoveries(
                    gram,
                    count,
                    seed=seed + 700000 + count,
                )
                oracle_selected = greedy_oracle_indices(gram, count)
                oracle_recovery = gram_recovery(gram, oracle_selected)

                selected_tile_seconds = float(
                    sum(tile_seconds[i] for i in edge_selected)
                )
                route_seconds = float(selector_seconds + selected_tile_seconds)
                speedup = float(
                    teacher_seconds / route_seconds
                    if route_seconds > 0
                    else float("inf")
                )

                budgets[str(count)] = {
                    "edge_selected_tiles": edge_selected,
                    "edge_recovery": float(edge_recovery),
                    "edge_recovery_gram_crosscheck": float(gram_edge),
                    "random_median_recovery": float(np.median(random_values)),
                    "random_p10_recovery": float(np.percentile(random_values, 10)),
                    "random_p90_recovery": float(np.percentile(random_values, 90)),
                    "greedy_oracle_selected_tiles": oracle_selected,
                    "greedy_oracle_recovery": float(oracle_recovery),
                    "layout_psnr_db": layout_psnr(edge_image, teacher),
                    "edge_correlation": edge_corr(edge_image, teacher),
                    "selector_seconds": float(selector_seconds),
                    "selected_tile_seconds": selected_tile_seconds,
                    "estimated_route_seconds_from_measured_tile_calls": route_seconds,
                    "all_tile_seconds": teacher_seconds,
                    "measured_speedup": speedup,
                }

                edge_image.save(
                    args.output_dir / f"{stem}_edge_{count}of{TOTAL_TILES}.jpg",
                    quality=92,
                )
                contact_variants.append(
                    (f"edge {count}/{TOTAL_TILES}", edge_image)
                )

                print(
                    f"{count:2d}/{TOTAL_TILES} edge recovery {edge_recovery:.3f}  "
                    f"random-med {np.median(random_values):.3f}  "
                    f"greedy-oracle {oracle_recovery:.3f}  "
                    f"speedup {speedup:.2f}x"
                )

            base.save(args.output_dir / f"{stem}_base.jpg", quality=92)
            teacher.save(args.output_dir / f"{stem}_teacher.jpg", quality=92)
            contact = args.output_dir / f"{stem}_contact.jpg"
            make_contact(base, teacher, contact_variants, contact)

            cases.append({
                "prompt": prompt,
                "seed": int(seed),
                "source_seconds": float(source_s),
                "all_tile_seconds": teacher_seconds,
                "mean_tile_seconds": float(np.mean(tile_seconds)),
                "tile_seconds": tile_seconds,
                "teacher_seam_ratio": float(seam_ratio),
                "edge_scores": [float(x) for x in scores],
                "budgets": budgets,
                "contact_sheet": str(contact),
            })

            del deltas, gram
            torch.cuda.empty_cache()

    validation = {
        f"refine_{count}": summarize(
            cases, set(VALIDATION_SEEDS), count
        )
        for count in REFINE_COUNTS
    }

    n_validation = sum(
        int(case["seed"]) in VALIDATION_SEEDS for case in cases
    )
    thresholds = {
        "min_recovery": MIN_RECOVERY,
        "min_layout_psnr_db": MIN_LAYOUT_PSNR_DB,
        "min_edge_correlation": MIN_EDGE_CORRELATION,
        "min_edge_random_margin": MIN_EDGE_RANDOM_MARGIN,
        "min_speedup": MIN_SPEEDUP,
        "required_wins": required_wins(n_validation),
        "max_refine_fraction": MAX_REFINE_FRACTION,
        "max_teacher_seam_ratio": MAX_TEACHER_SEAM_RATIO,
    }

    for summary in validation.values():
        summary["pass"] = candidate_pass(summary, **thresholds)

    selected = choose_smallest_passing(validation, **thresholds)

    held_out = None
    held_out_pass = False
    if selected is not None:
        count = int(selected.split("_")[-1])
        held_out = summarize(cases, set(HELD_OUT_SEEDS), count)
        held_thresholds = dict(thresholds)
        n_test = sum(
            int(case["seed"]) in HELD_OUT_SEEDS for case in cases
        )
        held_thresholds["required_wins"] = required_wins(n_test)
        held_out_pass = candidate_pass(held_out, **held_thresholds)
        held_out["pass"] = held_out_pass

    payload = {
        "gate": "H0b overlap-blended whole-tile omission",
        "invalidated_predecessor": (
            "H0 hard 4x4 tile paste is invalid as visual evidence because the "
            "all-tile teacher itself had visible block seams."
        ),
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "vae": FP16_VAE_ID,
        "source_size": SOURCE_SIZE,
        "output_size": OUTPUT_SIZE,
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "stride": STRIDE,
        "grid": GRID,
        "total_tiles": TOTAL_TILES,
        "partition_max_abs_error": partition_error,
        "refine_counts": list(REFINE_COUNTS),
        "random_repeats": RANDOM_REPEATS,
        "validation_seeds": list(VALIDATION_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "thresholds": thresholds,
        "timing_note": (
            "Selective route time is selector CPU time plus the actual measured "
            "per-tile diffusion times of selected independent 512px calls. "
            "Omitted calls are genuinely absent."
        ),
        "scope_note": (
            "H0b uses reproducible generated photographic scenes. A pass earns "
            "a real-natural-photo H1; it is not yet an arbitrary-photo claim."
        ),
        "validation": validation,
        "selected_candidate": selected,
        "held_out": held_out,
        "claim_receipt": {
            "pass": bool(selected is not None and held_out_pass),
            "selected_candidate": selected,
            "interpretation": (
                "overlap-blended edge routing earns whole-call omission at 2048px"
                if selected is not None and held_out_pass
                else "no eligible <=50%-refinement overlap-blended budget preserves the declared frontier"
            ),
        },
        "cases": cases,
    }

    out = args.output_dir / "summary.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n", json.dumps({
        "validation": validation,
        "selected_candidate": selected,
        "held_out": held_out,
        "claim_receipt": payload["claim_receipt"],
    }, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
