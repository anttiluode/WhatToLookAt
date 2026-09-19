#!/usr/bin/env python3
"""High-resolution Gate H0: skip whole 512px diffusion tiles at 2048px.

The 512px SDXL-Turbo speed branch is closed. Its final engineering result is an
fp16-friendly full SDXL VAE: ~1.36-1.38x faster with ~99.5-99.7% recovery.

H0 moves WhatToLookAt to a regime where omission changes economics directly.

For each case:
  1. generate one 512px source image;
  2. Lanczos-upscale it to 2048px;
  3. refine all sixteen independent 512px tiles with the same SDXL-Turbo
     img2img operation using the fp16-fix VAE;
  4. treat that all-tile output as the tiled-refinement teacher;
  5. ask whether cheap edge energy on the 2048px base can identify which tile
     calls are worth keeping.

Because every tile is independent in both teacher and selective routes, skipping
one tile literally deletes one full VAE+UNet+VAE diffusion invocation. There is
no sparse kernel, crop-trajectory mismatch, or hidden global full-frame call.

Candidate budgets:
    refine 4/16  (skip 75%)
    refine 8/16  (skip 50%)
    refine 12/16 (skip 25%, diagnostic only; ineligible for the main claim)

Random selection is evaluated with 64 matched-budget draws per case using the
already-computed teacher tiles. Oracle selection ranks tiles by actual teacher
correction energy and is a ceiling only.

Validation seeds 100/101 choose the smallest passing eligible budget. Seed 102
is held out. A pass therefore means a cheap selector has converted into actual
whole-call omission at a useful quality/speed frontier.
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
    correction_capture,
    edge_energy_scores,
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
GRID = 4
TILE_SIZE = OUTPUT_SIZE // GRID
REFINE_COUNTS = (4, 8, 12)
RANDOM_REPEATS = 64

VALIDATION_SEEDS = (100, 101)
HELD_OUT_SEEDS = (102,)

MIN_RECOVERY = 0.80
MIN_LAYOUT_PSNR_DB = 30.0
MIN_EDGE_CORRELATION = 0.90
MIN_EDGE_RANDOM_MARGIN = 0.10
MIN_SPEEDUP = 1.75
MAX_REFINE_FRACTION = 0.50


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
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


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
    x = arr(image)
    gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    return np.sqrt(gx * gx + gy * gy).ravel()


def edge_corr(a: Image.Image, b: Image.Image) -> float:
    x, y = edges(a), edges(b)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float(np.allclose(x, y))
    return float(np.corrcoef(x, y)[0, 1])


def tile_box(index: int) -> tuple[int, int, int, int]:
    row, col = divmod(int(index), GRID)
    x0 = col * TILE_SIZE
    y0 = row * TILE_SIZE
    return (x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)


def assemble(base: Image.Image, teacher_tiles: list[Image.Image], selected: list[int]) -> Image.Image:
    out = base.convert("RGB").copy()
    for index in selected:
        out.paste(teacher_tiles[int(index)], tile_box(int(index))[:2])
    return out


def correction_energy(base: Image.Image, teacher_tiles: list[Image.Image]) -> np.ndarray:
    values = []
    for index, tile in enumerate(teacher_tiles):
        base_tile = base.crop(tile_box(index))
        values.append(mse(base_tile, tile))
    return np.asarray(values, dtype=np.float64)


def make_contact(
    base: Image.Image,
    teacher: Image.Image,
    variants: list[tuple[str, Image.Image]],
    path: Path,
):
    items = [("2048 base", base), ("all 16 tiles", teacher)] + variants
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
    canvas.save(path)


def random_distribution(
    energy: np.ndarray,
    count: int,
    seed: int,
    repeats: int = RANDOM_REPEATS,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    values = []
    total = len(energy)
    for _ in range(int(repeats)):
        selected = [int(i) for i in rng.choice(total, int(count), replace=False)]
        values.append(correction_capture(energy, selected))
    return np.asarray(values, dtype=np.float64)


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

    return {
        "cases": len(rows),
        "refine_count": int(count),
        "refine_fraction": float(count / (GRID * GRID)),
        "skip_fraction": float(1.0 - count / (GRID * GRID)),
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
        "mean_oracle_recovery": float(np.mean([row["oracle_recovery"] for row in rows])),
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
        default=Path("results/highres_tile_h0"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("H0 requires CUDA.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = tuple(args.prompt) if args.prompt else PROMPTS

    print("GPU:", torch.cuda.get_device_name(0))
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

            teacher_tiles = []
            tile_seconds = []
            for tile_index in range(GRID * GRID):
                tile = base.crop(tile_box(tile_index))
                refined, dt = timed(lambda tile=tile, tile_index=tile_index: pipe_i2i(
                    prompt=prompt,
                    image=tile,
                    num_inference_steps=args.steps,
                    strength=args.strength,
                    guidance_scale=0.0,
                    generator=gen(seed + 100000 + tile_index),
                ).images[0])
                teacher_tiles.append(refined)
                tile_seconds.append(float(dt))

            teacher = assemble(
                base,
                teacher_tiles,
                list(range(GRID * GRID)),
            )
            teacher_seconds = float(sum(tile_seconds))

            base_np = arr(base)
            edge_scores = edge_energy_scores(base_np, GRID)
            energy = correction_energy(base, teacher_tiles)

            budgets = {}
            contact_variants = []

            for count in REFINE_COUNTS:
                selector_t0 = time.perf_counter()
                edge_selected = top_indices(edge_scores, count)
                selector_seconds = time.perf_counter() - selector_t0

                edge_image = assemble(base, teacher_tiles, edge_selected)
                edge_recovery = correction_capture(energy, edge_selected)

                random_values = random_distribution(
                    energy,
                    count,
                    seed=seed + 700000 + count,
                )
                oracle_selected = top_indices(energy, count)
                oracle_recovery = correction_capture(energy, oracle_selected)

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
                    "random_median_recovery": float(np.median(random_values)),
                    "random_p10_recovery": float(np.percentile(random_values, 10)),
                    "random_p90_recovery": float(np.percentile(random_values, 90)),
                    "oracle_selected_tiles": oracle_selected,
                    "oracle_recovery": float(oracle_recovery),
                    "layout_psnr_db": layout_psnr(edge_image, teacher),
                    "edge_correlation": edge_corr(edge_image, teacher),
                    "selector_seconds": float(selector_seconds),
                    "selected_tile_seconds": selected_tile_seconds,
                    "estimated_route_seconds_from_measured_tile_calls": route_seconds,
                    "all_tile_seconds": teacher_seconds,
                    "measured_speedup": speedup,
                }
                contact_variants.append(
                    (f"edge {count}/16", edge_image)
                )
                print(
                    f"{count:2d}/16 edge recovery {edge_recovery:.3f}  "
                    f"random-med {np.median(random_values):.3f}  "
                    f"oracle {oracle_recovery:.3f}  "
                    f"speedup {speedup:.2f}x"
                )

            contact = args.output_dir / f"{stem}_contact.jpg"
            make_contact(base, teacher, contact_variants, contact)

            cases.append({
                "prompt": prompt,
                "seed": int(seed),
                "source_seconds": float(source_s),
                "all_tile_seconds": teacher_seconds,
                "mean_tile_seconds": float(np.mean(tile_seconds)),
                "tile_seconds": tile_seconds,
                "edge_scores": [float(x) for x in edge_scores],
                "correction_energy": [float(x) for x in energy],
                "budgets": budgets,
                "contact_sheet": str(contact),
            })

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
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "vae": FP16_VAE_ID,
        "source_size": SOURCE_SIZE,
        "output_size": OUTPUT_SIZE,
        "grid": GRID,
        "tile_size": TILE_SIZE,
        "refine_counts": list(REFINE_COUNTS),
        "random_repeats": RANDOM_REPEATS,
        "validation_seeds": list(VALIDATION_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "thresholds": thresholds,
        "timing_note": (
            "Selective route time is selector CPU time plus the actual measured "
            "per-tile diffusion times of the selected independent calls from the "
            "all-tile pass. Because tile calls are independent and deterministic, "
            "skipping a tile deletes that measured call exactly."
        ),
        "scope_note": (
            "H0 uses reproducible generated photographic scenes. A pass earns "
            "a natural-photo follow-up; it is not yet a claim about arbitrary "
            "real 2K/4K photographs."
        ),
        "validation": validation,
        "selected_candidate": selected,
        "held_out": held_out,
        "claim_receipt": {
            "pass": bool(selected is not None and held_out_pass),
            "selected_candidate": selected,
            "interpretation": (
                "cheap edge evidence earns whole-tile diffusion omission at 2048px"
                if selected is not None and held_out_pass
                else "no eligible <=50%-refinement budget preserves the declared quality/speed frontier"
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
