#!/usr/bin/env python3
"""Practical GPU Gate P0: can a cheap spatial cue route diffusion compute?

This gate is the direct handoff from Sihti AI-G1 to WhatToLookAt.

Sihti AI-G1 found that Sihti residue was *not* the best predictor of where a
4-step SDXL-Turbo teacher differed from a cheap 2-step refinement. Simple draft
edge energy was best on that 9-case panel.

P0 therefore refuses to force a Sihti-specific mechanism. It asks the practical
question instead:

    Can edge-selected local extra refinement recover more of the expensive
    teacher than same-compute random tiles, while remaining faster than a
    full-frame teacher route?

Route:
    256x256 1-step draft
      -> 512x512 cheap 2-step full-frame img2img base
      -> choose 4 of 16 tiles from the *draft*
      -> batch-refine 192x192 context crops around those 4 tiles
      -> paste the 128x128 centers back into the cheap base

Selectors, all with the same number of refined tiles:
    edge_energy       primary candidate from Sihti AI-G1
    local_variance    strong simple baseline
    gaussian_residue  ordinary smoothing-error baseline
    sihti_residue     historical control
    random            same-compute attacker

Teacher:
    the same 256 draft -> 512 full-frame 4-step img2img.

Primary quality quantity:
    correction_recovery =
        1 - MSE(router, teacher) / MSE(cheap_base, teacher)

A positive value means the selective route moved closer to the teacher.

Predeclared PASS:
    - edge mean recovery >= random mean recovery + 0.05
    - edge beats random recovery in >= 6/9 cases
    - edge mean end-to-end route time < teacher mean route time
    - edge route is faster than teacher in >= 6/9 cases

This is a compute-routing gate, not a claim that the 4-step teacher is ground
truth or that crop refinement is the final architecture.
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
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

from diffusion_router_utils import (
    context_crops,
    edge_scores,
    matched_gaussian_residue_scores,
    paste_refined_centers,
    random_tiles,
    sihti_residue_scores,
    top_tiles,
    variance_scores,
)

MODEL_ID = "stabilityai/sdxl-turbo"
PROMPTS = (
    "Oil painting of an ancient stone castle surrounded by stormy ocean waves",
    "A futuristic solar bicycle parked on a Finnish forest gravel road, morning mist",
    "A cyberpunk cat wearing reflective sunglasses in a neon alley, cinematic lighting",
)
GRID = 4
SELECTED_TILES = 4
CONTEXT = 32


def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync()
    t0 = time.perf_counter()
    value = fn()
    sync()
    return value, time.perf_counter() - t0


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(int(seed))


def arr(image):
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def mse(a, b):
    aa, bb = arr(a), arr(b)
    return float(np.mean((aa - bb) ** 2))


def psnr(a, b):
    value = mse(a, b)
    return 120.0 if value <= 1e-14 else float(10.0 * np.log10(1.0 / value))


def layout_psnr(a, b):
    return psnr(
        a.resize((64, 64), Image.Resampling.BILINEAR),
        b.resize((64, 64), Image.Resampling.BILINEAR),
    )


def edges(image):
    x = arr(image)
    gray = 0.299*x[:, :, 0] + 0.587*x[:, :, 1] + 0.114*x[:, :, 2]
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    return np.sqrt(gx*gx + gy*gy).ravel()


def edge_corr(a, b):
    x, y = edges(a), edges(b)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float(np.allclose(x, y))
    return float(np.corrcoef(x, y)[0, 1])


def sheet(items, path):
    thumb = 256
    label_h = 28
    cols = min(4, len(items))
    rows = math.ceil(len(items) / cols)
    canvas = Image.new("RGB", (cols*thumb, rows*(thumb+label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(items):
        r, c = divmod(i, cols)
        x0, y0 = c*thumb, r*(thumb+label_h)
        canvas.paste(image.resize((thumb, thumb), Image.Resampling.LANCZOS), (x0, y0))
        draw.text((x0+4, y0+thumb+5), label, fill="black")
    canvas.save(path)


def upscale_draft(draft):
    return draft.resize((512, 512), Image.Resampling.BILINEAR)


def img2img(pipe, prompt, image, steps, strength, seed):
    return timed(lambda: pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=int(steps),
        strength=float(strength),
        guidance_scale=0.0,
        generator=gen(seed),
    ).images[0])


def refine_selected_tiles(
    pipe,
    prompt,
    base,
    selected,
    steps,
    strength,
    seed,
):
    crops, meta = context_crops(
        base,
        selected,
        grid=GRID,
        context=CONTEXT,
    )
    generators = [
        torch.Generator(device="cuda").manual_seed(
            int(seed) + 1000 * int(tile)
        )
        for tile in selected
    ]

    refined, dt = timed(lambda: pipe(
        prompt=[prompt] * len(crops),
        image=crops,
        num_inference_steps=int(steps),
        strength=float(strength),
        guidance_scale=0.0,
        generator=generators,
    ).images)

    return paste_refined_centers(base, refined, meta), dt


def selector_scores(name, draft_np, seed):
    t0 = time.perf_counter()
    extra = {}
    if name == "edge_energy":
        scores = edge_scores(draft_np, GRID)
    elif name == "local_variance":
        scores = variance_scores(draft_np, GRID)
    elif name == "gaussian_residue":
        scores, chosen_sigma = matched_gaussian_residue_scores(draft_np, GRID)
        extra["gaussian_sigma"] = chosen_sigma
    elif name == "sihti_residue":
        scores = sihti_residue_scores(draft_np, GRID)
    elif name == "random":
        selected = random_tiles(GRID*GRID, SELECTED_TILES, seed)
        return None, selected, time.perf_counter() - t0, extra
    else:
        raise ValueError(name)

    selected = top_tiles(scores, SELECTED_TILES)
    return scores, selected, time.perf_counter() - t0, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    ap.add_argument("--base-steps", type=int, default=2)
    ap.add_argument("--teacher-steps", type=int, default=4)
    ap.add_argument("--crop-steps", type=int, default=2)
    ap.add_argument("--strength", type=float, default=0.60)
    ap.add_argument("--crop-strength", type=float, default=0.60)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/practical_diffusion_router"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This gate requires CUDA.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = tuple(args.prompt) if args.prompt else PROMPTS

    print("GPU:", torch.cuda.get_device_name(0))
    pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=args.local_only,
    ).to("cuda")
    pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i).to("cuda")
    pipe_t2i.set_progress_bar_config(disable=True)
    pipe_i2i.set_progress_bar_config(disable=True)

    _ = pipe_t2i(
        prompt="warmup",
        num_inference_steps=1,
        guidance_scale=0.0,
        width=256,
        height=256,
        generator=gen(1),
    ).images[0]
    sync()

    selectors = (
        "edge_energy",
        "local_variance",
        "gaussian_residue",
        "sihti_residue",
        "random",
    )
    cases = []

    for pidx, prompt in enumerate(prompts, start=1):
        for seed in args.seeds:
            stem = f"p{pidx:02d}_s{seed}"
            print("\n", stem, prompt)

            draft, draft_s = timed(lambda: pipe_t2i(
                prompt=prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
                width=256,
                height=256,
                generator=gen(seed),
            ).images[0])
            draft.save(args.output_dir / f"{stem}_draft.png")
            init = upscale_draft(draft)

            base, base_s = img2img(
                pipe_i2i,
                prompt,
                init,
                args.base_steps,
                args.strength,
                seed + 100000,
            )
            teacher, teacher_s = img2img(
                pipe_i2i,
                prompt,
                init,
                args.teacher_steps,
                args.strength,
                seed + 100000,
            )
            base.save(args.output_dir / f"{stem}_base.png")
            teacher.save(args.output_dir / f"{stem}_teacher.png")

            base_error = mse(base, teacher)
            draft_np = arr(draft)
            methods = []
            items = [("draft", draft), ("cheap base", base), ("teacher", teacher)]

            for selector in selectors:
                _, selected, select_s, extra = selector_scores(
                    selector,
                    draft_np,
                    seed + 700000,
                )
                routed, crop_s = refine_selected_tiles(
                    pipe_i2i,
                    prompt,
                    base,
                    selected,
                    args.crop_steps,
                    args.crop_strength,
                    seed + 200000,
                )
                routed.save(args.output_dir / f"{stem}_{selector}.png")
                remain = mse(routed, teacher)
                recovery = (
                    0.0
                    if base_error <= 1e-14
                    else 1.0 - remain / base_error
                )
                route_s = draft_s + base_s + select_s + crop_s

                row = {
                    "selector": selector,
                    "tiles": selected,
                    "selector_seconds": float(select_s),
                    "crop_refine_seconds": float(crop_s),
                    "route_seconds": float(route_s),
                    "correction_recovery": float(recovery),
                    "psnr_to_teacher_db": psnr(routed, teacher),
                    "layout_psnr_to_teacher_db": layout_psnr(routed, teacher),
                    "edge_correlation_to_teacher": edge_corr(routed, teacher),
                    **extra,
                }
                methods.append(row)
                items.append((selector, routed))

                print(
                    f"{selector:17s} recovery {recovery:+.3f}  "
                    f"route {route_s:.3f}s  crop {crop_s:.3f}s  "
                    f"tiles {selected}"
                )

            contact = args.output_dir / f"{stem}_contact.png"
            sheet(items, contact)

            case = {
                "prompt": prompt,
                "seed": int(seed),
                "draft_seconds": float(draft_s),
                "base_seconds": float(base_s),
                "teacher_seconds": float(teacher_s),
                "teacher_route_seconds": float(draft_s + teacher_s),
                "base_mse_to_teacher": float(base_error),
                "base_psnr_to_teacher_db": psnr(base, teacher),
                "base_layout_psnr_to_teacher_db": layout_psnr(base, teacher),
                "base_edge_correlation_to_teacher": edge_corr(base, teacher),
                "methods": methods,
                "contact_sheet": str(contact),
            }
            cases.append(case)

    aggregate = {}
    for selector in selectors:
        rows = [
            method
            for case in cases
            for method in case["methods"]
            if method["selector"] == selector
        ]
        aggregate[selector] = {
            "cases": len(rows),
            "mean_correction_recovery": float(
                np.mean([r["correction_recovery"] for r in rows])
            ),
            "median_correction_recovery": float(
                np.median([r["correction_recovery"] for r in rows])
            ),
            "mean_route_seconds": float(
                np.mean([r["route_seconds"] for r in rows])
            ),
            "mean_psnr_to_teacher_db": float(
                np.mean([r["psnr_to_teacher_db"] for r in rows])
            ),
            "mean_layout_psnr_to_teacher_db": float(
                np.mean([r["layout_psnr_to_teacher_db"] for r in rows])
            ),
            "mean_edge_correlation_to_teacher": float(
                np.mean([r["edge_correlation_to_teacher"] for r in rows])
            ),
        }

    edge_rows = [
        next(m for m in c["methods"] if m["selector"] == "edge_energy")
        for c in cases
    ]
    random_rows = [
        next(m for m in c["methods"] if m["selector"] == "random")
        for c in cases
    ]
    teacher_times = np.asarray([c["teacher_route_seconds"] for c in cases])
    edge_times = np.asarray([m["route_seconds"] for m in edge_rows])
    edge_recovery = np.asarray([m["correction_recovery"] for m in edge_rows])
    random_recovery = np.asarray([m["correction_recovery"] for m in random_rows])

    receipt = {
        "edge_minus_random_mean_recovery": float(
            edge_recovery.mean() - random_recovery.mean()
        ),
        "edge_recovery_case_wins": int(np.sum(edge_recovery > random_recovery)),
        "edge_mean_route_seconds": float(edge_times.mean()),
        "teacher_mean_route_seconds": float(teacher_times.mean()),
        "edge_faster_case_count": int(np.sum(edge_times < teacher_times)),
        "required_recovery_margin": 0.05,
        "required_case_wins": 6,
    }
    receipt["pass"] = bool(
        receipt["edge_minus_random_mean_recovery"] >= 0.05
        and receipt["edge_recovery_case_wins"] >= 6
        and receipt["edge_mean_route_seconds"] < receipt["teacher_mean_route_seconds"]
        and receipt["edge_faster_case_count"] >= 6
    )

    payload = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "grid": GRID,
        "selected_tiles": SELECTED_TILES,
        "context": CONTEXT,
        "base_steps": args.base_steps,
        "teacher_steps": args.teacher_steps,
        "crop_steps": args.crop_steps,
        "strength": args.strength,
        "crop_strength": args.crop_strength,
        "aggregate": aggregate,
        "claim_receipt": receipt,
        "cases": cases,
    }
    out = args.output_dir / "summary.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n", json.dumps({"aggregate": aggregate, "claim_receipt": receipt}, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
