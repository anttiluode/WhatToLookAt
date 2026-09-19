#!/usr/bin/env python3
"""Practical router P4: preserve the full frame and cache U-Net features across timesteps.

P3 killed hard spatial U-Net tiling:
- every halo had negative recovery on validation;
- even halo 8 (100% naive spatial elements) failed;
- every sparse-crop route was slower than the ordinary full-frame teacher.

That says the useful full-frame interaction graph should not be cut apart.

P4 therefore switches axes. It reproduces DeepCache on the *same* SDXL-Turbo
img2img continuation used by P2/P3. DeepCache is prior work (Ma, Fang & Wang,
CVPR 2024), not a WhatToLookAt invention. It reuses high-level U-Net features
across adjacent denoising steps while leaving the image frame global.

This gate asks only:
    Does feature reuse give a real quality-preserving wall-clock win on this
    RTX-3060 / current-diffusers / SDXL-Turbo route?

Candidates:
    cache_interval = 2
    cache_branch_id in {0, 1, 2}

Protocol:
    seeds 100/101 = validation
    choose the fastest candidate that passes quality + speed
    seed 102 = untouched held-out evaluation

If P4 passes, P5 may ask the WhatToLookAt question again: can a cheap observable
choose how aggressively to refresh the cache? If P4 itself fails on this short
Turbo trajectory, the practical branch should not pretend caching is useful here
just because it works on longer diffusion samplers.
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

from feature_cache_gate_utils import cache_candidate_pass, select_fastest_passing

try:
    from DeepCache import DeepCacheSDHelper
except ImportError as exc:  # pragma: no cover - exercised on user's GPU env
    raise SystemExit(
        "P4 requires DeepCache. Install it with: pip install DeepCache"
    ) from exc


MODEL_ID = "stabilityai/sdxl-turbo"
PROMPTS = (
    "Oil painting of an ancient stone castle surrounded by stormy ocean waves",
    "A futuristic solar bicycle parked on a Finnish forest gravel road, morning mist",
    "A cyberpunk cat wearing reflective sunglasses in a neon alley, cinematic lighting",
)
CACHE_INTERVAL = 2
BRANCH_IDS = (0, 1, 2)
VALIDATION_SEEDS = (100, 101)
HELD_OUT_SEEDS = (102,)

MIN_MEAN_RECOVERY = 0.80
MIN_PSNR_DB = 28.0
MIN_STAGE_SPEEDUP = 1.10


def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t0


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


def recovery(base, candidate, teacher):
    base_err = mse(base, teacher)
    if base_err <= 1e-20:
        return 0.0
    return float(1.0 - mse(candidate, teacher) / base_err)


def sheet(items, path):
    thumb = 256
    label_h = 28
    cols = min(4, len(items))
    rows = math.ceil(len(items) / cols)
    canvas = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(items):
        r, c = divmod(i, cols)
        x0, y0 = c * thumb, r * (thumb + label_h)
        canvas.paste(
            image.resize((thumb, thumb), Image.Resampling.LANCZOS),
            (x0, y0),
        )
        draw.text((x0 + 4, y0 + thumb + 5), label, fill="black")
    canvas.save(path)


def run_cached(pipe, helper, prompt, base, steps, strength, seed, branch_id):
    helper.set_params(
        cache_interval=CACHE_INTERVAL,
        cache_branch_id=int(branch_id),
    )
    helper.enable()
    try:
        return timed(lambda: pipe(
            prompt=prompt,
            image=base,
            num_inference_steps=int(steps),
            strength=float(strength),
            guidance_scale=0.0,
            generator=gen(seed),
        ).images[0])
    finally:
        helper.disable()


def summarize(cases, seed_set, branch_id):
    rows = [
        case["cache"][str(int(branch_id))]
        for case in cases
        if int(case["seed"]) in seed_set
    ]
    teacher_times = np.asarray([
        case["teacher_stage_seconds"]
        for case in cases
        if int(case["seed"]) in seed_set
    ])
    cache_times = np.asarray([row["stage_seconds"] for row in rows])
    recovery_values = np.asarray([row["recovery"] for row in rows])
    psnr_values = np.asarray([row["psnr_to_teacher_db"] for row in rows])

    return {
        "cases": len(rows),
        "branch_id": int(branch_id),
        "cache_interval": CACHE_INTERVAL,
        "mean_recovery": float(recovery_values.mean()),
        "min_recovery": float(recovery_values.min()),
        "mean_psnr_to_teacher_db": float(psnr_values.mean()),
        "min_psnr_to_teacher_db": float(psnr_values.min()),
        "mean_stage_seconds": float(cache_times.mean()),
        "teacher_mean_stage_seconds": float(teacher_times.mean()),
        "mean_stage_speedup": float(teacher_times.mean() / cache_times.mean()),
        "faster_case_count": int(np.sum(cache_times < teacher_times)),
    }


def required_faster_cases(n):
    return int(math.ceil(2 * int(n) / 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    ap.add_argument("--base-steps", type=int, default=2)
    ap.add_argument("--stage-steps", type=int, default=4)
    ap.add_argument("--strength", type=float, default=0.60)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/practical_diffusion_router_p4"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("P4 requires CUDA.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = tuple(args.prompt) if args.prompt else PROMPTS

    pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=args.local_only,
    ).to("cuda")
    pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i).to("cuda")
    pipe_t2i.set_progress_bar_config(disable=True)
    pipe_i2i.set_progress_bar_config(disable=True)

    helper = DeepCacheSDHelper(pipe=pipe_i2i)

    print("GPU:", torch.cuda.get_device_name(0))
    print("DeepCache interval:", CACHE_INTERVAL, "branches:", BRANCH_IDS)
    print("Warmup...")
    _ = pipe_t2i(
        prompt="warmup",
        num_inference_steps=1,
        guidance_scale=0.0,
        width=256,
        height=256,
        generator=gen(1),
    ).images[0]
    sync()

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
            init = draft.resize((512, 512), Image.Resampling.BILINEAR)

            base, base_s = timed(lambda: pipe_i2i(
                prompt=prompt,
                image=init,
                num_inference_steps=args.base_steps,
                strength=args.strength,
                guidance_scale=0.0,
                generator=gen(seed + 100000),
            ).images[0])

            stage_seed = seed + 300000
            teacher, teacher_s = timed(lambda: pipe_i2i(
                prompt=prompt,
                image=base,
                num_inference_steps=args.stage_steps,
                strength=args.strength,
                guidance_scale=0.0,
                generator=gen(stage_seed),
            ).images[0])

            cache = {}
            items = [("draft", draft), ("base", base), ("teacher", teacher)]

            for branch_id in BRANCH_IDS:
                cached, dt = run_cached(
                    pipe_i2i,
                    helper,
                    prompt,
                    base,
                    args.stage_steps,
                    args.strength,
                    stage_seed,
                    branch_id,
                )
                cached.save(
                    args.output_dir / f"{stem}_cache_b{branch_id}.png"
                )
                row = {
                    "branch_id": int(branch_id),
                    "cache_interval": CACHE_INTERVAL,
                    "stage_seconds": float(dt),
                    "speedup_vs_teacher_stage": float(teacher_s / dt),
                    "recovery": recovery(base, cached, teacher),
                    "psnr_to_teacher_db": psnr(cached, teacher),
                }
                cache[str(branch_id)] = row
                items.append((f"cache b{branch_id}", cached))
                print(
                    f"branch {branch_id}  "
                    f"recovery {row['recovery']:+.3f}  "
                    f"PSNR {row['psnr_to_teacher_db']:.2f} dB  "
                    f"{dt:.3f}s  speedup {row['speedup_vs_teacher_stage']:.2f}x"
                )

            draft.save(args.output_dir / f"{stem}_draft.png")
            base.save(args.output_dir / f"{stem}_base.png")
            teacher.save(args.output_dir / f"{stem}_teacher.png")
            contact = args.output_dir / f"{stem}_contact.png"
            sheet(items, contact)

            cases.append({
                "prompt": prompt,
                "seed": int(seed),
                "draft_seconds": float(draft_s),
                "base_seconds": float(base_s),
                "teacher_stage_seconds": float(teacher_s),
                "base_mse_to_teacher": mse(base, teacher),
                "cache": cache,
                "contact_sheet": str(contact),
            })

    validation = {
        f"branch_{branch_id}": summarize(
            cases, set(VALIDATION_SEEDS), branch_id
        )
        for branch_id in BRANCH_IDS
    }
    req_val = required_faster_cases(
        sum(int(c["seed"]) in VALIDATION_SEEDS for c in cases)
    )
    for summary in validation.values():
        summary["pass"] = cache_candidate_pass(
            summary,
            MIN_MEAN_RECOVERY,
            MIN_PSNR_DB,
            MIN_STAGE_SPEEDUP,
            req_val,
        )

    selected = select_fastest_passing(
        validation,
        MIN_MEAN_RECOVERY,
        MIN_PSNR_DB,
        MIN_STAGE_SPEEDUP,
        req_val,
    )

    held_out = None
    held_out_pass = False
    if selected is not None:
        branch_id = int(selected.split("_")[-1])
        held_out = summarize(
            cases, set(HELD_OUT_SEEDS), branch_id
        )
        req_test = required_faster_cases(
            sum(int(c["seed"]) in HELD_OUT_SEEDS for c in cases)
        )
        held_out_pass = cache_candidate_pass(
            held_out,
            MIN_MEAN_RECOVERY,
            MIN_PSNR_DB,
            MIN_STAGE_SPEEDUP,
            req_test,
        )
        held_out["pass"] = held_out_pass

    payload = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "prior_work": {
            "method": "DeepCache",
            "paper": "Ma, Fang & Wang, DeepCache: Accelerating Diffusion Models for Free, CVPR 2024",
            "note": "P4 is a reproduction/compatibility gate on this exact route, not a novelty claim.",
        },
        "cache_interval": CACHE_INTERVAL,
        "candidate_branch_ids": list(BRANCH_IDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "thresholds": {
            "min_mean_recovery": MIN_MEAN_RECOVERY,
            "min_psnr_db": MIN_PSNR_DB,
            "min_stage_speedup": MIN_STAGE_SPEEDUP,
            "faster_case_fraction": "ceil(2N/3)",
        },
        "validation": validation,
        "selected_candidate": selected,
        "held_out": held_out,
        "claim_receipt": {
            "pass": bool(selected is not None and held_out_pass),
            "selected_candidate": selected,
            "interpretation": (
                "feature caching is a viable acceleration substrate on this route"
                if selected is not None and held_out_pass
                else "no tested DeepCache branch earned the quality/speed frontier on this short Turbo route"
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
