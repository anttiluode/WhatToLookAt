#!/usr/bin/env python3
"""SDXL 512px engineering closeout: replace the VAE, not the routing algorithm.

The component profile found:
    VAE encode  ~168 ms
    VAE decode  ~277 ms
    VAE total   ~445 ms = 58.4% of a ~763 ms stage
    UNet total  ~238 ms = 31.2%

That earns one final engineering benchmark.

Candidates:
- madebyollin/sdxl-vae-fp16-fix (full SDXL AutoencoderKL, fp16-friendly)
- madebyollin/taesdxl (AutoencoderTiny)

The ordinary SDXL-Turbo VAE is the reference teacher. The same prompt, base
image, scheduler, stage seed and UNet are used. Only the VAE is replaced.

Seeds 100/101 are validation. The fastest candidate that satisfies the fixed
quality+speed frontier is selected. Seed 102 is then held out.

This is not a WhatToLookAt novelty claim. It is the engineering closeout forced
by the measured bottleneck.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    AutoencoderKL,
    AutoencoderTiny,
)

from profile_sdxl_stage import run_stage_profile
from vae_closeout_utils import select_fastest_vae, vae_candidate_pass

MODEL_ID = "stabilityai/sdxl-turbo"
FP16_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
TINY_VAE_ID = "madebyollin/taesdxl"

PROMPTS = (
    "Oil painting of an ancient stone castle surrounded by stormy ocean waves",
    "A futuristic solar bicycle parked on a Finnish forest gravel road, morning mist",
    "A cyberpunk cat wearing reflective sunglasses in a neon alley, cinematic lighting",
)

VALIDATION_SEEDS = (100, 101)
HELD_OUT_SEEDS = (102,)

MIN_MEAN_RECOVERY = 0.85
MIN_LAYOUT_PSNR_DB = 28.0
MIN_EDGE_CORRELATION = 0.90
MIN_STAGE_SPEEDUP = 1.30


def gen(seed: int):
    return torch.Generator(device="cuda").manual_seed(int(seed))


def sync():
    torch.cuda.synchronize()


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
        a.resize((64, 64), Image.Resampling.BILINEAR),
        b.resize((64, 64), Image.Resampling.BILINEAR),
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


def recovery(base: Image.Image, candidate: Image.Image, teacher: Image.Image) -> float:
    base_err = mse(base, teacher)
    if base_err <= 1e-20:
        return 0.0
    return float(1.0 - mse(candidate, teacher) / base_err)


def sheet(items, path: Path) -> None:
    thumb = 256
    label_h = 28
    cols = min(4, len(items))
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
        draw.text((x0 + 4, y0 + thumb + 5), label, fill="black")
    canvas.save(path)


def required_faster_cases(n: int) -> int:
    return int(math.ceil(2 * int(n) / 3))


def summarize(cases: list[dict], seed_set: set[int], variant: str) -> dict:
    rows = [
        case["variants"][variant]
        for case in cases
        if int(case["seed"]) in seed_set
    ]
    baseline = [
        case["baseline"]
        for case in cases
        if int(case["seed"]) in seed_set
    ]

    stage = np.asarray([row["stage_seconds"] for row in rows], dtype=float)
    base_stage = np.asarray([row["stage_seconds"] for row in baseline], dtype=float)
    recovery_values = np.asarray([row["recovery"] for row in rows], dtype=float)
    layout_values = np.asarray(
        [row["layout_psnr_to_teacher_db"] for row in rows], dtype=float
    )
    edge_values = np.asarray(
        [row["edge_correlation_to_teacher"] for row in rows], dtype=float
    )

    vae_encode = np.asarray(
        [row["profile"]["components"]["vae_encode"]["cuda_ms"] for row in rows],
        dtype=float,
    )
    vae_decode = np.asarray(
        [row["profile"]["components"]["vae_decode"]["cuda_ms"] for row in rows],
        dtype=float,
    )
    unet = np.asarray(
        [row["profile"]["components"]["unet"]["cuda_ms"] for row in rows],
        dtype=float,
    )

    return {
        "cases": len(rows),
        "mean_recovery": float(recovery_values.mean()),
        "min_recovery": float(recovery_values.min()),
        "mean_psnr_to_teacher_db": float(
            np.mean([row["psnr_to_teacher_db"] for row in rows])
        ),
        "mean_layout_psnr_to_teacher_db": float(layout_values.mean()),
        "min_layout_psnr_to_teacher_db": float(layout_values.min()),
        "mean_edge_correlation_to_teacher": float(edge_values.mean()),
        "min_edge_correlation_to_teacher": float(edge_values.min()),
        "mean_stage_seconds": float(stage.mean()),
        "baseline_mean_stage_seconds": float(base_stage.mean()),
        "mean_stage_speedup": float(base_stage.mean() / stage.mean()),
        "faster_case_count": int(np.sum(stage < base_stage)),
        "mean_vae_encode_ms": float(vae_encode.mean()),
        "mean_vae_decode_ms": float(vae_decode.mean()),
        "mean_vae_total_ms": float((vae_encode + vae_decode).mean()),
        "mean_unet_ms": float(unet.mean()),
    }


def load_variant(name: str, local_only: bool):
    if name == "fp16_fix":
        return AutoencoderKL.from_pretrained(
            FP16_VAE_ID,
            dtype=torch.float16,
            local_files_only=local_only,
        ).to("cuda")
    if name == "taesdxl":
        return AutoencoderTiny.from_pretrained(
            TINY_VAE_ID,
            dtype=torch.float16,
            local_files_only=local_only,
        ).to("cuda")
    raise ValueError(name)


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
        default=Path("results/sdxl_vae_closeout"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = tuple(args.prompt) if args.prompt else PROMPTS

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading SDXL-Turbo reference pipeline...")

    pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        variant="fp16",
        local_files_only=args.local_only,
    ).to("cuda")
    pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i).to("cuda")
    pipe_t2i.set_progress_bar_config(disable=True)
    pipe_i2i.set_progress_bar_config(disable=True)

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

    # Phase 1: build every base and the ordinary-VAE teacher first.
    for pidx, prompt in enumerate(prompts, start=1):
        for seed in args.seeds:
            stem = f"p{pidx:02d}_s{seed}"
            print("\nreference", stem)

            draft = pipe_t2i(
                prompt=prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
                width=256,
                height=256,
                generator=gen(seed),
            ).images[0]
            init = draft.resize((512, 512), Image.Resampling.BILINEAR)

            base = pipe_i2i(
                prompt=prompt,
                image=init,
                num_inference_steps=args.base_steps,
                strength=args.strength,
                guidance_scale=0.0,
                generator=gen(seed + 100000),
            ).images[0]

            teacher, profile = run_stage_profile(
                pipe_i2i,
                prompt,
                base,
                args.stage_steps,
                args.strength,
                seed + 300000,
            )

            draft.save(args.output_dir / f"{stem}_draft.png")
            base.save(args.output_dir / f"{stem}_base.png")
            teacher.save(args.output_dir / f"{stem}_teacher.png")

            cases.append({
                "prompt": prompt,
                "prompt_index": int(pidx),
                "seed": int(seed),
                "stem": stem,
                "base_path": str(args.output_dir / f"{stem}_base.png"),
                "teacher_path": str(args.output_dir / f"{stem}_teacher.png"),
                "baseline": {
                    "stage_seconds": float(profile["wall_ms"] / 1000.0),
                    "profile": profile,
                },
                "variants": {},
            })

            print(
                f"standard VAE stage {profile['wall_ms'] / 1000.0:.3f}s  "
                f"VAE {profile['components']['vae_encode']['cuda_ms'] + profile['components']['vae_decode']['cuda_ms']:.1f}ms"
            )

    # We no longer need the original VAE on CUDA after teachers are fixed.
    standard_vae = pipe_i2i.vae
    standard_vae.to("cpu")
    torch.cuda.empty_cache()

    for variant_name in ("fp16_fix", "taesdxl"):
        print("\nLoading", variant_name)
        variant_vae = load_variant(variant_name, args.local_only)
        pipe_i2i.vae = variant_vae

        for case in cases:
            prompt = case["prompt"]
            seed = int(case["seed"])
            stem = case["stem"]
            base = Image.open(case["base_path"]).convert("RGB")
            teacher = Image.open(case["teacher_path"]).convert("RGB")

            candidate, profile = run_stage_profile(
                pipe_i2i,
                prompt,
                base,
                args.stage_steps,
                args.strength,
                seed + 300000,
            )
            candidate.save(
                args.output_dir / f"{stem}_{variant_name}.png"
            )

            row = {
                "stage_seconds": float(profile["wall_ms"] / 1000.0),
                "speedup_vs_standard": float(
                    case["baseline"]["stage_seconds"]
                    / (profile["wall_ms"] / 1000.0)
                ),
                "recovery": recovery(base, candidate, teacher),
                "psnr_to_teacher_db": psnr(candidate, teacher),
                "layout_psnr_to_teacher_db": layout_psnr(candidate, teacher),
                "edge_correlation_to_teacher": edge_corr(candidate, teacher),
                "profile": profile,
            }
            case["variants"][variant_name] = row

            print(
                f"{stem} {variant_name:9s} "
                f"{row['stage_seconds']:.3f}s "
                f"{row['speedup_vs_standard']:.2f}x  "
                f"recovery {row['recovery']:+.3f}  "
                f"layout {row['layout_psnr_to_teacher_db']:.2f}dB  "
                f"edge {row['edge_correlation_to_teacher']:.3f}"
            )

        pipe_i2i.vae = standard_vae
        variant_vae.to("cpu")
        del variant_vae
        torch.cuda.empty_cache()

    # Contact sheets after both alternatives are available.
    for case in cases:
        stem = case["stem"]
        items = [
            ("base", Image.open(case["base_path"]).convert("RGB")),
            ("standard teacher", Image.open(case["teacher_path"]).convert("RGB")),
            (
                "fp16-fix VAE",
                Image.open(args.output_dir / f"{stem}_fp16_fix.png").convert("RGB"),
            ),
            (
                "TAESDXL",
                Image.open(args.output_dir / f"{stem}_taesdxl.png").convert("RGB"),
            ),
        ]
        contact = args.output_dir / f"{stem}_contact.png"
        sheet(items, contact)
        case["contact_sheet"] = str(contact)

    variants = ("fp16_fix", "taesdxl")
    validation = {
        name: summarize(cases, set(VALIDATION_SEEDS), name)
        for name in variants
    }

    n_validation = sum(
        int(case["seed"]) in VALIDATION_SEEDS
        for case in cases
    )
    required_validation_faster = required_faster_cases(n_validation)

    for summary in validation.values():
        summary["pass"] = vae_candidate_pass(
            summary,
            MIN_MEAN_RECOVERY,
            MIN_LAYOUT_PSNR_DB,
            MIN_EDGE_CORRELATION,
            MIN_STAGE_SPEEDUP,
            required_validation_faster,
        )

    selected = select_fastest_vae(
        validation,
        MIN_MEAN_RECOVERY,
        MIN_LAYOUT_PSNR_DB,
        MIN_EDGE_CORRELATION,
        MIN_STAGE_SPEEDUP,
        required_validation_faster,
    )

    held_out = None
    held_out_pass = False
    if selected is not None:
        held_out = summarize(cases, set(HELD_OUT_SEEDS), selected)
        n_test = sum(
            int(case["seed"]) in HELD_OUT_SEEDS
            for case in cases
        )
        required_test_faster = required_faster_cases(n_test)
        held_out_pass = vae_candidate_pass(
            held_out,
            MIN_MEAN_RECOVERY,
            MIN_LAYOUT_PSNR_DB,
            MIN_EDGE_CORRELATION,
            MIN_STAGE_SPEEDUP,
            required_test_faster,
        )
        held_out["pass"] = held_out_pass

    payload = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "reference_vae": "SDXL-Turbo bundled AutoencoderKL",
        "candidates": {
            "fp16_fix": FP16_VAE_ID,
            "taesdxl": TINY_VAE_ID,
        },
        "validation_seeds": list(VALIDATION_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "thresholds": {
            "min_mean_recovery": MIN_MEAN_RECOVERY,
            "min_layout_psnr_db": MIN_LAYOUT_PSNR_DB,
            "min_edge_correlation": MIN_EDGE_CORRELATION,
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
                "a known VAE substitution earns a quality-preserving engineering speedup"
                if selected is not None and held_out_pass
                else "no tested VAE substitution earns the fixed quality/speed frontier"
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
