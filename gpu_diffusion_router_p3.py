#!/usr/bin/env python3
"""Practical router P3: make P2's globally coherent update actually spatially sparse.

P2 proved the semantic mechanism:
- full-mask parity was exact;
- edge-selected 4/16 masks recovered 30.87% of the full teacher correction;
- random masks recovered 17.30%;
- edge won 9/9 cases.

But P2 still ran the full 512x512 UNet, so it could not save compute.

P3 keeps the *same global latent/noise/timestep trajectory* and the same hard
mask semantics, but evaluates the UNet only on the selected latent tiles plus a
halo. Four selected image tiles correspond to four 16x16 cells in the 64x64
SDXL latent.

Candidate latent halos:
    0 -> four 16x16 crops, naive spatial work 25% of full
    4 -> four 24x24 crops, naive spatial work 56.25%
    8 -> four 32x32 crops, naive spatial work 100%

Each crop starts from the current global latent at the exact same timestep and
noise state as P2. Only its center tile prediction is written back. This is very
different from P0: no independent image-space diffusion trajectory is launched.

Selection protocol:
- seeds 100/101 (6 prompt-seed cases) are validation;
- the smallest halo satisfying the predeclared quality+routing+speed criteria is
  selected;
- seed 102 (3 cases) is untouched held-out evaluation.

Every halo is also run with a same-compute random 4/16 selector.

P3 earns an actual sparse-execution claim only if one validation-selected halo
survives held-out while being faster than the ordinary full-frame 4-step stage.
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
    edge_scores,
    make_tile_mask,
    random_tiles,
    sparse_latent_work_fraction,
    top_tiles,
)
from masked_sdxl_img2img import MaskedStableDiffusionXLImg2ImgPipeline

MODEL_ID = "stabilityai/sdxl-turbo"
PROMPTS = (
    "Oil painting of an ancient stone castle surrounded by stormy ocean waves",
    "A futuristic solar bicycle parked on a Finnish forest gravel road, morning mist",
    "A cyberpunk cat wearing reflective sunglasses in a neon alley, cinematic lighting",
)
GRID = 4
SELECTED_TILES = 4
HALOS = (0, 4, 8)

VALIDATION_SEEDS = (100, 101)
HELD_OUT_SEEDS = (102,)

MIN_RECOVERY = 0.20
MIN_EDGE_MARGIN = 0.08
MIN_RECOVERY_RETENTION = 0.70
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


def recovery(base, routed, teacher):
    base_err = mse(base, teacher)
    if base_err <= 1e-20:
        return 0.0
    return float(1.0 - mse(routed, teacher) / base_err)


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


def build_masked_pipe(pipe):
    masked = MaskedStableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        image_encoder=getattr(pipe, "image_encoder", None),
        feature_extractor=getattr(pipe, "feature_extractor", None),
        requires_aesthetics_score=getattr(
            pipe.config, "requires_aesthetics_score", False
        ),
        force_zeros_for_empty_prompt=getattr(
            pipe.config, "force_zeros_for_empty_prompt", True
        ),
        add_watermarker=False,
    )
    masked.set_progress_bar_config(disable=True)
    return masked


def run_masked(
    masked,
    prompt,
    base,
    selected,
    steps,
    strength,
    stage_seed,
    mask_noise_seed,
    halo=None,
):
    kwargs = {}
    if halo is not None:
        kwargs = {
            "sparse_tiles": [int(x) for x in selected],
            "sparse_grid": GRID,
            "sparse_halo_latent": int(halo),
        }

    return timed(lambda: masked(
        prompt=prompt,
        image=None,
        original_image=base,
        mask=make_tile_mask(selected, size=512, grid=GRID),
        blur=0,
        blur_compose=0,
        num_inference_steps=int(steps),
        strength=float(strength),
        guidance_scale=0.0,
        generator=gen(stage_seed),
        mask_noise_seed=int(mask_noise_seed),
        **kwargs,
    ).images[0])


def rows_for(cases, seed_set, halo, selector):
    rows = []
    for case in cases:
        if int(case["seed"]) not in seed_set:
            continue
        rows.append(case["sparse"][str(int(halo))][selector])
    return rows


def full_edge_rows(cases, seed_set):
    return [
        case["full_edge"]
        for case in cases
        if int(case["seed"]) in seed_set
    ]


def summarize_candidate(cases, seed_set, halo):
    edge = rows_for(cases, seed_set, halo, "edge")
    rand = rows_for(cases, seed_set, halo, "random")
    full = full_edge_rows(cases, seed_set)

    edge_recovery = np.asarray([r["recovery"] for r in edge])
    random_recovery = np.asarray([r["recovery"] for r in rand])
    full_recovery = np.asarray([r["recovery"] for r in full])
    edge_stage = np.asarray([r["stage_seconds"] for r in edge])
    teacher_stage = np.asarray([
        case["teacher_stage_seconds"]
        for case in cases
        if int(case["seed"]) in seed_set
    ])

    full_mean = float(full_recovery.mean())
    edge_mean = float(edge_recovery.mean())
    retention = 0.0 if full_mean <= 1e-12 else edge_mean / full_mean

    return {
        "cases": len(edge),
        "halo_latent": int(halo),
        "naive_spatial_work_fraction": sparse_latent_work_fraction(
            SELECTED_TILES, GRID, int(halo), 64
        ),
        "edge_mean_recovery": edge_mean,
        "random_mean_recovery": float(random_recovery.mean()),
        "edge_minus_random_mean_recovery": float(
            edge_recovery.mean() - random_recovery.mean()
        ),
        "edge_case_wins": int(np.sum(edge_recovery > random_recovery)),
        "full_edge_mean_recovery": full_mean,
        "edge_recovery_retention": float(retention),
        "edge_mean_stage_seconds": float(edge_stage.mean()),
        "teacher_mean_stage_seconds": float(teacher_stage.mean()),
        "mean_stage_speedup_vs_teacher": float(
            teacher_stage.mean() / edge_stage.mean()
        ),
        "faster_case_count": int(np.sum(edge_stage < teacher_stage)),
        "mean_psnr_to_full_edge_db": float(np.mean([
            r["psnr_to_full_edge_db"] for r in edge
        ])),
    }


def validation_pass(summary):
    n = int(summary["cases"])
    required_wins = math.ceil(2 * n / 3)
    return bool(
        summary["edge_mean_recovery"] >= MIN_RECOVERY
        and summary["edge_minus_random_mean_recovery"] >= MIN_EDGE_MARGIN
        and summary["edge_case_wins"] >= required_wins
        and summary["edge_recovery_retention"] >= MIN_RECOVERY_RETENTION
        and summary["mean_stage_speedup_vs_teacher"] >= MIN_STAGE_SPEEDUP
        and summary["faster_case_count"] >= required_wins
    )


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
        default=Path("results/practical_diffusion_router_p3"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("P3 requires CUDA.")

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
    masked = build_masked_pipe(pipe_i2i)

    print("GPU:", torch.cuda.get_device_name(0))
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
            mask_noise_seed = seed + 900000

            teacher, teacher_s = timed(lambda: pipe_i2i(
                prompt=prompt,
                image=base,
                num_inference_steps=args.stage_steps,
                strength=args.strength,
                guidance_scale=0.0,
                generator=gen(stage_seed),
            ).images[0])

            edge_tiles = top_tiles(edge_scores(arr(draft), GRID), SELECTED_TILES)
            random_sel = random_tiles(
                GRID * GRID, SELECTED_TILES, seed + 700000
            )

            full_edge, full_edge_s = run_masked(
                masked,
                prompt,
                base,
                edge_tiles,
                args.stage_steps,
                args.strength,
                stage_seed,
                mask_noise_seed,
                halo=None,
            )
            full_edge_recovery = recovery(base, full_edge, teacher)

            sparse = {}
            items = [
                ("draft", draft),
                ("base", base),
                ("teacher", teacher),
                ("P2 edge", full_edge),
            ]

            print(
                f"P2-edge recovery {full_edge_recovery:+.3f}  "
                f"stage {full_edge_s:.3f}s  teacher {teacher_s:.3f}s"
            )

            for halo in HALOS:
                sparse[str(halo)] = {}
                for selector, selected in (
                    ("edge", edge_tiles),
                    ("random", random_sel),
                ):
                    routed, dt = run_masked(
                        masked,
                        prompt,
                        base,
                        selected,
                        args.stage_steps,
                        args.strength,
                        stage_seed,
                        mask_noise_seed,
                        halo=halo,
                    )
                    routed.save(
                        args.output_dir
                        / f"{stem}_{selector}_halo{halo}.png"
                    )
                    row = {
                        "tiles": [int(x) for x in selected],
                        "stage_seconds": float(dt),
                        "recovery": recovery(base, routed, teacher),
                        "psnr_to_teacher_db": psnr(routed, teacher),
                        "psnr_to_full_edge_db": psnr(routed, full_edge),
                        "speedup_vs_teacher_stage": float(teacher_s / dt),
                    }
                    sparse[str(halo)][selector] = row
                    print(
                        f"h{halo:02d} {selector:6s} "
                        f"recovery {row['recovery']:+.3f}  "
                        f"P2-PSNR {row['psnr_to_full_edge_db']:.2f} dB  "
                        f"{dt:.3f}s  speedup {row['speedup_vs_teacher_stage']:.2f}x"
                    )

                items.append(
                    (
                        f"edge h{halo}",
                        Image.open(
                            args.output_dir
                            / f"{stem}_edge_halo{halo}.png"
                        ).convert("RGB"),
                    )
                )

            draft.save(args.output_dir / f"{stem}_draft.png")
            base.save(args.output_dir / f"{stem}_base.png")
            teacher.save(args.output_dir / f"{stem}_teacher.png")
            full_edge.save(args.output_dir / f"{stem}_p2_edge.png")

            contact = args.output_dir / f"{stem}_contact.png"
            sheet(items, contact)

            cases.append({
                "prompt": prompt,
                "seed": int(seed),
                "draft_seconds": float(draft_s),
                "base_seconds": float(base_s),
                "teacher_stage_seconds": float(teacher_s),
                "full_edge": {
                    "stage_seconds": float(full_edge_s),
                    "recovery": float(full_edge_recovery),
                    "psnr_to_teacher_db": psnr(full_edge, teacher),
                },
                "sparse": sparse,
                "contact_sheet": str(contact),
            })

    validation = {
        str(halo): summarize_candidate(
            cases, set(VALIDATION_SEEDS), halo
        )
        for halo in HALOS
    }
    for summary in validation.values():
        summary["pass"] = validation_pass(summary)

    passing = [
        int(halo)
        for halo, summary in validation.items()
        if summary["pass"]
    ]
    selected_halo = min(passing) if passing else None

    held_out = None
    held_out_pass = False
    if selected_halo is not None:
        held_out = summarize_candidate(
            cases, set(HELD_OUT_SEEDS), selected_halo
        )
        held_out_pass = validation_pass(held_out)
        held_out["pass"] = held_out_pass

    payload = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "grid": GRID,
        "selected_tiles": SELECTED_TILES,
        "candidate_halos_latent": list(HALOS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "thresholds": {
            "min_recovery": MIN_RECOVERY,
            "min_edge_margin": MIN_EDGE_MARGIN,
            "min_recovery_retention": MIN_RECOVERY_RETENTION,
            "min_stage_speedup": MIN_STAGE_SPEEDUP,
            "case_win_fraction": "ceil(2N/3)",
        },
        "validation": validation,
        "selected_halo_latent": selected_halo,
        "held_out": held_out,
        "claim_receipt": {
            "pass": bool(selected_halo is not None and held_out_pass),
            "selected_halo_latent": selected_halo,
            "interpretation": (
                "sparse global-state UNet execution earned"
                if selected_halo is not None and held_out_pass
                else "no sparse halo earned both fidelity/routing and wall-clock speed"
            ),
        },
        "cases": cases,
    }

    out = args.output_dir / "summary.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n", json.dumps({
        "validation": validation,
        "selected_halo_latent": selected_halo,
        "held_out": held_out,
        "claim_receipt": payload["claim_receipt"],
    }, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
