#!/usr/bin/env python3
"""Practical router P2: preserve one global diffusion trajectory, mask the update.

P0 launched selected crops as independent img2img trajectories. P1 showed that
the *where* signal survived, but those crop actions were badly misaligned with
the full-frame correction.

P2 changes only the actuator.

For every case:
  1. make the same 256 draft;
  2. make a cheap 512 base from that draft;
  3. from the cheap base, define a new full-frame 4-step img2img teacher;
  4. run the *same* 4-step latent trajectory through a masked SDXL img2img
     pipeline that keeps non-selected latents tied to the base image while the
     selected region denoises in the full 512x512 UNet context.

Masks:
  - edge: 4/16 tiles selected from draft edge energy
  - random: same 4/16 budget
  - oracle: 4/16 tiles with largest teacher correction energy (ceiling)
  - full: all pixels, used only to verify trajectory parity

This gate deliberately does NOT claim a speedup: masked P2 still evaluates the
full-frame UNet. It asks whether global-state masking restores a valid local
actuation semantics. Only after that is earned should we optimize spatial
compute inside the UNet.
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

from diffusion_router_utils import edge_scores, random_tiles, top_tiles
from masked_sdxl_img2img import MaskedStableDiffusionXLImg2ImgPipeline

MODEL_ID = "stabilityai/sdxl-turbo"
PROMPTS = (
    "Oil painting of an ancient stone castle surrounded by stormy ocean waves",
    "A futuristic solar bicycle parked on a Finnish forest gravel road, morning mist",
    "A cyberpunk cat wearing reflective sunglasses in a neon alley, cinematic lighting",
)
GRID = 4
SELECTED_TILES = 4


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


def tile_energy(base, teacher):
    err = np.mean((arr(teacher) - arr(base)) ** 2, axis=2)
    h, w = err.shape
    th, tw = h // GRID, w // GRID
    return err.reshape(GRID, th, GRID, tw).mean(axis=(1, 3)).ravel()


def capture(energy, selected):
    total = float(np.sum(energy))
    if total <= 1e-20:
        return SELECTED_TILES / (GRID * GRID)
    return float(np.sum(energy[selected]) / total)


def tile_mask(selected, size=512):
    tile = size // GRID
    m = np.zeros((size, size), dtype=np.uint8)
    for idx in selected:
        row, col = divmod(int(idx), GRID)
        m[row*tile:(row+1)*tile, col*tile:(col+1)*tile] = 255
    return Image.fromarray(m, mode="L")


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
        requires_aesthetics_score=getattr(pipe.config, "requires_aesthetics_score", False),
        force_zeros_for_empty_prompt=getattr(pipe.config, "force_zeros_for_empty_prompt", True),
        add_watermarker=False,
    )
    masked.set_progress_bar_config(disable=True)
    return masked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    ap.add_argument("--base-steps", type=int, default=2)
    ap.add_argument("--masked-steps", type=int, default=4)
    ap.add_argument("--strength", type=float, default=0.60)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/practical_diffusion_router_p2"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("P2 requires CUDA.")

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
            teacher, teacher_s = timed(lambda: pipe_i2i(
                prompt=prompt,
                image=base,
                num_inference_steps=args.masked_steps,
                strength=args.strength,
                guidance_scale=0.0,
                generator=gen(stage_seed),
            ).images[0])

            energy = tile_energy(base, teacher)
            edge_tiles = top_tiles(edge_scores(arr(draft), GRID), SELECTED_TILES)
            random_sel = random_tiles(GRID*GRID, SELECTED_TILES, seed + 700000)
            oracle_tiles = [int(i) for i in np.argsort(energy)[-SELECTED_TILES:][::-1]]

            masks = {
                "edge": tile_mask(edge_tiles),
                "random": tile_mask(random_sel),
                "oracle": tile_mask(oracle_tiles),
                "full": Image.new("L", (512, 512), 255),
            }
            selections = {
                "edge": edge_tiles,
                "random": random_sel,
                "oracle": oracle_tiles,
                "full": list(range(GRID*GRID)),
            }

            methods = {}
            items = [
                ("draft", draft),
                ("cheap base", base),
                ("full teacher", teacher),
            ]

            for name in ("edge", "random", "oracle", "full"):
                out, dt = timed(lambda name=name: masked(
                    prompt=prompt,
                    image=None,
                    original_image=base,
                    mask=masks[name],
                    blur=0,
                    blur_compose=0,
                    num_inference_steps=args.masked_steps,
                    strength=args.strength,
                    guidance_scale=0.0,
                    generator=gen(stage_seed),
                    mask_noise_seed=seed + 900000,
                ).images[0])

                out.save(args.output_dir / f"{stem}_{name}.png")
                methods[name] = {
                    "tiles": selections[name],
                    "stage_seconds": float(dt),
                    "recovery": recovery(base, out, teacher),
                    "psnr_to_teacher_db": psnr(out, teacher),
                    "correction_energy_capture": (
                        1.0 if name == "full" else capture(energy, selections[name])
                    ),
                }
                items.append((name, out))
                print(
                    f"{name:7s} recovery {methods[name]['recovery']:+.3f}  "
                    f"capture {methods[name]['correction_energy_capture']:.3f}  "
                    f"PSNR {methods[name]['psnr_to_teacher_db']:.2f} dB  "
                    f"{dt:.3f}s"
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
                "methods": methods,
                "contact_sheet": str(contact),
            })

    aggregate = {}
    for name in ("edge", "random", "oracle", "full"):
        rows = [c["methods"][name] for c in cases]
        aggregate[name] = {
            "cases": len(rows),
            "mean_recovery": float(np.mean([r["recovery"] for r in rows])),
            "median_recovery": float(np.median([r["recovery"] for r in rows])),
            "mean_capture": float(np.mean([r["correction_energy_capture"] for r in rows])),
            "mean_psnr_to_teacher_db": float(np.mean([r["psnr_to_teacher_db"] for r in rows])),
            "min_psnr_to_teacher_db": float(np.min([r["psnr_to_teacher_db"] for r in rows])),
            "mean_stage_seconds": float(np.mean([r["stage_seconds"] for r in rows])),
        }

    edge = np.asarray([c["methods"]["edge"]["recovery"] for c in cases])
    rand = np.asarray([c["methods"]["random"]["recovery"] for c in cases])
    parity = np.asarray([c["methods"]["full"]["psnr_to_teacher_db"] for c in cases])

    receipt = {
        "full_mask_mean_psnr_db": float(parity.mean()),
        "full_mask_min_psnr_db": float(parity.min()),
        "edge_mean_recovery": float(edge.mean()),
        "random_mean_recovery": float(rand.mean()),
        "edge_minus_random_mean_recovery": float(edge.mean() - rand.mean()),
        "edge_recovery_case_wins": int(np.sum(edge > rand)),
        "required_full_mask_min_psnr_db": 35.0,
        "required_edge_mean_recovery": 0.20,
        "required_edge_margin": 0.08,
        "required_case_wins": 6,
    }
    receipt["pass"] = bool(
        receipt["full_mask_min_psnr_db"] >= 35.0
        and receipt["edge_mean_recovery"] >= 0.20
        and receipt["edge_minus_random_mean_recovery"] >= 0.08
        and receipt["edge_recovery_case_wins"] >= 6
    )

    payload = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "grid": GRID,
        "selected_tiles": SELECTED_TILES,
        "base_steps": args.base_steps,
        "masked_steps": args.masked_steps,
        "strength": args.strength,
        "note": "P2 tests global-state masked actuation only; masked UNet execution is still full-frame and no speedup is claimed.",
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
