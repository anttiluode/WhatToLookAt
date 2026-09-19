#!/usr/bin/env python3
"""Profile one ordinary SDXL-Turbo img2img stage by major component.

This is a closeout measurement, not another acceleration gate.

It times the exact 4-step continuation used by the practical P2-P4 branch and
answers a simple systems question:

    where is the wall time actually going?

CUDA events are inserted around:
- text encoder 1
- text encoder 2
- VAE encode
- UNet forward
- VAE decode

The full stage wall time is measured separately. "other wall" is the residual
after subtracting summed CUDA-event component time; it includes Python,
scheduler, preprocessing/postprocessing and synchronization overhead.

Run:
    python profile_sdxl_stage.py --local-only

Output:
    results/sdxl_stage_profile/summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

MODEL_ID = "stabilityai/sdxl-turbo"
PROMPTS = (
    "Oil painting of an ancient stone castle surrounded by stormy ocean waves",
    "A futuristic solar bicycle parked on a Finnish forest gravel road, morning mist",
    "A cyberpunk cat wearing reflective sunglasses in a neon alley, cinematic lighting",
)
COMPONENTS = (
    "text_encoder_1",
    "text_encoder_2",
    "vae_encode",
    "unet",
    "vae_decode",
)


def gen(seed: int):
    return torch.Generator(device="cuda").manual_seed(int(seed))


class EventAccumulator:
    def __init__(self):
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            name: [] for name in COMPONENTS
        }
        self.originals = []

    def _wrap_attr(self, obj, attr: str, name: str):
        original = getattr(obj, attr)
        self.originals.append((obj, attr, original))

        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = original(*args, **kwargs)
            end.record()
            self.events[name].append((start, end))
            return out

        setattr(obj, attr, wrapped)

    def install(self, pipe):
        self._wrap_attr(pipe.text_encoder, "forward", "text_encoder_1")
        self._wrap_attr(pipe.text_encoder_2, "forward", "text_encoder_2")
        self._wrap_attr(pipe.vae, "encode", "vae_encode")
        self._wrap_attr(pipe.unet, "forward", "unet")
        self._wrap_attr(pipe.vae, "decode", "vae_decode")

    def restore(self):
        for obj, attr, original in reversed(self.originals):
            setattr(obj, attr, original)
        self.originals.clear()

    def totals_ms(self):
        torch.cuda.synchronize()
        result = {}
        for name, pairs in self.events.items():
            values = [float(start.elapsed_time(end)) for start, end in pairs]
            result[name] = {
                "calls": len(values),
                "cuda_ms": float(sum(values)),
                "per_call_ms": [] if not values else [float(v) for v in values],
            }
        return result


def run_stage_profile(pipe, prompt, base, steps, strength, seed):
    acc = EventAccumulator()
    acc.install(pipe)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        image = pipe(
            prompt=prompt,
            image=base,
            num_inference_steps=int(steps),
            strength=float(strength),
            guidance_scale=0.0,
            generator=gen(seed),
        ).images[0]
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        acc.restore()

    components = acc.totals_ms()
    component_sum = float(sum(row["cuda_ms"] for row in components.values()))
    other = float(max(0.0, wall_ms - component_sum))

    return image, {
        "wall_ms": float(wall_ms),
        "components": components,
        "component_cuda_sum_ms": component_sum,
        "other_wall_ms": other,
    }


def aggregate(cases):
    names = list(COMPONENTS)
    walls = np.asarray([case["profile"]["wall_ms"] for case in cases], dtype=float)
    result = {
        "cases": len(cases),
        "mean_wall_ms": float(walls.mean()),
        "median_wall_ms": float(np.median(walls)),
    }

    comp = {}
    for name in names:
        values = np.asarray([
            case["profile"]["components"][name]["cuda_ms"]
            for case in cases
        ], dtype=float)
        calls = np.asarray([
            case["profile"]["components"][name]["calls"]
            for case in cases
        ], dtype=float)
        comp[name] = {
            "mean_cuda_ms": float(values.mean()),
            "median_cuda_ms": float(np.median(values)),
            "mean_calls": float(calls.mean()),
            "mean_fraction_of_wall": float(values.mean() / walls.mean()),
        }

    other = np.asarray([
        case["profile"]["other_wall_ms"] for case in cases
    ], dtype=float)
    comp["other_wall"] = {
        "mean_ms": float(other.mean()),
        "median_ms": float(np.median(other)),
        "mean_fraction_of_wall": float(other.mean() / walls.mean()),
    }
    result["components"] = comp

    ranked = sorted(
        (
            (name, row["mean_cuda_ms"])
            for name, row in comp.items()
            if name != "other_wall"
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    result["gpu_component_rank"] = [
        {"component": name, "mean_cuda_ms": float(value)}
        for name, value in ranked
    ]
    return result


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
        default=Path("results/sdxl_stage_profile"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This profiler requires CUDA.")

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
    torch.cuda.synchronize()

    cases = []

    for pidx, prompt in enumerate(prompts, start=1):
        for seed in args.seeds:
            stem = f"p{pidx:02d}_s{seed}"
            print("\n", stem, prompt)

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

            teacher.save(args.output_dir / f"{stem}_teacher.png")

            compact = {
                name: {
                    "calls": row["calls"],
                    "cuda_ms": round(row["cuda_ms"], 2),
                }
                for name, row in profile["components"].items()
            }
            print(
                f"wall {profile['wall_ms']:.2f} ms  "
                + "  ".join(
                    f"{name}={row['cuda_ms']:.2f}ms/{row['calls']}x"
                    for name, row in profile["components"].items()
                )
                + f"  other={profile['other_wall_ms']:.2f}ms"
            )

            cases.append({
                "prompt": prompt,
                "seed": int(seed),
                "profile": profile,
                "compact": compact,
            })

    payload = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "stage_steps": args.stage_steps,
        "strength": args.strength,
        "method": (
            "CUDA events around top-level text encoders, VAE encode/decode and "
            "UNet forward; full synchronized wall time measured separately."
        ),
        "caution": (
            "Component CUDA-event totals are a diagnostic breakdown, not an "
            "operator-level FLOP profile. Residual wall includes CPU/scheduler/"
            "pre/postprocessing overhead."
        ),
        "aggregate": aggregate(cases),
        "cases": cases,
    }

    out = args.output_dir / "summary.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nAggregate:")
    print(json.dumps(payload["aggregate"], indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
