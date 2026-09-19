# SDXL VAE engineering closeout

Status: **pre-registered engineering benchmark; ready for CUDA.**

The stage profiler resolved the remaining 512×512 bottleneck question.

Across 9 RTX-3060 cases:

| component | mean CUDA time | fraction of 763 ms wall |
|---|---:|---:|
| **VAE decode** | **277.5 ms** | **36.36%** |
| U-Net, 2 calls total | 238.4 ms | 31.24% |
| **VAE encode** | **167.9 ms** | **22.00%** |
| text encoder 2 | 24.7 ms | 3.23% |
| text encoder 1 | 11.2 ms | 1.47% |
| residual wall | 43.5 ms | 5.69% |

So:

```text
VAE encode + decode = 445.4 ms = 58.4% of the stage
U-Net total         = 238.4 ms = 31.2%
```

This explains why P3's U-Net routing could never rescue the whole route at this
scale. The majority of the stage was outside the operation being sparsified.

## The final 512px engineering benchmark

No new routing algorithm is tested.

We replace only the VAE with two established SDXL-compatible alternatives:

1. `madebyollin/sdxl-vae-fp16-fix`
   - full AutoencoderKL;
   - designed to run SDXL VAE inference in fp16 without the ordinary fp32
     upcast.

2. `madebyollin/taesdxl`
   - Diffusers `AutoencoderTiny`;
   - much smaller encode/decode network with a stronger fidelity tradeoff.

The ordinary SDXL-Turbo VAE defines the teacher output.

Prompt, base image, scheduler, stage seed, text encoders and U-Net are fixed.
Only the VAE changes.

## Selection protocol

Seeds **100/101** across all three prompts are validation.

A candidate must satisfy:

```text
mean correction recovery vs standard VAE    >= 85%
minimum 64x64 layout PSNR                   >= 28 dB
mean edge correlation                       >= 0.90
mean full-stage speedup                     >= 1.30x
faster than standard                        >= ceil(2N/3)
```

Among passing candidates, choose the fastest. Seed **102** is held out and must
pass the same frontier.

The script also profiles VAE encode, VAE decode and U-Net time under each
candidate, so any speedup is attributable rather than inferred.

## Interpretation

This is explicitly **not** a WhatToLookAt result. It is a systems baseline
forced by the profiler.

Whatever happens, the 512×512 Turbo branch closes after this benchmark:

- if fp16-fix passes, use it as the sane engineering default;
- if TAESDXL passes, it is the fast-preview/default candidate;
- if neither passes, the standard VAE remains the fidelity baseline.

The original WhatToLookAt mechanism should then move to workloads where omission
removes an entire expensive operation: tiled 2K–4K refinement, streaming/video,
or physical sensing.

## Run

```bash
git pull
python benchmark_sdxl_vae.py --local-only
```

If either VAE is not cached yet, omit `--local-only` for the first run.

Upload:

```text
results/sdxl_vae_closeout/
```
