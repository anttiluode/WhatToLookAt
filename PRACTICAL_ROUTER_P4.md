# Practical router P4 — feature caching instead of spatially cutting the U-Net

Status: **pre-registered; ready for CUDA.**

P3 decisively rejects the hard spatial-execution route.

Validation seeds 100/101:

| latent halo | naive spatial work | edge recovery | edge-random | stage speedup |
|---:|---:|---:|---:|---:|
| 0 | 25% | -27.54% | -15.10 pp | 0.780× |
| 4 | 56.25% | -14.36% | +8.34 pp | 0.782× |
| 8 | 100% | -7.82% | +16.98 pp | 0.771× |

No halo was selected, so seed 102 is **not** consulted for model selection.

The halo-8 result is the important boundary. Four 32×32 latent crops contain as
many raw spatial elements as one 64×64 frame, yet the update is still wrong and
slower. This points to two coupled problems:

1. cutting the U-Net changes cross-region attention / normalization / feature
   interactions;
2. small batched crop kernels underutilize the RTX 3060.

So hard tiling stops here.

## Why P4 switches to caching

[DeepCache](https://arxiv.org/abs/2312.00858) is established prior work. It
accelerates diffusion U-Nets by reusing high-level features across adjacent
denoising steps while updating cheaper features. The frame and its global
interaction graph stay intact.

P4 does **not** claim this as our algorithm. It asks whether the known mechanism
works on our exact short SDXL-Turbo img2img continuation, current software stack,
and RTX 3060.

That matters because Turbo has very few denoising steps; a caching method that is
excellent on 20–50 step samplers may have too little temporal redundancy here.

## Candidates

All candidates use:

```text
cache_interval = 2
```

and test DeepCache branch IDs:

```text
0, 1, 2
```

The baseline is the ordinary full-frame P2/P3 teacher continuation.

## Selection protocol

Seeds **100/101** across three prompts are validation.

A candidate must satisfy:

```text
mean correction recovery vs ordinary teacher    >= 80%
minimum PSNR to ordinary teacher                 >= 28 dB
mean stage speedup                               >= 1.10x
faster than teacher                              >= ceil(2N/3)
```

Among passing candidates, choose the fastest. Seed **102** is then held out and
must satisfy the same rule.

## What a pass means

A P4 pass does **not** make DeepCache ours. It establishes a practical substrate:

```text
full global U-Net state
+ selective reuse across time/depth
= quality-preserving measured speedup on this route
```

Only after that does the WhatToLookAt-specific question return:

> can a cheap observable predict when the cache is safe, and dynamically choose
> how aggressively to refresh it?

That would be P5.

If P4 fails, the short SDXL-Turbo trajectory simply does not have enough useful
temporal redundancy for this caching configuration, and we should stop trying to
force a generic diffusion-acceleration story onto Turbo.

## Run

```bash
git pull
pip install DeepCache
python gpu_diffusion_router_p4_cache.py --local-only
```

If necessary, omit `--local-only`.

Upload:

```text
results/practical_diffusion_router_p4/
```
