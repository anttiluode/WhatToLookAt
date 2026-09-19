# Practical router P3 — sparse UNet execution inside the global trajectory

Status: **pre-registered; ready for CUDA.**

P2 passed every predeclared criterion:

```text
full-mask parity                120 dB minimum PSNR
edge mean recovery              30.87%
random mean recovery            17.30%
edge - random                   +13.57 percentage points
edge case wins                  9 / 9
```

The oracle 4/16 mask recovers 40.33%, so the cheap edge selector achieves about
three quarters of the available oracle recovery at this spatial budget.

That means we finally have a semantically valid actuator. P2's only remaining
problem is computational: it still evaluates the **full 64×64 latent UNet** and
therefore costs about the same as full-frame refinement.

P3 makes the expensive operation sparse.

## Sparse executor

At every denoising timestep, the global latent/noise/timestep state remains
exactly the P2 state.

For each selected 4×4 image-grid tile, P3 extracts the corresponding **16×16
latent tile** plus a halo, batches the four crops through the same SDXL UNet, and
writes only the center-tile noise prediction back into the global 64×64 noise
field.

Candidate latent halos:

| halo | crop per selected tile | naive total spatial work |
|---:|---:|---:|
| 0 | 16×16 | 25.0% of full |
| 4 | 24×24 | 56.25% |
| 8 | 32×32 | 100% |

The percentages are transparent spatial-element ratios, not FLOP claims. Actual
CUDA wall time decides whether anything was saved.

This is not P0 again. P0 started four independent image-space diffusion
trajectories. P3 crops the **current global latent at the current global
timestep**, predicts a local noise update, and returns that update to the same
global scheduler state.

## Selection protocol

Seeds **100 and 101** across all three prompts form six validation cases.

For every halo we run:

- edge-selected 4/16 tiles;
- same-compute random 4/16 tiles;
- the ordinary full-frame teacher stage;
- the P2 full-UNet edge-mask route as the semantic reference.

The smallest halo is selected only if validation satisfies all of:

```text
edge mean recovery                       >= 20%
edge - random recovery margin            >= 8 percentage points
edge beats random                        >= ceil(2N/3) cases
recovery retained vs P2 full-edge        >= 70%
mean stage speedup vs ordinary teacher   >= 1.10x
faster than teacher                      >= ceil(2N/3) cases
```

Seed **102** is untouched held-out evaluation and must satisfy the same rule.

## What a pass would mean

A P3 pass would be the first actual compute-saving mechanism in this entire
Sihti → WhatToLookAt branch:

```text
cheap evidence identifies where future compute matters
        +
one globally coherent diffusion state
        +
UNet work is physically restricted to those regions
        =
measured wall-clock saving with retained useful correction
```

It would still be a prototype, not a production diffusion accelerator. The crop
UNet loses long-range spatial context beyond its halo, and batched small kernels
may have poor GPU utilization.

If no halo passes, P2 remains scientifically useful but hard spatial sparsity is
not computationally viable on this architecture/GPU. The next target should be
feature/token caching or layer-wise early exit rather than more tiling.

## Run

```bash
git pull
python gpu_diffusion_router_p3.py --local-only
```

If necessary, omit `--local-only`.

Upload:

```text
results/practical_diffusion_router_p3/
```
