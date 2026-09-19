# High-resolution Gate H0 — skip whole diffusion tiles at 2048px

Status: **pre-registered; ready for CUDA.**

The 512×512 SDXL-Turbo branch is closed.

Its final engineering result is useful but conventional:
`madebyollin/sdxl-vae-fp16-fix` passes the fixed validation and held-out
quality/speed frontier, giving roughly **1.36–1.38×** full-stage speedup with
~**99.5–99.7%** recovery of the ordinary-VAE output. TAESDXL is much faster
(~2.34× on validation) but far outside the fidelity frontier.

That speedup belongs to the VAE, not WhatToLookAt.

H0 moves the original mechanism to a scale where *not looking* means deleting
an entire expensive operation.

## Workload

For each prompt/seed:

```text
512×512 SDXL-Turbo source
        ↓ Lanczos 4×
2048×2048 base
        ↓ split 4×4
16 independent 512×512 img2img refinement calls
        ↓
all-tile 2048×2048 teacher
```

The tile refinements use the now-earned fp16-fix VAE.

This is intentionally a standard independent-tile workload. Unlike P0/P3,
there is **no global full-frame U-Net state that a selective route must somehow
approximate**. The all-tile teacher itself is sixteen independent calls.

Therefore skipping one tile literally removes one full diffusion call.

## Selector

Before refinement, compute cheap edge energy in each of the sixteen 512px
regions of the 2048px base.

Test three budgets:

| refined | skipped | main-claim eligible? |
|---:|---:|---|
| 4/16 | 75% | yes |
| 8/16 | 50% | yes |
| 12/16 | 25% | diagnostic only |

For each budget:

- **edge**: top-k edge-energy tiles;
- **random**: 64 matched-budget random selections, summarized by median and
  10th/90th percentiles;
- **oracle**: top-k tiles ranked by actual all-tile correction energy, ceiling
  only.

Because teacher tiles are already computed independently, selective outputs are
assembled exactly by keeping teacher-refined selected tiles and leaving the
2048px base elsewhere.

## Timing

Every one of the sixteen tile calls is individually wall-timed on CUDA.

Selective-route time is:

```text
cheap selector CPU time
+ sum(actual measured diffusion-call times of selected tiles)
```

This is not a FLOP estimate. It is call-level measured accounting: an omitted
tile corresponds to an actual measured call that is absent from the selective
route.

## Selection protocol

Seeds **100/101** across three photographic prompts are validation.

The *smallest* eligible budget must satisfy:

```text
mean correction recovery               >= 80%
minimum 512px-layout PSNR               >= 30 dB
mean edge correlation                   >= 0.90
edge - random-median recovery margin    >= 10 percentage points
edge beats random median                >= ceil(2N/3)
measured tile-call speedup              >= 1.75×
refined fraction                        <= 50%
```

Seed **102** is held out and must satisfy the same frontier.

The three source prompts deliberately contain the kinds of structure where
tile omission might make economic sense—large sky/water regions, broad
architectural surfaces, and bokeh around a detailed subject.

That is also a limitation: H0 is a reproducible generated-photo gate, not yet a
natural-image claim.

## Kill rule

If neither 4/16 nor 8/16 passes, stop the high-resolution image branch. Refining
12/16 may be visually fine but skipping only 25% is below the practical target
for this node.

If H0 passes, H1 must replace the generated source panel with **real natural
photographs** and keep the same frozen budget/thresholds.

## Run

```bash
git pull
python highres_tile_gate.py --local-only
```

If the models are not cached, omit `--local-only` once.

Upload:

```text
results/highres_tile_h0/
```
