# High-resolution Gate H0b — overlap-blended whole-tile omission at 2048px

Status: **pre-registered; ready for CUDA. H0 hard-paste result is invalidated.**

The first H0 implementation used sixteen non-overlapping 512×512 img2img tiles
and pasted them edge-to-edge. The contact sheet exposed the flaw immediately:
the **all-tile teacher itself had visible rectangular seams**. A selective image
was therefore being scored against a broken reference.

That H0 run is not evidence, even if its JSON metrics appear favorable.

H0b fixes the geometry before any result is scored.

## Smooth tiled teacher

Use:

```text
output        2048×2048
tile          512×512
overlap       128 px
stride        384 px
grid          5×5
tile calls    25
```

Five 512px tiles with 128px overlap cover one axis exactly:

```text
512 + 4×384 = 2048
```

Each tile has a cosine feather window. Neighboring windows are complementary in
their overlap and the full set forms a **partition of unity**.

The all-tile teacher is therefore a weighted blend of all 25 independently
refined tiles rather than a hard paste.

A selective output begins with the smooth Lanczos 2048px base and adds only the
selected weighted tile corrections. Missing tile weight belongs to the base.

So a transition between “refined” and “not refined” is gradual.

## Why whole-call omission is still real

Every teacher tile is still a complete independent 512px SDXL-Turbo img2img
call using the earned fp16-fix VAE.

Skipping one tile deletes exactly one:

```text
VAE encode → U-Net denoising → VAE decode
```

call.

Overlap makes the image coherent; it does not hide a global full-frame
diffusion pass.

## Budgets

| refined | fraction | skipped | claim eligible? |
|---:|---:|---:|---|
| 6/25 | 24% | 76% | yes |
| 12/25 | 48% | 52% | yes |
| 18/25 | 72% | 28% | diagnostic only |

Edge energy is computed on each overlapping 512px base crop.

Matched controls:

- **edge** — top-k cheap edge-energy tiles;
- **random** — 64 matched-budget subsets;
- **teacher-aware greedy oracle** — greedily maximizes exact recovery using the
  already-known teacher corrections; ceiling/reference only.

Because overlapping corrections interact, H0b no longer estimates recovery by
summing per-tile energies. It builds the exact Gram matrix of feathered tile
corrections. Random and oracle recovery are then computed from the exact
quadratic correction energy.

The edge-selected output is also assembled as an image and its recovery is
measured directly against the smooth teacher.

## Seam validity control

H0b explicitly measures gradient jumps at the four tile-stride boundaries on
each axis and compares each jump with nearby ordinary gradients.

The gate is invalid if the validation teacher has:

```text
max teacher seam ratio > 2.0
```

This is deliberately a validity condition, not a performance score. The first
H0 would have failed the visual spirit of this check immediately.

## Selection protocol

Seeds **100/101** across three photographic prompts are validation.

The smallest eligible budget must satisfy:

```text
mean correction recovery               >= 80%
minimum 512px-layout PSNR               >= 30 dB
mean edge correlation                   >= 0.90
edge - random-median recovery margin    >= 10 percentage points
edge beats random median                >= ceil(2N/3)
measured whole-call speedup             >= 1.75×
refined fraction                        <= 50%
max teacher seam ratio                  <= 2.0
```

Seed **102** is held out and must satisfy the same frontier.

## Kill rule

If neither 6/25 nor 12/25 passes, stop the high-resolution image branch. A
visually acceptable 18/25 result is not enough: saving only ~28% of tile calls
is below the practical target for this node.

If H0b passes, H1 repeats the selected budget on **real natural photographs**.

## Run

```bash
git pull
python highres_tile_gate.py --local-only
```

Output:

```text
results/highres_tile_h0b/
```

The contact sheets now include:

```text
smooth 2048 base
smooth overlap-blended all-25 teacher
edge 6/25
edge 12/25
edge 18/25
```

Inspect those images before trusting the JSON. That visual validity check is now
part of the experiment rather than an afterthought.
