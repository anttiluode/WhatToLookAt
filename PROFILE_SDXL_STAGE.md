# SDXL-Turbo stage profile — closing the 512² speed question

Status: **measurement ready; no optimization claim.**

The P3 and P4 results now justify stopping the 512×512 acceleration search
unless a profile identifies a large, simple bottleneck.

## What P4 added

DeepCache *does* expose some removable U-Net work on this short Turbo route.

Validation seeds 100/101:

| cache branch | mean recovery | minimum PSNR | stage speedup | faster cases |
|---:|---:|---:|---:|---:|
| 0 | 38.22% | 23.13 dB | 1.152× | 6/6 |
| 1 | 64.12% | 24.97 dB | **1.166×** | 6/6 |
| 2 | **72.20%** | **25.83 dB** | 1.158× | 6/6 |

No candidate clears the pre-registered quality frontier:

```text
mean recovery >= 80%
minimum PSNR  >= 28 dB
```

Therefore no candidate is selected and seed 102 is not used to rescue the gate.

The useful quantitative conclusion is:

> feature reuse can remove about 0.10–0.11 s from a ~0.775 s stage here, but
> the tested 4-step Turbo trajectory does not contain enough *quality-preserving*
> temporal redundancy for DeepCache to pass.

This slightly corrects the stronger claim that "the clock never saw the U-Net":
the clock does see U-Net reuse, but the savings are modest and arrive with too
much output drift.

## Why profile now

One remaining systems hypothesis is that the official SDXL VAE dominates enough
of this very short route that a tiny/fp16 VAE would be the only obvious free
latency win.

That is plausible, but it has not yet been measured in this repository.

`profile_sdxl_stage.py` inserts CUDA events around:

- text encoder 1;
- text encoder 2;
- VAE encode;
- every U-Net forward;
- VAE decode.

It also records synchronized end-to-end stage wall time and the residual
scheduler/Python/pre/postprocessing time.

This is deliberately a profiler, not P5. The 512² compute-routing lineage is
closed unless the measurement exposes a large trivial bottleneck.

## Run

```bash
git pull
python profile_sdxl_stage.py --local-only
```

If necessary, omit `--local-only`.

Upload:

```text
results/sdxl_stage_profile/
```

## Decision after the profile

- If VAE encode+decode dominate the stage, benchmark the known SDXL-compatible
  tiny/fp16 VAE as an engineering baseline.
- If the U-Net remains dominant, do not spend more gates on 512² Turbo speed.
- Either way, the original WhatToLookAt idea should move to a regime where
  omission deletes a **whole expensive operation**: tiled high-resolution
  refinement, streaming/video updates, or physical sensing.
