# SDXL VAE engineering closeout

Status: **run on RTX 3060; fp16-fix PASS on validation and held-out. 512×512 branch closed.**

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

The benchmark now preflights **both candidate VAEs before generating any of the
nine standard-VAE references**. This avoids wasting a reference pass only to
discover a missing Hugging Face cache entry afterward.

First run, if either candidate has not been cached:

```bash
git pull
python benchmark_sdxl_vae.py
```

That downloads/caches:

```text
madebyollin/sdxl-vae-fp16-fix
madebyollin/taesdxl
```

Repeat runs may then be fully offline:

```bash
python benchmark_sdxl_vae.py --local-only
```

With `--local-only`, a missing candidate now fails immediately during
preflight instead of after the standard reference panel has already run.

Upload:

```text
results/sdxl_vae_closeout/
```


## Result — fp16-fix wins the engineering frontier

The result is decisive.

Validation seeds 100/101:

| VAE | mean recovery | min layout PSNR | mean edge corr. | stage | speedup | pass |
|---|---:|---:|---:|---:|---:|---|
| **sdxl-vae-fp16-fix** | **99.50%** | **46.69 dB** | **0.959** | **0.540 s** | **1.382×** | **yes** |
| TAESDXL | 20.67% | 21.51 dB | 0.488 | 0.319 s | 2.336× | no |
| standard SDXL VAE | reference | reference | reference | 0.746 s | 1.000× | — |

`sdxl-vae-fp16-fix` is therefore selected before looking at seed 102.

Held-out seed 102:

```text
mean recovery             99.74%
minimum recovery          99.64%
mean PSNR                 48.98 dB
minimum layout PSNR       49.22 dB
mean edge correlation      0.965
mean stage                 0.540 s
baseline stage             0.733 s
speedup                    1.356×
faster cases               3 / 3
PASS
```

The component profile also shows where the gain comes from:

```text
standard VAE total        ≈ 445 ms
fp16-fix VAE total        ≈ 243–245 ms
UNet time                 ≈ unchanged (~235–237 ms)
```

TAESDXL demonstrates the opposite corner of the frontier. Its VAE itself costs
only ~25 ms and the whole stage is ~2.34× faster, but the output is not a
drop-in approximation to the standard-VAE teacher under this test.

So the final 512×512 engineering conclusion is:

> **Use the fp16-fix full SDXL VAE as the sane low-latency default on this RTX
> 3060 route. It earns a ~1.36× stage speedup while preserving essentially all
> of the reference continuation.**

This is not a WhatToLookAt mechanism. It is a conventional systems improvement
identified by profiling.

The 512×512 speed branch is now closed. The actual WhatToLookAt question moves
to [HIGHRES_TILE_GATE.md](HIGHRES_TILE_GATE.md), where skipping a region removes
an entire 512px diffusion invocation from a 2048px tiled-refinement workload.
