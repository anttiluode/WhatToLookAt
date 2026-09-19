# H0 design corrected — hard tile paste invalidated

The first 2048px H0 contact sheet exposed a design error before the result was
scored: the **all-16 tile teacher itself was visibly blocky**. Independent
512px refinements had been pasted edge-to-edge, so selective outputs were being
judged against a broken high-resolution reference.

That H0 run is invalid as visual-quality evidence.

H0b replaces it with:

```text
5×5 overlapping 512px tiles
128px overlap
384px stride
cosine partition-of-unity feathering
smooth all-25 teacher
smooth base + selected feathered corrections
```

Skipping a tile still deletes one complete diffusion call. The change is only
that the image assembly is now coherent.

A teacher-seam validity control is added, and the claim-eligible budgets become
6/25 and 12/25. Seed 102 remains held out.

See `HIGHRES_TILE_GATE.md`.

---

# 512×512 branch closed — fp16-fix VAE passes

The final engineering benchmark selects
`madebyollin/sdxl-vae-fp16-fix`.

Validation seeds 100/101:

```text
recovery                 99.50%
minimum layout PSNR      46.69 dB
edge correlation          0.959
stage time                0.540 s
speedup                   1.382x
faster cases              6 / 6
```

Held-out seed 102 independently passes:

```text
recovery                 99.74%
minimum layout PSNR      49.22 dB
edge correlation          0.965
stage time                0.540 s
speedup                   1.356x
faster cases              3 / 3
```

TAESDXL is much faster (~2.34x on validation) but does not preserve the
reference output.

So 512×512 SDXL-Turbo speed work is finished. The useful engineering default
is the fp16-fix full VAE; no routing claim is attached to it.

The next actual WhatToLookAt gate is high-resolution whole-call omission:
2048×2048 refinement as sixteen independent 512px diffusion tiles, with edge
energy deciding which full calls can be skipped.

See `HIGHRES_TILE_GATE.md`.

---

# 512×512 profile result — VAE bottleneck confirmed

The ordinary SDXL-Turbo four-step continuation on the RTX 3060 averages
**763.11 ms**.

```text
VAE decode       277.48 ms   36.36%
VAE encode       167.91 ms   22.00%
U-Net total      238.36 ms   31.24%
text encoders     35.91 ms    4.70%
other wall        43.46 ms    5.69%
```

VAE encode+decode therefore consumes **445.39 ms / 58.36%** of the stage.

That closes the mystery behind P3: the routed U-Net was not the majority of the
wall clock. P4 showed ~0.10–0.11 s of real reusable U-Net work, but not at the
required quality.

One final engineering baseline is now prepared in `VAE_CLOSEOUT.md`: compare
the standard VAE against the fp16-friendly full SDXL VAE and TAESDXL. No new
routing mechanism is allowed at 512×512 after that result.

---

# Practical diffusion branch — P4 result

P4 tests established DeepCache on the exact short SDXL-Turbo continuation and
fails the pre-registered quality/speed frontier.

Validation seeds 100/101:

```text
branch 0: recovery 38.22%, min PSNR 23.13 dB, speedup 1.152x
branch 1: recovery 64.12%, min PSNR 24.97 dB, speedup 1.166x
branch 2: recovery 72.20%, min PSNR 25.83 dB, speedup 1.158x
```

All three are genuinely faster, but none reaches the required 80% mean recovery
and 28 dB minimum PSNR. No candidate is selected; seed 102 is not used to rescue
the result.

This is a useful correction to the P3 timing story: the clock *can* see reused
U-Net work, but only about 0.10–0.11 s is safely exposed by this tested cache
family, and the quality loss is too large on a four-step Turbo trajectory.

The 512² algorithmic speed branch stops here. One final profiler measures where
the ordinary stage time lives before the project moves to high-resolution tiles,
streaming/video, or physical sensing.

See `PROFILE_SDXL_STAGE.md`.

---

# Practical diffusion branch — P3 result

P3 kills hard spatial U-Net tiling on the RTX 3060.

Validation seeds 100/101:

```text
halo 0: edge recovery -27.54%, speedup 0.780x
halo 4: edge recovery -14.36%, speedup 0.782x
halo 8: edge recovery  -7.82%, speedup 0.771x
```

No halo was selected, so the held-out seed-102 panel is not used for model
selection. The halo-8 result is especially informative: nominal spatial work is
already 100% of the full frame, yet split execution remains wrong and slower.

The practical boundary is therefore:

```text
global masked authority works (P2)
hard spatial U-Net splitting does not (P3)
```

P4 now tests established DeepCache feature reuse on the same SDXL-Turbo route.
This is a reproduction/compatibility test of prior work, not a novelty claim.
If it passes, the next original question is whether cheap evidence can control
cache refresh aggressiveness.

See `PRACTICAL_ROUTER_P4.md`.

---

# Practical diffusion branch — latest result

P2 is the first **positive causal mechanism** in the real SDXL branch.

On 9 RTX-3060 cases:

```text
edge 4/16 mask      mean recovery 30.87%
random 4/16 mask    mean recovery 17.30%
oracle 4/16 mask    mean recovery 40.33%
full mask            exact teacher parity, 120 dB
edge wins            9 / 9
```

The important correction to P0 is architectural. Independent crop diffusion
failed because every crop became a separate trajectory. P2 keeps one global
latent/noise/timestep trajectory and only controls which regions are allowed to
continue changing. Under that actuator, the cheap edge cue works causally.

P2 still performs a full-frame UNet evaluation and therefore earns no speed
claim.

P3 is now the next falsifier: keep P2's global state, but run the UNet only on
the selected latent tiles plus a 0/4/8 latent-pixel halo. Seeds 100/101 select
the smallest halo meeting quality+routing+speed criteria; seed 102 is held out.

See `PRACTICAL_ROUTER_P3.md`.

---

# Latest result — Gate 8

Gate 8 turns Gate 7's "measurement diversity" requirement into an explicit
sparse sensing design.

Structured sensors are built from independent 64-wide partition layers over the
same 192 coordinates:

```text
degree 1   3 rows    coordinate touch cost  3072
degree 2   6 rows    coordinate touch cost  6144
degree 3   9 rows    coordinate touch cost  9216
degree 4  12 rows    coordinate touch cost 12288
```

Thresholds are trained on a mixture of innovation supports
`K = 1,2,4,8,16,32,64`. The smallest degree whose *worst* validation cell
reaches 95% is selected before held-out testing.

Result:

```text
structured degree     validation minimum     held-out minimum

1                       48.75%                 48.33%
2                       87.50%                 88.33%
3                       98.75%                 96.67%
4                      100.00%                100.00%
```

So degree 3 is the first design that earns the target.

At the exact same 9216 coordinate-touch budget:

```text
sensor                         validation min     held-out min

regular degree-3 overlap          98.75%             96.67%
random sparse, 9 rows             93.75%             95.00%
post-hoc pruned, 9 rows           91.25%             90.00%
dense, 3 rows                     93.75%             96.67%
```

The validation rule prevents us from retroactively crediting controls that only
look acceptable on the held-out sample.

The previous robust 12-row sparse ceiling remains:

```text
random sparse, 12 rows          validation 96.25%   held-out 96.67%
post-hoc pruned, 12 rows        validation 100%     held-out 99.17%
```

Gate 8 therefore buys the required multi-coordinate robustness with

```text
9216 coordinate touches
```

instead of

```text
12288 coordinate touches
```

for a **25% reduction in guide mixing work**.

The mechanism is now:

```text
coverage:
    every coordinate is observed somewhere

coded overlap:
    every coordinate participates in several different measurement contexts
    -> signed sparse combinations have more independent chances to remain visible
```

Degree 2 is not enough in this regime. Degree 3 is.

Next: hold degree 3, row width 64, and 9216 touches fixed, then vary only the
combinatorics of overlap. That will tell us whether regular degree alone earned
the result or whether low-coincidence / expander-like code geometry is doing the
real work.
