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
