# Practical diffusion router P1 — separate selection from actuation

Status: **run in GitHub Actions on all 9 P0 cases; selection PASS, scalar actuation salvage FAIL.**

P0 failed much harder than "edge did not beat random":

- every selector had **negative** correction recovery;
- random was actually least bad;
- edge routing averaged **-2.02** recovery versus **-1.29** for random;
- the edge route took **1.400 s** versus **1.009 s** for the full 4-step teacher;
- edge was faster in **0/9** cases.

That immediately raises a different question:

> did the selector choose the wrong places, or did the local crop diffusion
> operation itself fail to approximate what full-frame diffusion would do?

P1 answers that without another GPU run.

## Measurements

For each P0 route, let

```text
T = teacher - cheap_base
D = routed  - cheap_base
```

The selected tiles are already recorded by P0.

P1 measures:

1. **correction-energy capture** — what fraction of ||T||² lies inside the
   selected 4/16 tiles;
2. **actuation cosine** — whether D points in the same pixel-space direction as
   T inside the selected tiles;
3. **oracle scalar blend** — best alpha in [0,1] for
   `cheap_base + alpha * D`;
4. **held-out scalar blend** — fit one alpha per selector on seeds 100/101,
   freeze it, and evaluate seed 102.

This distinguishes "right location, wrong action" from "wrong location."

## Pre-registered interpretation

Selection survives if edge-selected tiles capture at least **0.05** more teacher
correction energy than random on average.

Crop actuation is considered salvageable by simple strength calibration only if
the alpha learned on seeds 100/101 gives, on held-out seed 102:

- mean recovery >= **0.05**;
- mean recovery >= random + **0.03**;
- positive recovery in at least **2/3** held-out cases.

If selection passes but calibrated actuation fails, the next GPU mechanism
should stop doing independent crop img2img. The likely replacement is an
operation that preserves the global denoising state while masking/skipping
spatial compute, rather than generating each crop as a separate diffusion
trajectory.

## Run

CPU only:

```bash
python practical_router_p1_actuation.py
```

Output:

```text
results/practical_diffusion_router_p1_summary.json
```


## Result

The discriminator is clean.

| selector | correction-energy capture | actuation cosine | P0 recovery | best-per-case blend recovery |
|---|---:|---:|---:|---:|
| **edge energy** | **43.51%** | 0.257 | -2.022 | 3.53% |
| local variance | 40.35% | 0.269 | -1.782 | 3.76% |
| Gaussian residual | 41.65% | 0.239 | -2.087 | 3.26% |
| Sihti residue | 35.95% | 0.228 | -1.995 | 2.70% |
| random | 24.31% | 0.311 | -1.289 | 3.00% |

So the *where* signal survives strongly at the exact 4×4 geometry used by P0:

```text
edge capture - random capture = +19.21 percentage points
```

But the local crop action is badly scaled and only weakly aligned. The
least-squares edge blend fitted on seeds 100/101 is only:

```text
alpha = 0.1041
```

On held-out seed 102, applying that frozen alpha gives:

```text
edge recovery     3.98%
random recovery   2.70%
positive edge cases 3/3
```

That is a real positive direction, but it misses the pre-registered 5% recovery
and +3 point margin requirements. Even the per-case oracle scalar blend only
recovers 3.53% on average.

Therefore:

> **selection survives; independent crop diffusion is the wrong local
> actuator.**

P0 did not fail because edge energy points to irrelevant regions. It failed
because launching each selected crop as its own diffusion trajectory produces a
change far larger than, and only weakly aligned with, the corresponding
full-frame denoising correction.

The next mechanism must preserve the global diffusion state/noise/timestep and
make computation sparse *inside that trajectory*, rather than regenerate image
crops independently.
