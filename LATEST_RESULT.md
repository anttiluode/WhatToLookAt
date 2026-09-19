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
