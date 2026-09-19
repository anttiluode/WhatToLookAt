# WhatToLookAt

**Can learned structure buy fewer physical observations?**

This repo starts from a compressed-sensing question and pushes it into the
SighImageFactorization / Sihti lineage:

> If experience has already learned which parts of the world belong together,
> can that relation state reduce how much of the next world must be measured?

The long-term loop is:

```text
learn structure
    -> choose what to measure
    -> reconstruct from sparse observations
    -> measure surprise
    -> spend more sensing when the model becomes unreliable
    -> relearn / reduce sensing again
```

The project is deliberately gate-driven. Pretty reconstructions do not count
as evidence that missing information was observed. Each gate includes an
attacker that destroys the structural relation while preserving the raw
measurement budget.

## Gate 0 — does a relation graph buy measurements at all?

Before learning a sensing policy, Gate 0 asks the cheap upper-bound question.
The world is a 16×16 RGB image containing four piecewise-constant objects.
Each object is split into four **spatially disconnected** 4×4 patches arranged
as a Latin square, so ordinary spatial smoothness is the wrong prior.

Three priors receive exactly the same randomly selected pixels:

1. **oracle relation graph** — knows which disconnected pixels belong to the
   same object;
2. **spatial lattice** — knows only four-neighbour geometry;
3. **shuffled relation graph** — same four group sizes as the oracle graph, but
   group membership is permuted away from the true objects.

Reconstruction is intentionally simple graph-harmonic interpolation. There is
no neural network and no learned decoder to hide the receipt.

Reference run: 48 random masks per measurement count.

| method | 4 measurements | 16 measurements | 128 measurements | first count with ≥90% of trials at PSNR ≥40 dB |
|---|---:|---:|---:|---:|
| true relation graph | 12.65 dB median | **120 dB** | **120 dB** | **16 / 256 = 6.25%** |
| spatial lattice | 8.52 dB | 9.82 dB | 17.03 dB | not reached by 50% sampling |
| shuffled relation | 7.29 dB | 8.57 dB | 12.25 dB | not reached by 50% sampling |

The stronger active upper bound is even simpler. If the relation graph is
trusted, spend one observation on each of its four components. Four pixels out
of 256 (**1.5625%**) reconstruct the synthetic scene exactly.

That result is intentionally *not* the scientific claim we ultimately want.
The graph is oracle in Gate 0. The receipt is only:

```text
if a relation operator is correct,
then relation structure can replace measurements;
if the relation is shuffled, the advantage disappears.
```

That earns the next question instead of assuming it.

## What Gate 0 does **not** prove

- The graph is not learned from images yet.
- The objects are constant-colour synthetic components.
- The active sampler already knows the connected relation components.
- No camera, Fourier mask, single-pixel device, or noisy sensor is modelled yet.
- A strong generative prior is not allowed to hallucinate a missing answer and
  receive credit for observing it.

The key boundary inherited from SighImageFactorization remains:

> **structure may eliminate ambiguity; it may not manufacture an observable
> distinction that the measurements do not contain.**

## Next gate — history must earn the graph

Gate 1 should remove the oracle relation labels. Use earlier fully observed
frames to learn a persistent common-fate relation, then undersample a *future*
frame. The learned graph only gets credit if it reduces the future measurement
budget relative to:

- ordinary spatial interpolation;
- a graph learned from shuffled temporal order;
- a graph that only memorizes old coordinates;
- and a same-size relation graph with randomized membership.

The central score for this repository is not just PSNR:

```text
B(Q) = minimum physical measurement cost needed to maintain quality Q.
```

Eventually `B(Q)` should become dynamic: stable worlds cost fewer observations;
novelty, broken contingencies, or uncertainty should temporarily buy more
sensing.

## Run

```bash
python -m pip install -r requirements.txt
python what_to_look_at.py
python -m pytest -q
```

The experiment writes `results/gate0_summary.json`.

The live browser instrument is served by GitHub Pages from `index.html`.

## Lineage

- **SighImageSuper / Sihti** — persistence/residue machinery asks what structure
  survives an operator.
- **SighImageFactorization** — history writes grouping relations; confidence and
  predictive evidence decide how much causal authority a relation receives.
- **WhatToLookAt** — asks whether that stored relation state can now buy a
  smaller future sensing budget.

The compressed-sensing connection is therefore not “use ℓ1 because MRI did.”
It is the stricter engineering question: **how much observation did learned
structure actually save?**
