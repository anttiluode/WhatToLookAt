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

The central score is not just reconstruction quality:

```text
B(Q) = minimum physical measurement cost needed to maintain quality Q.
```

A pretty reconstruction does not count as evidence that missing information was
observed. Each gate therefore includes an attacker that preserves the sensing
budget while destroying the structural relation.

## Gate 0 — can a correct relation graph buy measurements at all?

The upper-bound world is a 16×16 RGB image containing four piecewise-constant
objects. Each object is split into four **spatially disconnected** 4×4 patches,
so ordinary spatial smoothness is the wrong prior.

Three methods receive exactly the same random pixels:

1. oracle relation graph;
2. ordinary four-neighbour lattice;
3. a size-matched shuffled relation graph.

Reconstruction is deliberately simple graph-harmonic interpolation: no neural
network and no learned decoder.

Reference run: 48 random masks per measurement count.

| method | 4 measurements | 16 measurements | 128 measurements | first count with ≥90% of trials at PSNR ≥40 dB |
|---|---:|---:|---:|---:|
| true relation graph | 12.65 dB median | **120 dB** | **120 dB** | **16 / 256 = 6.25%** |
| spatial lattice | 8.52 dB | 9.82 dB | 17.03 dB | not reached by 50% sampling |
| shuffled relation | 7.29 dB | 8.57 dB | 12.25 dB | not reached by 50% sampling |

If the relation graph is trusted enough to choose the samples, one observation
per relation component reconstructs the synthetic world exactly:

```text
4 / 256 pixels = 1.5625%
```

That only proves an oracle upper bound.

## Gate 1 — history earns the graph

Gate 1 removes the oracle *object labels*.

There are 16 tracked patches, four per hidden object. During eight history
steps, the four parts of one object share the same noisy motion sequence. The
four object sequences deliberately contain the **same multiset of velocity
vectors in different temporal orders**. Mean velocity, speed distribution and
the unordered motion vocabulary therefore cannot identify the groups; temporal
common fate can.

A tiny deterministic k-means clusters the 16 ordered motion traces. The learned
relation is then carried to a **new random spatial arrangement** and used to
reconstruct an undersampled future frame.

Reference: 48 independent worlds.

```text
motion-history grouping:
    median ARI                     1.000

independently time-shuffled history:
    median ARI                     0.000
```

The sensing result:

| prior on the future frame | first tested budget with ≥90% success at PSNR ≥40 dB |
|---|---:|
| oracle relation | **12 / 256 = 4.6875%** |
| **history-learned common-fate relation** | **12 / 256 = 4.6875%** |
| time-shuffled-history relation | not reached by 50% sampling |
| absolute-coordinate memory | not reached by 50% sampling |
| spatial lattice | not reached by 50% sampling |

The active version again spends one observation per learned component:

```text
median measurements       4 / 256 = 1.5625%
success at PSNR >= 40 dB  100%
```

So Gate 1 upgrades the receipt from

```text
a correct graph can replace measurements
```

to

```text
past temporal structure
    -> learned persistent relation
    -> fewer measurements of a rearranged future
```

The coordinate attacker matters: remembering where the parts used to be does
not survive the rearrangement. The temporal-order attacker matters for the same
reason: merely having the same motion vocabulary is not enough; the shared
ordered history is what earns the relation.

### Gate 1 scaffold

Patch correspondence is still given across history and the future frame. In
other words, the system knows which tracked patch is which; it does **not** know
which patches belong to the same object. That is deliberate. The next attacker
should make correspondence uncertain rather than quietly crediting tracking to
the sensing mechanism.


## Gate 2 — uncertainty buys local measurements

Gate 1 still assumed that patch correspondence was trustworthy. Gate 2 attacks
that scaffold.

A synthetic tracker proposes current-patch → historical-patch matches. Zero,
one, two, or three cross-object swap pairs create 0, 2, 4, or 6 wrong
correspondences. The tracker also emits a noisy local confidence value.

A confidence threshold is selected on 80 separate training worlds and then
frozen:

```text
learned threshold              0.525
training balanced accuracy     0.99375
```

Five held-out policies compete:

- **always trust** — keep the relation graph at full authority and spend only
  four measurements;
- **global caution** — distrust the whole relation graph and measure all 16
  patches;
- **confidence-adaptive** — trusted matches share measurements through the
  relation graph; low-confidence patches become local singleton components and
  must pay for their own observation;
- **shuffled confidence** — identical confidence values and almost identical
  sensing cost, assigned to the wrong patches;
- **oracle uncertainty** — ceiling that knows which correspondences are wrong.

Reference: 80 untouched worlds per ambiguity level.

| wrong matches | always trust success / budget | confidence-adaptive success / budget | global caution | shuffled confidence | oracle |
|---:|---:|---:|---:|---:|---:|
| 0 | 100% / 4.00 | **100% / 4.00** | 100% / 16 | 100% / 4.00 | 100% / 4 |
| 2 | 0% / 4.00 | **96.25% / 5.99** | 100% / 16 | 1.25% / 5.99 | 100% / 6 |
| 4 | 0% / 4.00 | **95.0% / 7.95** | 100% / 16 | 0% / 7.95 | 100% / 8 |
| 6 | 0% / 4.00 | **95.0% / 9.95** | 100% / 16 | 1.25% / 9.91 | 100% / 10 |

So the new mechanism is not simply “be cautious”:

```text
local relation uncertainty
        -> withdraw local relation authority
        -> spend extra sensing only there
```

The shuffled-confidence attacker is the important receipt. Merely spending the
same number of extra measurements does not work when they are spent in the
wrong places.

### Gate 2 scope fence

The confidence channel is synthetic. Gate 2 establishes the *budgeting
mechanism*, not a real tracker. The next gate should derive correspondence
confidence from image evidence itself and preserve the same local-budget
advantage.


## Gate 3 — image-derived confidence, with sensing cost counted

Gate 2's confidence scalar was synthetic. Gate 3 replaces it with a cheap
**2×4 grayscale guide image** for each tracked patch and explicitly charges that
guide channel to the sensing budget.

Each patch has:

```text
cheap guide image       8 scalar samples
expensive content       8×8 RGB = 192 scalar samples
full 16-patch scan      3072 scalar samples
all 16 guide images      128 scalar samples
```

The high-resolution content is shared inside each hidden object relation. The
guide images are used only for identity matching. Ambiguous guide patches are
blended slightly past the midpoint toward the nearest cross-object look-alike,
so the nearest historical template often becomes genuinely wrong rather than
having a wrong ID injected by hand.

Three image-derived confidence diagnostics are calibrated on 80 training
worlds, compared on 40 separate validation worlds, and then frozen:

| cue | validation balanced accuracy |
|---|---:|
| **best-vs-second-best template margin** | **0.9759** |
| negative best-match error | 0.9631 |
| forward/backward cycle consistency | 0.9072 |

The validation-selected cue is the **template margin**, threshold
`0.42853`.

On 80 untouched worlds per ambiguity level:

| ambiguous guides | mean wrong matches | always trust success / total cost | image-confidence adaptive success / total cost | shuffled confidence | oracle uncertainty | full scan |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 100% / 29.17% | **100% / 29.71%** | 100% / 29.71% | 100% / 29.17% | 100% / 100% |
| 2 | 1.69 | 1.25% / 29.17% | **100% / 41.90%** | 6.25% / 41.90% | 100% / 39.71% | 100% / 100% |
| 4 | 3.35 | 0% / 29.17% | **96.25% / 55.18%** | 5.0% / 54.79% | 100% / 50.10% | 100% / 100% |
| 6 | 5.01 | 0% / 29.17% | **96.25% / 67.37%** | 0% / 66.74% | 100% / 60.49% | 100% / 100% |

Percentages are total scalar sensing cost relative to a full high-resolution
scan, including the guide channel.

This removes the most important hidden cheat in Gate 2: confidence now comes
from measured appearance evidence, and that evidence has a cost.

The matched-cost shuffled-confidence attacker survives the transition. It uses
the same margin values and nearly the same total sensing cost, but assigns
caution to the wrong locations and collapses.

So the current mechanism is:

```text
cheap image evidence
    -> local correspondence confidence
    -> relation authority
    -> expensive sensing only where confidence fails
```

or, in the project's emerging currency:

```text
certainty earns compression;
uncertainty buys observation.
```

### Gate 3 scope fence

The guide/content split is still synthetic, and the relation class is exact:
all four expensive patches inside one hidden object share identical content.
Gate 4 should therefore attack **model mismatch inside a relation**. If related
patches are only approximately predictable from one another, the machine must
decide whether another expensive sample is worth buying from its observed
reconstruction residual rather than relying on exact equality.

## Boundary inherited from SighImageFactorization

> **Structure may eliminate ambiguity; it may not manufacture an observable
> distinction that the measurements do not contain.**

A generative model that guesses a plausible missing image does not get credit
for observing it. Later gates should include measurement-equivalent worlds and
score calibration/uncertainty separately from image prettiness.

## Next gates

Gate 3 has now derived confidence from a charged low-resolution image channel.
The next attacker is **relation model mismatch**:

- let related high-resolution patches share only a predictable component rather
  than being exactly equal;
- make one representative measurement leave a measurable residual on the
  unobserved relation members;
- ask whether residual/uncertainty can buy a second expensive measurement only
  when it reduces error enough to justify its cost;
- compare against fixed one-per-relation, full sensing, and equal-cost random
  extra measurements.

After that, the sparse-measurement / hardware-cost gate can vary how many
inputs each physical measurement is allowed to touch.

After that, the Gate-19/20 lesson from SighImageFactorization can enter directly:
when the meaning of a predictor changes, sensing budget should rise, stale
authority should retreat, and the system should relearn before compressing again.

## Run

```bash
python -m pip install -r requirements.txt

python what_to_look_at.py
python gate1_history_graph.py
python -m pytest -q
```

Receipts are written to:

```text
results/gate0_summary.json
results/gate1_summary.json
```

The live browser instrument is served by GitHub Pages from `index.html`.

## Lineage

- **SighImageSuper / Sihti** — asks what survives an operator and what falls into
  residues.
- **SighImageFactorization** — history writes grouping relations; confidence and
  predictive evidence decide how much causal authority a relation receives.
- **WhatToLookAt** — asks whether stored relational state can now buy a smaller
  future sensing budget.

The compressed-sensing connection is therefore not “use ℓ1 because MRI did.”
It is the stricter engineering question:

> **How many physical observations did learned structure actually save?**
