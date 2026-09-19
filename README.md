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


## Gate 4 — a relation is a prediction, not an equality

Gate 3 still made every expensive high-resolution patch inside one relation
exactly equal. Gate 4 weakens that assumption.

Each four-patch object now has a shared high-resolution base plus a local
residual. Ordinary patches have tiny residual amplitude `0.003`; a controlled
number of patches receive large innovations of amplitude `0.18`.

The cheap 8-scalar guide is now a fixed linear projection of the **current**
high-resolution content, so its cost is explicit and its role is narrow: audit
whether a relation still predicts the expensive signal well enough.

For every relation:

1. choose one high-resolution anchor using only the guide measurements;
2. tentatively let the other members inherit that anchor;
3. compute each member's guide residual from the anchor;
4. if the residual is too large, buy that patch's own expensive high-resolution
   measurement.

The residual threshold is learned on 100 training worlds against an explicit
local target: would copying the relation anchor put this patch below 40 dB?
Validation uses 50 separate worlds.

```text
learned guide-residual threshold     0.0239927
training balanced accuracy           1.000
validation balanced accuracy         1.000
```

On 80 untouched worlds per innovation level:

| large-innovation patches | fixed one/relation success / cost | residual-adaptive success / cost | equal-cost random extra | oracle residual | full scan |
|---:|---:|---:|---:|---:|---:|
| 0 | 100% / 29.17% | **100% / 29.17%** | 100% / 29.17% | 100% / 29.17% | 100% / 100% |
| 2 | 0% / 29.17% | **100% / 41.67%** | 0% / 41.67% | 100% / 41.67% | 100% / 100% |
| 4 | 0% / 29.17% | **100% / 54.24%** | 0% / 54.24% | 100% / 54.24% | 100% / 100% |
| 6 | 0% / 29.17% | **100% / 66.74%** | 1.25% / 66.74% | 100% / 66.74% | 100% / 100% |
| 8 | 0% / 29.17% | **100% / 78.62%** | 0% / 78.62% | 100% / 78.54% | 100% / 100% |

All costs include the guide channel. The equal-cost random attacker buys exactly
the same number of extra high-resolution patches as the adaptive policy, but at
random non-anchor locations.

That means the new mechanism is stronger than “good relations let us sample
less”:

```text
relation = a compression hypothesis

cheap residual test
    -> relation still predictive? keep compressing
    -> relation locally broken? buy the missing detail
```

The adaptive cost curve is essentially the oracle residual curve in this
synthetic regime.

### Gate 4 scope fence

The innovations are sparse and deliberately strong, which makes the cheap
projection an unusually clean discriminator. The next attacker should make
mismatch **diffuse and weak** rather than sparse and obvious. That is where the
compressed-sensing paper's "price of sparsity" question becomes directly
relevant: how sparse may the *measurement operator itself* become before the
computational saving is eaten by additional samples?


## Gate 5 — the price of a sparse measurement operator

Gate 4's cheap guide still used dense random projections: every guide row mixed
all 192 high-resolution coordinates. Gate 5 makes the measurement matrix itself
sparse.

The attacked world is deliberately support-recovery-like. A local relation
failure changes only **8 of 192** high-resolution coordinates. A sparse guide
row touches only (d) coordinates. If none of its touched coordinates overlap
the innovation support, that row literally contains no evidence of the change.

For each row support

```text
d = 1, 2, 4, 8, 16, 32, 64, 96, 192
```

the gate sweeps guide-row counts

```text
m = 1, 2, 4, 8, 16, 32, 64
```

and learns a residual threshold on 100 training worlds. A first validation set
checks the detector. A second, completely separate validation panel runs the
**end-to-end adaptive reconstruction** and chooses the smallest (m) whose
worst innovation level reaches at least 95% success. The chosen pair is then
frozen for 80 untouched worlds per innovation level.

Two currencies are kept separate:

```text
guide scalar samples       = 16 m
guide coordinate touches   = 16 m d
```

The second is the rough mixing / multiply-add / physical fan-in cost.

The selected validation frontier:

| row support d | smallest passing m | guide scalar samples | coordinate touches | min validation success |
|---:|---:|---:|---:|---:|
| 1 | none through 64 | 1024 at m=64 | 1024 | 60.0% |
| 2 | none through 64 | 1024 at m=64 | 2048 | 92.5% |
| 4 | 32 | 512 | 2048 | 95.0% |
| 8 | 32 | 512 | 4096 | 100% |
| 16 | 8 | 128 | 2048 | 95.0% |
| 32 | 8 | 128 | 4096 | 100% |
| **64** | **4** | **64** | **4096** | **100%** |
| 96 | 4 | 64 | 6144 | 100% |
| 192 | 4 | 64 | 12288 | 97.5% |

So measurement sparsity really does have a price. At (d=1), even 64 guide
measurements per patch are insufficient for the end-to-end target because sparse
innovations are often simply missed. As (d) rises, far fewer rows are needed.

But the other side is just as important. The selected (d=64,m=4) guide uses
the **same 64 guide scalar samples** as the selected dense (d=192,m=4)
guide, while touching only 4096 signal coordinates across the scene instead of
12288 — a **3× reduction in guide mixing work**.

Held-out behavior for (d=64,m=4):

```text
innovation patches        0       2       4       6       8
success                 100%    100%    100%    98.75%  100%
total sensing cost      27.08%  39.58%  52.08%  64.51%  77.08%
```

The dense selected guide reaches similar scalar sensing cost, but pays three
times the coordinate-touch cost.

This is the first gate where the recent compressed-sensing paper connects
almost literally to the experiment:

```text
sparser measurement rows
    -> cheaper individual measurements
    -> greater risk of missing sparse signal support
    -> more rows required
```

The useful object is therefore not "sparse is better." It is the **cost-quality
frontier**.

### Gate 5 scope fence

The sparse innovations and random linear measurements are synthetic, so this is
not an empirical validation of an asymptotic compressed-sensing theorem. It is
an executable analogue of the same trade-off.

The next clean attacker is the paper's second setting: **designed sparse sensing
versus post-hoc sparsification**. Start with a dense guide that observed the
world, delete most of its measurement interactions afterward, and ask whether
the same relation-failure decision can be preserved. That separates "build a
cheap sparse sensor" from "compress an already dense sensor/computation."


## Gate 6 — designed sparse sensing is not post-hoc sparsification

Gate 5 made the physical guide operator sparse **before** observation. Gate 6
tests whether that can be replaced by a cheaper downstream trick: observe with
a dense operator, discard most of its interactions afterward, and decode as if
the sparse operator had generated the response.

One relation residual is an 8-sparse vector in `R^192`. A fixed 96-row dense
Gaussian operator `A` produces the noisy observation. For each support
`d = 16, 32, 64, 96, 192`, `A_s` keeps only `d` entries per row.

The same OMP decoder, with the true sparsity `k=8`, is evaluated in three
conditions:

```text
dense
    y = A x + z
    decode with A

designed sparse
    y_s = A_s x + z
    decode with A_s

post-hoc sparse
    y = A x + z
    discard operator interactions to A_s
    decode density-scaled y with A_s
```

Reference: 240 untouched sparse signals.

| row support d | designed sparse support recall | designed exact support | post-hoc recall | post-hoc exact support |
|---:|---:|---:|---:|---:|
| 16 | 80.89% | 16.67% | 11.77% | 0% |
| 32 | **98.13%** | **87.50%** | 18.33% | 0% |
| 64 | **99.69%** | **97.92%** | 36.46% | 0% |
| 96 | **99.69%** | **97.50%** | 58.33% | 2.92% |
| 192 | 99.84% | 98.75% | 99.84% | 98.75% |

At `d=64`, the sparse operator is perfectly useful when it actually produced
the observation: mean support recall is **99.69%** and exact support recovery is
**97.92%**. Substitute that same sparse operator *after* a dense observation and
exact recovery falls to **0%**.

So in this finite OMP regime, the sparse physical sensor and the sparsified
downstream computation are not interchangeable:

```text
the response remembers the operator that generated it
```

A scalar density correction cannot repair that mismatch. For OMP specifically,
multiplying the whole response by a positive scalar rescales all correlations
and residuals together and therefore does not change the support-selection
path.

This is a useful negative result, not a contradiction of a general
post-sparsification theorem: the signal is extremely sparse, finite-dimensional,
the decoder is OMP, and the gate does not use the maximum-likelihood estimator
or asymptotic regime of the theory that motivated the attack.

### Gate 6 scope fence

Gate 6 says that **naively** replacing the measurement operator after the
observation is dangerous in this regime. The next useful direction is not more
sparsification. It is to ask whether the system can choose *which* sparse rows
to use from its learned relation/uncertainty state, instead of drawing every
measurement pattern independently of what it already knows.

That would finally close the loop promised by the repo name:

```text
what I know
    -> what I choose to look at
    -> what I learn
    -> what I look at next
```

## Boundary inherited from SighImageFactorization

> **Structure may eliminate ambiguity; it may not manufacture an observable
> distinction that the measurements do not contain.**

A generative model that guesses a plausible missing image does not get credit
for observing it. Later gates should include measurement-equivalent worlds and
score calibration/uncertainty separately from image prettiness.

## Next gates

Gate 5 now measures the hardware/statistical trade-off directly. Extreme row
sparsity saves mixing work but needs many more measurements and can still miss
sparse innovations; moderate sparsity preserves the sensing budget while
cutting coordinate-touch cost.

Gate 6 now separates designed sparse sensing from naive post-hoc operator
sparsification and finds that they are not interchangeable in the finite sparse
OMP regime.

The next gate should become genuinely **active** rather than only sparse:

- maintain several legal sparse measurement rows;
- use the learned relation graph and current uncertainty to choose the next row;
- compare adaptive row selection with random sparse rows at the same total
  coordinate-touch budget;
- include a world where all candidate rows are equally informative so the
  active policy must fall back to no advantage.

That turns the repo from "how little can I sense?" into its literal question:
**what should I look at next?**

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
