# Latest result — Gate 3

Gate 3 removes Gate 2's synthetic confidence oracle.

Each tracked patch now has a cheap 2×4 grayscale guide image and an expensive
8×8 RGB content patch.

```text
guide per patch          8 scalar samples
expensive content      192 scalar samples
all guides             128 scalar samples
full high-res scan    3072 scalar samples
```

Ambiguous guide images are blended toward a real cross-object look-alike.
Nearest-template matching therefore produces genuine wrong correspondences.

Three image-derived confidence diagnostics are calibrated on 80 training worlds
and selected on 40 separate validation worlds:

```text
best-vs-second-best margin       0.97593 validation balanced accuracy
negative best-match error        0.96315
forward/backward cycle           0.90722

selected cue: margin
threshold:    0.4285305
```

On 80 untouched worlds per ambiguity level:

```text
ambiguous guide patches          0        2        4        6
mean wrong correspondences     0.00     1.69     3.35     5.01

always trust
  success                      100%      1.25%     0%       0%
  total sensing cost            29.17%   29.17%   29.17%   29.17%

image-confidence adaptive
  success                      100%    100%       96.25%   96.25%
  total sensing cost            29.71%   41.90%   55.18%   67.37%

shuffled confidence
  success                      100%      6.25%     5.0%     0%
  total sensing cost            29.71%   41.90%   54.79%   66.74%

oracle uncertainty
  success                      100%    100%      100%     100%
  total sensing cost            29.17%   39.71%   50.10%   60.49%

full high-resolution scan
  success                      100%    100%      100%     100%
  total sensing cost           100%    100%      100%     100%
```

All cost fractions include the guide channel. The matched-cost shuffled cue is
the key attacker: nearly the same number of scalar measurements at the wrong
locations does not rescue reconstruction.

The mechanism earned through Gate 3 is therefore:

```text
cheap image evidence
    -> local match confidence
    -> local relation authority
    -> expensive sensing only where confidence fails
```

The next boundary is exact relation content. Related patches currently share
identical expensive content. Gate 4 should make the relation only approximately
predictive and test whether observed residual error can decide when a second
measurement is worth its cost.
