# Latest result — Gate 2

Gate 2 turns relation confidence into a **physical sensing budget**.

Gate 1 had learned a useful relation graph from common-fate history, but still
assumed patch correspondence was trustworthy. Gate 2 injects 0, 2, 4, or 6
cross-object correspondence errors and gives the tracker a noisy local
confidence value.

A threshold learned on 80 training worlds is frozen at **0.525**, with training
balanced accuracy **0.99375**.

On 80 untouched worlds per ambiguity level:

```text
wrong correspondences       0        2        4        6

always trust
  success                 100%       0%       0%       0%
  measurements             4.00     4.00     4.00     4.00

confidence-adaptive
  success                 100%      96.25%   95.0%    95.0%
  measurements             4.00     5.99     7.95     9.95

global caution
  success                 100%     100%     100%     100%
  measurements            16.00    16.00    16.00    16.00

shuffled confidence
  success                 100%       1.25%    0%       1.25%
  measurements             4.00     5.99     7.95     9.91

oracle uncertainty
  success                 100%     100%     100%     100%
  measurements             4.00     6.00     8.00    10.00
```

The adaptive policy is therefore close to the oracle cost curve. It does not
slow or densify sensing everywhere. It breaks only low-confidence relations
into local singleton components, forcing those patches to pay for their own
observations.

The key receipt is the shuffled-confidence attacker: spending essentially the
same number of measurements at the wrong locations does not recover the image.

```text
uncertainty is not only a veto;
uncertainty purchases observation.
```

Scope fence: the tracker confidence is synthetic in Gate 2. Gate 3 should derive
that confidence from image evidence itself and test whether the same local
budget mechanism survives occlusion / look-alike correspondence failures.
