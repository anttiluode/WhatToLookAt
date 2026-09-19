# Latest result — Gate 6

Gate 6 attacks a shortcut suggested by Gate 5.

Gate 5's sparse operator generated its own observations. Gate 6 asks whether we
can instead observe densely, delete most of the operator interactions afterward,
and decode with the sparsified operator.

One local relation residual is an 8-sparse vector in 192 dimensions. A fixed
96-row dense Gaussian operator is sparsified to row supports
`16,32,64,96,192`. OMP receives the correct sparsity `k=8`.

Reference: 240 untouched sparse signals.

```text
row support          16       32       64       96      192

designed sparse
  mean recall       80.89%    98.13%    99.69%    99.69%    99.84%
  exact support     16.67%    87.50%    97.92%    97.50%    98.75%

post-hoc sparse
  mean recall       11.77%    18.33%    36.46%    58.33%    99.84%
  exact support      0%        0%        0%        2.92%    98.75%

dense baseline
  mean recall       99.84%
  exact support     98.75%
```

At row support 64, the same sparse operator gives **97.92% exact support
recovery** when it generated the response, and **0%** when substituted after the
dense response was generated.

The receipt is therefore a boundary:

```text
designed sparse sensing
    !=
naive post-hoc sparsification of the measurement operator
```

The response carries the geometry of the operator that produced it. A global
density rescaling cannot fix the mismatch for OMP because positive response
scaling leaves its support-selection path unchanged.

This result is finite and decoder-specific. It does not claim a contradiction
with asymptotic post-sparsification theory.

The next gate should stop choosing measurement rows blindly. Give the system a
bank of legal sparse probes and let current relation uncertainty choose which
one to spend next, at a matched coordinate-touch budget.
