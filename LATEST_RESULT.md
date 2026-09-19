# Latest result — Gate 7

Gate 7 freezes Gate 6's selected sensors and attacks the interpretation rather
than tuning another sensor.

Gate 6 suggested that **support coverage** explained why three designed sparse
rows beat post-hoc pruning in a one-coordinate innovation world.

So Gate 7 changes only innovation support:

```text
K = 1, 2, 4, 8, 16, 32, 64
```

Per-coordinate amplitude scales as `0.65/sqrt(K)`, keeping total innovation
energy approximately constant. The sensing geometries and thresholds stay
frozen from K=1.

Minimum held-out success across 0/2/4/6/8 innovation patches:

```text
support K              1       2       4       8      16      32      64

coverage-designed    100%    44.17%  63.33%  80.0%  90.83%  96.67%  93.33%
post-hoc pruned      100%   100%    100%    100%   100%    100%    100%
random sparse         94.17%  99.17% 100%    100%   100%    100%    100%
dense                  95.83%  90.0%  82.5%   85.83% 91.67%  89.17%  85.83%
```

The coverage-designed sensor still touches **192/192 signal coordinates**.
Its K=2 collapse therefore cannot be blamed on blind coordinates.

The new failure mode is cancellation. With three disjoint-ish 64-wide rows,
multiple signed changes can share a measurement and partially cancel. Twelve
overlapping sparse rows give the same changed coordinates multiple independent
opportunities to leave a signature.

So Gate 7 corrects Gate 6:

```text
coverage is enough to prevent a 1-sparse change from being invisible

coverage is not enough to make multi-coordinate changes distinguishable
```

The mechanism we actually need is **measurement diversity**: overlapping or
coded sparse measurements that make different sparse combinations leave
different signatures.

This is also the first point where the compressed-sensing lineage becomes
structurally exact rather than merely inspirational. The next gate should try
to buy that diversity deliberately, ideally with a degree-2/expander-like sparse
design that uses far fewer than the 12 rows required by the robust controls.
