# Latest result — Gate 4

Gate 4 removes Gate 3's exact-within-relation assumption.

A relation now predicts a shared high-resolution base, while individual patches
carry local residuals. Most residuals are tiny; 0, 2, 4, 6, or 8 of the 16
patches receive large innovations.

The cheap guide is an 8-scalar linear projection of the current high-resolution
content and is charged to the sensing budget.

The policy uses one high-resolution anchor per relation, then asks whether each
other member's guide residual is small enough to trust the shared relation or
large enough to justify buying that patch's own expensive measurement.

Training / validation:

```text
quality target                        PSNR >= 40 dB
learned guide-residual threshold      0.0239927
training balanced accuracy            1.000
validation balanced accuracy          1.000
```

Held-out reference: 80 worlds per innovation level.

```text
large innovations             0        2        4        6        8

fixed one per relation
  success                   100%       0%       0%       0%       0%
  total sensing cost         29.17%    29.17%    29.17%    29.17%    29.17%

residual adaptive
  success                   100%     100%     100%     100%     100%
  total sensing cost         29.17%    41.67%    54.24%    66.74%    78.62%

equal-cost random extra
  success                   100%       0%       0%       1.25%     0%
  total sensing cost         29.17%    41.67%    54.24%    66.74%    78.62%

oracle residual
  success                   100%     100%     100%     100%     100%
  total sensing cost         29.17%    41.67%    54.24%    66.74%    78.54%

full scan
  success                   100%     100%     100%     100%     100%
  total sensing cost        100%     100%     100%     100%     100%
```

The equal-cost random attacker is decisive: extra measurements are useful only
when the cheap residual localizes where the relation stopped predicting well.

Current mechanism:

```text
learned relation
    = compression hypothesis

cheap current residual
    -> trust the hypothesis and omit sensing
    -> or falsify it locally and buy the missing observation
```

The next gate should attack the sensing operator itself: make each cheap guide
measurement sparse, sweep how many signal coordinates it touches, and measure
the trade-off between compute saved and additional samples required.
