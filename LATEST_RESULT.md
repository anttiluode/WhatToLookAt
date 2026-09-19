# Latest result — Gate 5

Gate 5 makes the cheap sensing operator itself sparse.

A relation failure changes only **8 of 192** high-resolution coordinates. A
guide measurement touches only (d) coordinates, so very sparse rows can
literally miss the changed support.

For every row support (d), guide-row count (m) is swept over
`1,2,4,8,16,32,64`. Thresholds are trained on 100 worlds. A separate detector
validation set is followed by a second end-to-end policy validation panel.
The smallest (m) whose *worst* validation innovation level reaches 95%
reconstruction success is frozen before the 80-world-per-level test.

Selected frontier:

```text
row support d      selected m     guide samples     coordinate touches
1                  none <=64      1024 @ m=64       1024
2                  none <=64      1024 @ m=64       2048
4                  32             512               2048
8                  32             512               4096
16                  8             128               2048
32                  8             128               4096
64                  4              64               4096
96                  4              64               6144
192                 4              64              12288
```

The extreme sparse cases expose the price directly:

```text
d=1, m=64   best minimum validation success   60.0%
d=2, m=64   best minimum validation success   92.5%
```

They save per-row work, but too many sparse innovations fall outside the
measurement support.

A useful middle regime appears at **d=64, m=4**. It uses the same 64 guide
scalar measurements as the selected dense **d=192, m=4** guide, but only
4096 coordinate touches across the scene instead of 12288.

Held-out d=64,m=4:

```text
innovation patches        0       2       4       6       8
success                 100%    100%    100%    98.75%  100%
total sensing cost      27.08%  39.58%  52.08%  64.51%  77.08%
```

So the gate earns a real cost curve rather than a slogan:

```text
measurement sparsity
    saves fan-in / mixing work
    but increases the chance that sparse signal support is never touched
    and therefore increases required sample count.
```

The next gate should compare **designed sparse sensing** with **post-hoc
sparsification of a dense measurement operator** at matched downstream quality
and compute cost.
