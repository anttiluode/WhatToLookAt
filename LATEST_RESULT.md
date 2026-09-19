# Latest result — Gate 6

Gate 6 separates **designed sparse sensing** from **post-hoc sparsification**.

The relation failure is intentionally extreme: each changed patch modifies only
**1 of 192** high-resolution coordinates. Sparse guide rows may touch only
**64 coordinates**.

Four sensing families sweep guide-row count
`m = 1,2,3,4,6,8,12,16`. Each candidate learns its own threshold on 150
training worlds. The smallest (m) whose *worst* validation innovation level
reaches 95% reconstruction success is frozen before 120-world-per-level testing.

Selected frontier:

```text
family                    m   coverage   guide scalars   coordinate touches
coverage-designed sparse  3   192/192    48              3072
post-hoc pruned           12  192/192    192            12288
random sparse             12  190/192    192            12288
dense                      2  192/192     32             6144
```

The designed sparse supports deliberately cover new coordinates before
repeating. Three 64-wide rows therefore cover the entire 192-coordinate signal
space.

Post-hoc pruning keeps the largest weights of independently generated dense
rows. At equal row count it leaves support holes, so it needs twelve rows before
its worst validation level reaches the same target.

Held-out success after freezing the selected sensors:

```text
innovation patches          0       2       4       6       8

coverage-designed sparse  100%    100%    100%    100%    100%
post-hoc pruned           100%    100%    100%    100%    100%
random sparse             100%     98.33%  97.5%   95.0%   98.33%
dense                     100%     95.83%  96.67%  93.33%  97.5%
```

The important comparison is not a slogan that sparse beats dense. Dense reaches
the validation target with fewer guide scalar samples (32 versus 48), but twice
the coordinate-touch work and slightly weaker held-out robustness.

The clean sparse-vs-sparse receipt is:

```text
coverage-designed sparse      3072 guide coordinate touches
post-hoc pruned sparse       12288 guide coordinate touches
held-out success             100% across every innovation level for both
```

So support layout itself has become part of the algorithm:

```text
build sparse first
    -> cover the possible support deliberately

build dense then prune
    -> retained large weights can leave blind coordinates
    -> extra rows are needed to close those holes
```

Scope fence: this is a one-coordinate innovation regime chosen specifically to
make support blindness visible. The next gate should freeze these selected
operators and sweep innovation support size. If the interpretation is correct,
the designed-sparse advantage should shrink as failures become more diffuse.
