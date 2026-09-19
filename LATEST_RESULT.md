# Latest result — Gate 1

Gate 0 showed only an oracle upper bound: if the correct relation partition is
given, that structure can replace measurements.

Gate 1 now learns the relation from **past temporal common fate** before the
future frame is undersampled.

The synthetic history contains 16 tracked patches, four per hidden object.
Each hidden object supplies one ordered eight-step motion sequence. All four
object sequences use the same multiset of velocity vectors, merely in different
orders, so unordered motion statistics do not identify the relation.

Across 48 independent worlds:

```text
ordered common-fate history:
    median relation ARI                 1.000

independently time-shuffled history:
    median relation ARI                 0.000
```

The patches are then rearranged spatially before the future image is sampled.

```text
quality criterion: PSNR >= 40 dB

history-learned relation:
    first tested random budget with >=90% success
    12 / 256 pixels = 4.6875%

oracle relation:
    12 / 256 pixels = 4.6875%

time-shuffled-history relation:
    criterion not reached by 128 / 256 pixels

absolute-coordinate memory:
    criterion not reached by 128 / 256 pixels

spatial lattice:
    criterion not reached by 128 / 256 pixels

active learned relation:
    median budget = 4 / 256 pixels = 1.5625%
    success       = 100%
```

So the current receipt is:

```text
past temporal structure
    -> learned persistent relation
    -> fewer measurements of a rearranged future
```

The remaining scaffold is explicit: patch correspondence is provided across
history and the future frame. Gate 2 should attack that correspondence and make
sensing budget respond to relation confidence rather than assuming the track is
trustworthy.
