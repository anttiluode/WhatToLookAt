# Latest result — Gate 0

Gate 0 establishes the oracle upper bound for the repository.

A 16×16 scene contains four constant-colour objects, each split into four
spatially disconnected patches. Random pixel measurements are reconstructed
under either the true relation partition, an ordinary four-neighbour lattice,
or a size-matched shuffled relation partition.

Reference run: 48 random masks for each measurement count.

```text
quality criterion: PSNR >= 40 dB

active true-relation sampling:
    4 / 256 pixels = 1.5625%
    PSNR            = 120 dB (numerically exact)

random true-relation sampling:
    first tested budget with >=90% success = 16 / 256 = 6.25%
    success at 16 measurements             = 100%

spatial lattice:
    median PSNR at 128 / 256 measurements  = 17.03 dB
    >=40 dB success at 128 measurements    = 0%

shuffled relation graph:
    median PSNR at 128 / 256 measurements  = 12.25 dB
    >=40 dB success at 128 measurements    = 0%
```

The receipt is deliberately narrow:

> A correct relation partition can replace measurements in a world that is
> simple on that partition, and destroying the partition destroys the gain.

This is not yet evidence that the system can *learn* such a graph without
looking at the future answer. Gate 1 must learn the relation from past temporal
evidence and then spend it on a later undersampled frame.
