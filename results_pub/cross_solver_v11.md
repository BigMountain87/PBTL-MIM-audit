# Cross-solver probe — pub geometries at fixed order 5 (seed 42)

paired on the identical committed geometries and the identical surrogate; only the Fourier truncation differs (adaptive 9/13/17 -> fixed 5). Single seed, 20 distinct targets per structure, so no repeated-target dependence; pooled tests are nominal across structures.

| S | n | pretenders adaptive → order 5 | flips (conf→pret / pret→conf) | McNemar p | median |ΔMAE| pp | median ΔMAE pp | max |ΔMAE| | Wilcoxon p | r | s/design order 5 vs adaptive |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 20 | 3 → 6 | 3 / 0 | 0.250 | 0.95 | +0.22 | 4.22 | 0.2305 | 0.74 | 16 / 156 |
| A | 20 | 1 → 4 | 3 / 0 | 0.250 | 0.85 | +0.67 | 4.44 | 0.0266 | 0.73 | 29 / 3526 |
| C | 20 | 2 → 3 | 1 / 0 | 1.000 | 1.08 | +0.80 | 6.05 | 0.0006 | 0.56 | 17 / 1592 |

Pooled: pretenders 6 → 13 of 60; flips 7 / 0, McNemar p = 0.0156; median |ΔMAE| 0.90 pp, median ΔMAE +0.61 pp, order 5 higher on 46 / lower on 14, Wilcoxon p = 1.37e-04.
