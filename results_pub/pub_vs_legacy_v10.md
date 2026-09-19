# Two-arm contrast — audited release vs published pipeline

Runs compared (finished in both arms): A_123, A_42, A_777, B_123, B_42, B_777, C_123, C_42, C_777.  tau = 5 %.

| run | forward test MAE | surrogate-claimed | oracle-confirmed | pretenders | median Δ | max Δ |
|---|---|---|---|---|---|---|
| A_123 (legacy) | 4.84 % | 20/20 | 11/20 | **9** | +4.19 pp | +10.02 pp |
| A_123 (pub) | 1.78 % | 20/20 | 20/20 | **0** | +1.51 pp | +4.11 pp |
| A_42 (legacy) | 4.84 % | 20/20 | 11/20 | **9** | +3.95 pp | +7.38 pp |
| A_42 (pub) | 1.76 % | 20/20 | 19/20 | **1** | +1.45 pp | +9.90 pp |
| A_777 (legacy) | 4.73 % | 20/20 | 11/20 | **9** | +4.24 pp | +10.52 pp |
| A_777 (pub) | 1.67 % | 19/19 | 18/19 | **1** | +1.18 pp | +5.10 pp |
| B_123 (legacy) | 3.26 % | 20/20 | 9/20 | **11** | +4.24 pp | +26.62 pp |
| B_123 (pub) | 1.58 % | 20/20 | 17/20 | **3** | +1.23 pp | +15.67 pp |
| B_42 (legacy) | 3.19 % | 20/20 | 11/20 | **9** | +3.19 pp | +31.03 pp |
| B_42 (pub) | 1.72 % | 20/20 | 17/20 | **3** | +2.34 pp | +12.11 pp |
| B_777 (legacy) | 3.28 % | 20/20 | 10/20 | **10** | +4.60 pp | +33.07 pp |
| B_777 (pub) | 1.68 % | 20/20 | 17/20 | **3** | +1.54 pp | +7.63 pp |
| C_123 (legacy) | 3.83 % | 20/20 | 8/20 | **12** | +4.32 pp | +10.26 pp |
| C_123 (pub) | 2.10 % | 20/20 | 19/20 | **1** | +0.46 pp | +3.11 pp |
| C_42 (legacy) | 3.86 % | 20/20 | 7/20 | **13** | +4.02 pp | +13.67 pp |
| C_42 (pub) | 2.11 % | 20/20 | 18/20 | **2** | +0.21 pp | +4.66 pp |
| C_777 (legacy) | 4.00 % | 19/19 | 6/19 | **13** | +3.32 pp | +9.46 pp |
| C_777 (pub) | 2.04 % | 20/20 | 18/20 | **2** | +0.29 pp | +2.41 pp |

## Pooled

| arm | pretenders / claimed | rate [Wilson 95 %] |
|---|---|---|
| legacy | 95/179 | 53.1 % [45.8, 60.2] |
| pub | 16/179 | 8.9 % [5.6, 14.0] |

Difference 44.1 pp, Newcombe 95 % CI [35.2, 52.1], Fisher exact p = 2.27e-20; target-cluster bootstrap 95 % CI [33.0, 54.8].

All intervals and tests on this page are nominal — nominal: the shared targets within a run are not accounted for here; the design effect of section 3.4 applies to the pooled counts. The per-structure Fisher tests below carry the same caveat.

## Per structure (all compared seeds pooled)

| structure | legacy | pub | Fisher p |
|---|---|---|---|
| B | 30/60 | 9/60 | 7.4e-05 |
| A | 27/60 | 2/59 | 5.8e-08 |
| C | 38/59 | 5/60 | 6.9e-11 |

## Caveat

the two arms differ in checkpoints, data pools and solver settings at once; this contrast is release-vs-printed-pipeline, not an ablation of any one factor. Targets differ between arms, so the runs are independent samples, not paired.
