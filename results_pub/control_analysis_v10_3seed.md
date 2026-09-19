# Control analysis, seeds 42, 123, 777 — replication of the seed-42 analysis and a target-clustered pool

Results directory `results_pub`, tau = 5 %.  All conditions present.

Post hoc extension of the seed-42 controls (Section 2.7, amendment 2); each seed is analysed exactly as seed 42 was; the pooled block resamples targets (a target's seeds together) for its interval and marks pair-independent p-values nominal.

## m0 vs base

| seed | n | HL shift [95 %] | median | mean | neg/pos | Wilcoxon p | pretenders | discordant | McNemar p | forward MAE base → control (B/A/C) |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 60 | +0.217 [-0.092, +0.576] | +0.135 | +0.065 | 27/33 | 0.1756 | 6 → 4 | 4/2 | 0.688 | 1.72→1.89 / 1.76→2.59 / 2.11→1.96 |
| 123 | 60 | +0.169 [-0.148, +0.498] | +0.085 | +0.325 | 25/35 | 0.2825 | 4 → 6 | 3/5 | 0.727 | 1.58→1.54 / 1.78→2.62 / 2.10→1.95 |
| 777 | 59 | +0.425 [+0.147, +0.732] | +0.344 | +0.615 | 17/42 | 0.0025 | 6 → 7 | 4/5 | 1.000 | 1.68→1.74 / 1.67→2.57 / 2.04→2.01 |
| **pooled** | 179 (60 targets) | +0.272 [+0.102, +0.455] (nominal) | +0.199 **[+0.035, +0.381] (target-cluster bootstrap)** | +0.333 | 69/110 | 0.0018 (nominal) | 16 → 17 | 11/12 | 1.000 (nominal) | |

| structure (pooled over seeds) | n (targets) | median shift [target-cluster 95 %] | HL shift [nominal 95 %] | mean | neg/pos | Wilcoxon p (nominal) | pretenders |
|---|---|---|---|---|---|---|---|
| B | 60 (20) | +0.024 [-0.242, +0.612] | +0.000 [-0.527, +0.437] | -0.008 | 29/31 | 0.9941 | 9 → 5 |
| A | 59 (20) | +0.823 [+0.545, +1.318] | +0.919 [+0.558, +1.287] | +1.000 | 12/47 | 0.0000 | 2 → 7 |
| C | 60 (20) | +0.005 [-0.096, +0.141] | +0.022 [-0.109, +0.172] | +0.019 | 28/32 | 0.6910 | 5 → 5 |

## feas vs base

| seed | n | HL shift [95 %] | median | mean | neg/pos | Wilcoxon p | pretenders | discordant | McNemar p | forward MAE base → control (B/A/C) |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 60 | -0.181 [-0.585, -0.003] | -0.001 | -0.793 | 40/20 | 0.0071 | 6 → 3 | 4/1 | 0.375 | 1.72→1.72 / 1.76→1.76 / 2.11→2.11 |
| 123 | 60 | -0.021 [-0.258, +0.004] | -0.000 | -0.464 | 33/26 | 0.2976 | 4 → 1 | 3/0 | 0.250 | 1.58→1.58 / 1.78→1.78 / 2.10→2.10 |
| 777 | 59 | -0.086 [-0.506, -0.001] | -0.001 | -0.550 | 36/23 | 0.0193 | 6 → 1 | 5/0 | 0.062 | 1.68→1.68 / 1.67→1.67 / 2.04→2.04 |
| **pooled** | 179 (60 targets) | -0.096 [-0.260, -0.008] (nominal) | -0.001 **[-0.025, -0.000] (target-cluster bootstrap)** | -0.602 | 109/69 | 0.0004 (nominal) | 16 → 5 | 12/1 | 0.003 (nominal) | |

| structure (pooled over seeds) | n (targets) | median shift [target-cluster 95 %] | HL shift [nominal 95 %] | mean | neg/pos | Wilcoxon p (nominal) | pretenders |
|---|---|---|---|---|---|---|---|
| B | 60 (20) | -0.244 [-0.889, -0.013] | -0.642 [-1.588, -0.156] | -1.308 | 40/20 | 0.0011 | 9 → 1 |
| A | 59 (20) | -0.035 [-0.498, +0.006] | -0.248 [-0.627, -0.003] | -0.495 | 35/24 | 0.0439 | 2 → 0 |
| C | 60 (20) | -0.000 [-0.000, +0.000] | -0.000 [-0.001, +0.000] | -0.003 | 34/25 | 0.6561 | 5 → 4 |
