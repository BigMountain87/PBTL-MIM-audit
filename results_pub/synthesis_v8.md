# v8 cross-structure synthesis — convention `preliminary` (preliminary pilot median r at pre-specification (2026-06-10))

| Quantity | A (r=+0.72) | C (r=+0.34) | B (r=-0.07) |
|---|---|---|---|
| n_train | 350 | 350 | 350 |
| Forward test MAE (%) | 1.76 | 2.11 | 1.72 |
| Oracle success (tau=5%) | 19/20 | 18/20 | 17/20 |
| Surrogate-claimed pass | 20/20 | 20/20 | 20/20 |
| Pretenders | 1/20 | 2/20 | 3/20 |
| Delta-flagged (k=2) | 2/20 | 1/20 | 7/20 |
| Flag threshold (%) | 3.51 | 4.21 | 3.44 |
| Max Delta (%) | 9.90 | 4.66 | 12.11 |
| Max amplification | 43.62x | 3.25x | 35.45x |
| T1 / 20 | 16 | 1 | 17 |
| T2 / 20 | 14 | 9 | 15 |
| T3 / 20 | 0 | 5 | 7 |
| T4 / 20 | 0 | 0 | 0 |
| Distinct modes | 20 | 17 | 16 |
| rho(RCWA,Surr) PRIMARY | +0.34 | +0.75 | +0.12 |
|   p-value | 0.143 | 0.000 | 0.622 |
| rho(Delta,RCWA) secondary | +0.93 | +0.65 | +0.98 |
| rho(Delta,Surr) [coupled] | +0.06 | +0.10 | -0.04 |
| Restart endpoint spread (u) | 1.83 | 1.41 | 1.60 |

## Fisher exact p-values (pairwise)
- **oracle_success**: A_vs_C: p=1.000, C_vs_B: p=1.000, A_vs_B: p=0.605
- **flagged**: A_vs_C: p=1.000, C_vs_B: p=0.044, A_vs_B: p=0.127
- **pretender**: A_vs_C: p=1.000, C_vs_B: p=1.000, A_vs_B: p=0.605

## Pre-specified hypotheses under convention `preliminary` (order ACB)
- H1a_oracle_success_nonincreasing: values A,C,B = [19, 18, 17] -> MONOTONE
- H1b_flag_rate_nondecreasing: values A,C,B = [2, 1, 7] -> NOT monotone
- H1c_rho_rcwa_surr_nonincreasing: values A,C,B = [0.3398496240601503, 0.7503759398496239, 0.11729323308270674] -> NOT monotone

## Hypothesis statements
- H1a: oracle-success rate is non-increasing as r falls (A >= C >= B)
- H1b: Delta-flag rate is non-decreasing as r falls (A <= C <= B)
- H1c: rho(RCWA MAE, Surr MAE) is non-increasing as r falls (A >= C >= B)
- pooled_order: structures ranked by preliminary pilot median r at pre-specification (2026-06-10): A (+0.72) > C (+0.34) > B (-0.07)