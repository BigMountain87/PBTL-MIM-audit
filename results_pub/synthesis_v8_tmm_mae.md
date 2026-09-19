# v8 cross-structure synthesis — convention `tmm_mae` (published operating-band TMM MAE (negated so that larger = better transfer))

| Quantity | A (r=-7.90) | B (r=-8.90) | C (r=-16.90) |
|---|---|---|---|
| n_train | 350 | 350 | 350 |
| Forward test MAE (%) | 1.76 | 1.72 | 2.11 |
| Oracle success (tau=5%) | 19/20 | 17/20 | 18/20 |
| Surrogate-claimed pass | 20/20 | 20/20 | 20/20 |
| Pretenders | 1/20 | 3/20 | 2/20 |
| Delta-flagged (k=2) | 2/20 | 7/20 | 1/20 |
| Flag threshold (%) | 3.51 | 3.44 | 4.21 |
| Max Delta (%) | 9.90 | 12.11 | 4.66 |
| Max amplification | 43.62x | 35.45x | 3.25x |
| T1 / 20 | 16 | 17 | 1 |
| T2 / 20 | 14 | 15 | 9 |
| T3 / 20 | 0 | 7 | 5 |
| T4 / 20 | 0 | 0 | 0 |
| Distinct modes | 20 | 16 | 17 |
| rho(RCWA,Surr) PRIMARY | +0.34 | +0.12 | +0.75 |
|   p-value | 0.143 | 0.622 | 0.000 |
| rho(Delta,RCWA) secondary | +0.93 | +0.98 | +0.65 |
| rho(Delta,Surr) [coupled] | +0.06 | -0.04 | +0.10 |
| Restart endpoint spread (u) | 1.83 | 1.60 | 1.41 |

## Fisher exact p-values (pairwise)
- **oracle_success**: A_vs_B: p=0.605, B_vs_C: p=1.000, A_vs_C: p=1.000
- **flagged**: A_vs_B: p=0.127, B_vs_C: p=0.044, A_vs_C: p=1.000
- **pretender**: A_vs_B: p=0.605, B_vs_C: p=1.000, A_vs_C: p=1.000

## Pre-specified hypotheses under convention `tmm_mae` (order ABC)
- H1a_oracle_success_nonincreasing: values A,B,C = [19, 17, 18] -> NOT monotone
- H1b_flag_rate_nondecreasing: values A,B,C = [2, 7, 1] -> NOT monotone
- H1c_rho_rcwa_surr_nonincreasing: values A,B,C = [0.3398496240601503, 0.11729323308270674, 0.7503759398496239] -> NOT monotone

## Hypothesis statements
- H1a: oracle-success rate is non-increasing as r falls (A >= B >= C)
- H1b: Delta-flag rate is non-decreasing as r falls (A <= B <= C)
- H1c: rho(RCWA MAE, Surr MAE) is non-increasing as r falls (A >= B >= C)
- pooled_order: structures ranked by published operating-band TMM MAE (negated so that larger = better transfer): A (-7.90) > B (-8.90) > C (-16.90)