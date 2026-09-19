# v8 cross-structure synthesis — convention `intermediate` (intermediate R1 revision a02aa8f (2026-06-19), superseded before publication)

| Quantity | A (r=+0.72) | B (r=+0.64) | C (r=+0.44) |
|---|---|---|---|
| n_train | 350 | 350 | 300 |
| Forward test MAE (%) | 4.84 | 3.19 | 3.86 |
| Oracle success (tau=5%) | 11/20 | 11/20 | 7/20 |
| Surrogate-claimed pass | 20/20 | 20/20 | 20/20 |
| Pretenders | 9/20 | 9/20 | 13/20 |
| Delta-flagged (k=2) | 0/20 | 6/20 | 3/20 |
| Flag threshold (%) | 9.69 | 6.37 | 7.73 |
| Max Delta (%) | 7.38 | 31.03 | 13.67 |
| Max amplification | 54.70x | 51.73x | 27.45x |
| T1 / 20 | 19 | 16 | 11 |
| T2 / 20 | 15 | 14 | 11 |
| T3 / 20 | 0 | 5 | 7 |
| T4 / 20 | 0 | 0 | 0 |
| Distinct modes | 20 | 17 | 15 |
| rho(RCWA,Surr) PRIMARY | +0.18 | -0.03 | -0.04 |
|   p-value | 0.435 | 0.890 | 0.880 |
| rho(Delta,RCWA) secondary | +0.96 | +0.98 | +0.88 |
| rho(Delta,Surr) [coupled] | -0.02 | -0.17 | -0.35 |
| Restart endpoint spread (u) | 1.98 | 1.64 | 1.62 |

## Fisher exact p-values (pairwise)
- **oracle_success**: A_vs_B: p=1.000, B_vs_C: p=0.341, A_vs_C: p=0.341
- **flagged**: A_vs_B: p=0.020, B_vs_C: p=0.451, A_vs_C: p=0.231
- **pretender**: A_vs_B: p=1.000, B_vs_C: p=0.341, A_vs_C: p=0.341

## Pre-specified hypotheses under convention `intermediate` (order ABC)
- H1a_oracle_success_nonincreasing: values A,B,C = [11, 11, 7] -> MONOTONE
- H1b_flag_rate_nondecreasing: values A,B,C = [0, 6, 3] -> NOT monotone
- H1c_rho_rcwa_surr_nonincreasing: values A,B,C = [0.18496240601503755, -0.03308270676691729, -0.03609022556390977] -> MONOTONE

## Hypothesis statements
- H1a: oracle-success rate is non-increasing as r falls (A >= B >= C)
- H1b: Delta-flag rate is non-decreasing as r falls (A <= B <= C)
- H1c: rho(RCWA MAE, Surr MAE) is non-increasing as r falls (A >= B >= C)
- pooled_order: structures ranked by intermediate R1 revision a02aa8f (2026-06-19), superseded before publication: A (+0.72) > B (+0.64) > C (+0.44)