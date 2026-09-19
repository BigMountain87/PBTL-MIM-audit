# v8 cross-structure synthesis — convention `preliminary` (preliminary pilot median r at pre-registration (2026-06-10))

| Quantity | A (r=+0.72) | C (r=+0.34) | B (r=-0.07) |
|---|---|---|---|
| n_train | 350 | 300 | 350 |
| Forward test MAE (%) | 4.84 | 3.86 | 3.19 |
| Oracle success (tau=5%) | 11/20 | 7/20 | 11/20 |
| Surrogate-claimed pass | 20/20 | 20/20 | 20/20 |
| Pretenders | 9/20 | 13/20 | 9/20 |
| Delta-flagged (k=2) | 0/20 | 3/20 | 6/20 |
| Flag threshold (%) | 9.69 | 7.73 | 6.37 |
| Max Delta (%) | 7.38 | 13.67 | 31.03 |
| Max amplification | 54.70x | 27.45x | 51.73x |
| T1 / 20 | 19 | 11 | 16 |
| T2 / 20 | 15 | 11 | 14 |
| T3 / 20 | 0 | 7 | 5 |
| T4 / 20 | 0 | 0 | 0 |
| Distinct modes | 20 | 15 | 17 |
| rho(RCWA,Surr) PRIMARY | +0.18 | -0.04 | -0.03 |
|   p-value | 0.435 | 0.880 | 0.890 |
| rho(Delta,RCWA) secondary | +0.96 | +0.88 | +0.98 |
| rho(Delta,Surr) [coupled] | -0.02 | -0.35 | -0.17 |
| Restart endpoint spread (u) | 1.98 | 1.62 | 1.64 |

## Fisher exact p-values (pairwise)
- **oracle_success**: A_vs_C: p=0.341, C_vs_B: p=0.341, A_vs_B: p=1.000
- **flagged**: A_vs_C: p=0.231, C_vs_B: p=0.451, A_vs_B: p=0.020
- **pretender**: A_vs_C: p=0.341, C_vs_B: p=0.341, A_vs_B: p=1.000

## Pre-specified hypotheses under convention `preliminary` (order ACB)
- H1a_oracle_success_nonincreasing: values A,C,B = [11, 7, 11] -> NOT monotone
- H1b_flag_rate_nondecreasing: values A,C,B = [0, 3, 6] -> MONOTONE
- H1c_rho_rcwa_surr_nonincreasing: values A,C,B = [0.18496240601503755, -0.03609022556390977, -0.03308270676691729] -> NOT monotone

## Hypothesis statements
- H1a: oracle-success rate is non-increasing as r falls (A >= C >= B)
- H1b: Delta-flag rate is non-decreasing as r falls (A <= C <= B)
- H1c: rho(RCWA MAE, Surr MAE) is non-increasing as r falls (A >= C >= B)
- pooled_order: structures ranked by preliminary pilot median r at pre-registration (2026-06-10): A (+0.72) > C (+0.34) > B (-0.07)