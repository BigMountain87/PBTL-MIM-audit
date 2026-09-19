# v8 cross-structure synthesis — convention `printed` (published Table 5 median r)

| Quantity | B (r=+0.96) | A (r=+0.83) | C (r=+0.65) |
|---|---|---|---|
| n_train | 350 | 350 | 350 |
| Forward test MAE (%) | 1.72 | 1.76 | 2.11 |
| Oracle success (tau=5%) | 17/20 | 19/20 | 18/20 |
| Surrogate-claimed pass | 20/20 | 20/20 | 20/20 |
| Pretenders | 3/20 | 1/20 | 2/20 |
| Delta-flagged (k=2) | 7/20 | 2/20 | 1/20 |
| Flag threshold (%) | 3.44 | 3.51 | 4.21 |
| Max Delta (%) | 12.11 | 9.90 | 4.66 |
| Max amplification | 35.45x | 43.62x | 3.25x |
| T1 / 20 | 17 | 16 | 1 |
| T2 / 20 | 15 | 14 | 9 |
| T3 / 20 | 7 | 0 | 5 |
| T4 / 20 | 0 | 0 | 0 |
| Distinct modes | 16 | 20 | 17 |
| rho(RCWA,Surr) PRIMARY | +0.12 | +0.34 | +0.75 |
|   p-value | 0.622 | 0.143 | 0.000 |
| rho(Delta,RCWA) secondary | +0.98 | +0.93 | +0.65 |
| rho(Delta,Surr) [coupled] | -0.04 | +0.06 | +0.10 |
| Restart endpoint spread (u) | 1.60 | 1.83 | 1.41 |

## Fisher exact p-values (pairwise)
- **oracle_success**: B_vs_A: p=0.605, A_vs_C: p=1.000, B_vs_C: p=1.000
- **flagged**: B_vs_A: p=0.127, A_vs_C: p=1.000, B_vs_C: p=0.044
- **pretender**: B_vs_A: p=0.605, A_vs_C: p=1.000, B_vs_C: p=1.000

## Pre-specified hypotheses under convention `printed` (order BAC)
- H1a_oracle_success_nonincreasing: values B,A,C = [17, 19, 18] -> NOT monotone
- H1b_flag_rate_nondecreasing: values B,A,C = [7, 2, 1] -> NOT monotone
- H1c_rho_rcwa_surr_nonincreasing: values B,A,C = [0.11729323308270674, 0.3398496240601503, 0.7503759398496239] -> NOT monotone

## Hypothesis statements
- H1a: oracle-success rate is non-increasing as r falls (B >= A >= C)
- H1b: Delta-flag rate is non-decreasing as r falls (B <= A <= C)
- H1c: rho(RCWA MAE, Surr MAE) is non-increasing as r falls (B >= A >= C)
- pooled_order: structures ranked by published Table 5 median r: B (+0.96) > A (+0.83) > C (+0.65)