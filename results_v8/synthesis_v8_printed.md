# v8 cross-structure synthesis — convention `printed` (printed Paper 1 Table 5 median r (PNFA 72(B) 101617))

| Quantity | B (r=+0.96) | A (r=+0.83) | C (r=+0.65) |
|---|---|---|---|
| n_train | 350 | 350 | 300 |
| Forward test MAE (%) | 3.19 | 4.84 | 3.86 |
| Oracle success (tau=5%) | 11/20 | 11/20 | 7/20 |
| Surrogate-claimed pass | 20/20 | 20/20 | 20/20 |
| Pretenders | 9/20 | 9/20 | 13/20 |
| Delta-flagged (k=2) | 6/20 | 0/20 | 3/20 |
| Flag threshold (%) | 6.37 | 9.69 | 7.73 |
| Max Delta (%) | 31.03 | 7.38 | 13.67 |
| Max amplification | 51.73x | 54.70x | 27.45x |
| T1 / 20 | 16 | 19 | 11 |
| T2 / 20 | 14 | 15 | 11 |
| T3 / 20 | 5 | 0 | 7 |
| T4 / 20 | 0 | 0 | 0 |
| Distinct modes | 17 | 20 | 15 |
| rho(RCWA,Surr) PRIMARY | -0.03 | +0.18 | -0.04 |
|   p-value | 0.890 | 0.435 | 0.880 |
| rho(Delta,RCWA) secondary | +0.98 | +0.96 | +0.88 |
| rho(Delta,Surr) [coupled] | -0.17 | -0.02 | -0.35 |
| Restart endpoint spread (u) | 1.64 | 1.98 | 1.62 |

## Fisher exact p-values (pairwise)
- **oracle_success**: B_vs_A: p=1.000, A_vs_C: p=0.341, B_vs_C: p=0.341
- **flagged**: B_vs_A: p=0.020, A_vs_C: p=0.231, B_vs_C: p=0.451
- **pretender**: B_vs_A: p=1.000, A_vs_C: p=0.341, B_vs_C: p=0.341

## Pre-specified hypotheses under convention `printed` (order BAC)
- H1a_oracle_success_nonincreasing: values B,A,C = [11, 11, 7] -> MONOTONE
- H1b_flag_rate_nondecreasing: values B,A,C = [6, 0, 3] -> NOT monotone
- H1c_rho_rcwa_surr_nonincreasing: values B,A,C = [-0.03308270676691729, 0.18496240601503755, -0.03609022556390977] -> NOT monotone

## Hypothesis statements
- H1a: oracle-success rate is non-increasing as r falls (B >= A >= C)
- H1b: Delta-flag rate is non-decreasing as r falls (B <= A <= C)
- H1c: rho(RCWA MAE, Surr MAE) is non-increasing as r falls (B >= A >= C)
- pooled_order: structures ranked by printed Paper 1 Table 5 median r (PNFA 72(B) 101617): B (+0.96) > A (+0.83) > C (+0.65)