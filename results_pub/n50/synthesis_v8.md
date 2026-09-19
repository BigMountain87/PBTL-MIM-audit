# v8 cross-structure synthesis

| Quantity | A (r=+0.72) | C (r=+0.34) | B (r=-0.07) |
|---|---|---|---|
| n_train | 350 | 350 | 350 |
| Forward test MAE (%) | 1.76 | 2.11 | 1.72 |
| Oracle success (tau=5%) | 47/49 | 44/50 | 44/50 |
| Surrogate-claimed pass | 49/49 | 50/50 | 50/50 |
| Pretenders | 2/49 | 6/50 | 6/50 |
| Delta-flagged (k=2) | 3/49 | 2/50 | 12/50 |
| Flag threshold (%) | 3.51 | 4.21 | 3.44 |
| Max Delta (%) | 9.90 | 7.54 | 12.11 |
| Max amplification | 43.62x | 5.18x | 35.45x |
| T1 / 20 | 35 | 3 | 38 |
| T2 / 20 | 40 | 32 | 38 |
| T3 / 20 | 10 | 26 | 27 |
| T4 / 20 (measured) | 0 | 0 | 0 |
| Distinct modes | 44 | 30 | 28 |
| rho(RCWA,Surr) PRIMARY | +0.41 | +0.87 | +0.28 |
|   p-value | 0.003 | 0.000 | 0.052 |
| rho(Delta,RCWA) secondary | +0.87 | +0.79 | +0.95 |
| rho(Delta,Surr) [coupled] | +0.01 | +0.43 | +0.02 |
| Restart endpoint spread (u) | 1.88 | 1.38 | 1.57 |

## Fisher exact p-values (pairwise)
- **oracle_success**: A_vs_C: p=0.269, C_vs_B: p=1.000, A_vs_B: p=0.269
- **flagged**: A_vs_C: p=0.678, C_vs_B: p=0.008, A_vs_B: p=0.023
- **pretender**: A_vs_C: p=0.269, C_vs_B: p=1.000, A_vs_B: p=0.269

## Pre-registered hypotheses (protocol §1)
- H1a_oracle_success_nonincreasing: values A,C,B = [47, 44, 44] -> MONOTONE
- H1b_flag_rate_nondecreasing: values A,C,B = [3, 2, 12] -> NOT monotone
- H1c_rho_rcwa_surr_nonincreasing: values A,C,B = [0.41010204081632645, 0.8717887154861945, 0.27615846338535416] -> NOT monotone