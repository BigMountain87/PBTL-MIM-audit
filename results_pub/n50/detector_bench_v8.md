# Detector benchmark — pooled AUROC [95 % CI] per structure (pretender vs confirmed)

Pooled AUROC = per-run AUROC averaged over the three seeds with weights n_pretender x n_confirmed. CI = 2000-draw percentile bootstrap over target clusters (each target's three seeds resampled together). 36 intervals are screened here; one excluding 0.5 is a selection, not a pointwise confirmation.

| detector | A | B | C |
|---|---|---|---|
| ens_std | 0.68 [0.30, 0.92] | 0.64 [0.51, 0.76] | 0.91 [0.82, 0.97] |
| heldout_mae | 0.55 [0.27, 0.84] | 0.65 [0.50, 0.78] | 0.97 [0.91, 1.00] |
| ensmean_mae | 0.42 [0.17, 0.73] | 0.64 [0.50, 0.78] | 0.97 [0.90, 1.00] |
| knn_train | 0.85 [0.74, 0.96] | 0.71 [0.58, 0.82] | 0.52 [0.37, 0.70] |
| knn_pool | 0.68 [0.34, 0.94] | 0.67 [0.54, 0.78] | 0.62 [0.46, 0.77] |
| tmm_mae_vs_surr | 0.57 [0.32, 0.78] | 0.53 [0.39, 0.66] | 0.50 [0.26, 0.68] |
| tmm_mae_vs_target | 0.58 [0.33, 0.79] | 0.53 [0.38, 0.66] | 0.51 [0.25, 0.71] |
| t4_mean_pert | 0.59 [0.21, 0.91] | 0.59 [0.44, 0.75] | 0.63 [0.46, 0.80] |
| t2_flag | 0.61 [0.56, 0.67] | 0.57 [0.47, 0.66] | 0.51 [0.41, 0.64] |
| endpoint_spread | 0.44 [0.10, 0.79] | 0.46 [0.34, 0.59] | 0.58 [0.40, 0.82] |
| mae_surr | 0.42 [0.19, 0.73] | 0.50 [0.33, 0.66] | 0.96 [0.89, 1.00] |
| infeasible_flag | 0.66 [0.45, 0.81] | 0.71 [0.64, 0.78] | 0.43 [0.33, 0.62] |
