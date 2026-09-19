# Detector benchmark — pooled AUROC [95 % CI] per structure (pretender vs confirmed)

Pooled AUROC = per-run AUROC averaged over the three seeds with weights n_pretender x n_confirmed. CI = 2000-draw percentile bootstrap over target clusters (each target's three seeds resampled together). 36 intervals are screened here; one excluding 0.5 is a selection, not a pointwise confirmation.

| detector | A | B | C |
|---|---|---|---|
| ens_std | 0.73 [0.42, 1.00] | 0.80 [0.63, 0.94] | 0.95 [0.88, 1.00] |
| heldout_mae | 0.51 [0.26, 0.79] | 0.75 [0.53, 0.91] | 0.98 [0.89, 1.00] |
| ensmean_mae | 0.38 [0.16, 0.60] | 0.73 [0.52, 0.89] | 0.98 [0.84, 1.00] |
| knn_train | 0.73 [0.50, 0.94] | 0.80 [0.63, 0.94] | 0.42 [0.16, 0.61] |
| knn_pool | 0.73 [0.50, 1.00] | 0.79 [0.62, 0.92] | 0.49 [0.30, 0.78] |
| tmm_mae_vs_surr | 0.62 [0.39, 0.84] | 0.42 [0.23, 0.63] | 0.23 [0.00, 0.46] |
| tmm_mae_vs_target | 0.62 [0.38, 0.84] | 0.43 [0.24, 0.63] | 0.22 [0.00, 0.44] |
| t4_mean_pert | 0.92 [0.79, 1.00] | 0.48 [0.20, 0.77] | 0.66 [0.44, 0.87] |
| t2_flag | 0.66 [0.57, 0.77] | 0.62 [0.49, 0.75] | 0.46 [0.35, 0.60] |
| endpoint_spread | 0.27 [0.00, 0.63] | 0.50 [0.29, 0.68] | 0.74 [0.54, 0.92] |
| mae_surr | 0.27 [0.09, 0.49] | 0.40 [0.19, 0.67] | 0.98 [0.89, 1.00] |
| infeasible_flag | 0.74 [0.63, 0.87] | 0.76 [0.68, 0.86] | 0.37 [0.28, 0.46] |
