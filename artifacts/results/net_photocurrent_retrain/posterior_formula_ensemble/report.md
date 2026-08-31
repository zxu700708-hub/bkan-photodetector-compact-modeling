# Posterior-Sampled Formula Ensemble Audit

- Fixed VI posterior draws per task: 20
- BKAN architecture/training: unchanged; checkpoints loaded read-only
- Formula optimizer seed: fixed across posterior draws
- Calibration groups are not used for formula fitting or selection

## Stability

| task | pair_count | term_jaccard_mean | term_jaccard_min | prediction_disagreement_rmse_mean | prediction_disagreement_rmse_max |
| --- | --- | --- | --- | --- | --- |
| I_photo | 190 | 0.8946954255390189 | 0.8227848101265823 | 0.0050936258512863505 | 0.009371053400160716 |

## Formula-ensemble calibration

| task | posterior_draws | calibration_groups | test_groups | nominal_coverage | conformal_rank | sigma_floor | conformal_q | test_rmse | test_r2 | test_point_coverage_90 | test_group_coverage_90 | test_mpiw | raw_gaussian_nll | raw_gaussian_crps | interval_score_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I_photo | 20 | 24 | 16 | 0.9 | 23 | 0.00029042228437708035 | 4.314212280262477 | 0.0042409165793346314 | 0.9645323373353055 | 100.0 | 100.0 | 0.03035663738482016 | -3.872647724426602 | 0.002496963192888735 | 0.03035663738482016 |
