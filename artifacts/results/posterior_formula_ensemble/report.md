# Posterior-Sampled Formula Ensemble Audit

- Fixed VI posterior draws per task: 20
- BKAN architecture/training: unchanged; checkpoints loaded read-only
- Formula optimizer seed: fixed across posterior draws
- Calibration groups are not used for formula fitting or selection

## Stability

| task | pair_count | term_jaccard_mean | term_jaccard_min | prediction_disagreement_rmse_mean | prediction_disagreement_rmse_max |
| --- | --- | --- | --- | --- | --- |
| AC_Response | 190 | 0.6867483393799184 | 0.42857142857142855 | 0.05881379963635363 | 0.15248817927171157 |
| Capacitance | 190 | 0.992 | 0.92 | 1.008841815332246e-19 | 4.320561906213421e-19 |
| I_dark | 190 | 0.9601706970128022 | 0.9459459459459459 | 0.016921625448238845 | 0.03443578317404242 |
| I_photo | 190 | 0.8991909057439703 | 0.8 | 0.005151003865991861 | 0.015603970759733432 |

## Formula-ensemble calibration

| task | posterior_draws | calibration_groups | test_groups | nominal_coverage | conformal_rank | sigma_floor | conformal_q | test_rmse | test_r2 | test_point_coverage_90 | test_group_coverage_90 | test_mpiw | raw_gaussian_nll | raw_gaussian_crps | interval_score_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I_dark | 20 | 16 | 16 | 0.9 | 16 | 0.0010799103175503349 | 3.4236053577151826 | 0.01225356055611951 | 0.995451424313732 | 100.0 | 100.0 | 0.08142194257081521 | -2.946807850337116 | 0.006928596042460154 | 0.08142194257081521 |
| I_photo | 20 | 16 | 16 | 0.9 | 16 | 0.0003499139368597932 | 4.451040042081064 | 0.006154818528363081 | 0.9273310389665144 | 99.609375 | 93.75 | 0.035835881291120815 | -3.1758771152019047 | 0.003836980383398105 | 0.03584409635425758 |
| AC_Response | 20 | 16 | 16 | 0.9 | 16 | 0.002323013285049796 | 10.687533844163708 | 0.1281381853364462 | 0.9972295543691073 | 99.55357142857143 | 93.75 | 0.7915489963925272 | 4.948102149476475 | 0.07944565663336885 | 0.7940004044286055 |
| Capacitance | 20 | 16 | 16 | 0.9 | 16 | 6.40715146133168e-21 | 49.314869123661026 | 5.746879549101077e-19 | 0.9690431340424254 | 99.90625 | 75.0 | 7.80320782780768e-18 | 22.102006414720346 | 3.8478424216468307e-19 | 7.804571334342487e-18 |
