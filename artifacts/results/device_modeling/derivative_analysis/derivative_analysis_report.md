# Derivative Analysis

| task | method | split | count | derivative_mae | derivative_rmse | derivative_median_abs_error | derivative_medape | derivative_medape_nonzero | mean_derivative_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AC_Response | bayesian_kan_vi | test | 448.0 | 0.255922 | 0.310498 | 0.227159 | 0.999404 | 0.999404 | 0.0411972 |
| I_dark | bayesian_kan_vi | test | 240.0 | 0.0181267 | 0.0429513 | 0.00389405 | 0.0303311 | 0.0303311 | 0.018816 |
| I_dark | deterministic_kan | test | 240.0 | 0.00606205 | 0.0106919 | 0.00281839 | 0.0301214 | 0.0301214 | nan |
| I_photo | bayesian_kan_vi | test | 256.0 | 0.00190353 | 0.00376199 | 0.000795564 | 0.042516 | 0.042516 | 0.00349361 |
| I_photo | deterministic_kan | test | 256.0 | 0.00233438 | 0.00393595 | 0.00141514 | 0.0529062 | 0.0529062 | nan |

## Missing Or Skipped Inputs

- I_dark: no saved prediction directory for symbolic_gated_kan
- I_dark: no checkpoint for bayesian_kan_dropout
- I_dark: no checkpoint for bayesian_kan_hmc
- I_photo: no saved prediction directory for symbolic_gated_kan
- I_photo: no checkpoint for bayesian_kan_dropout
- I_photo: no checkpoint for bayesian_kan_hmc
- AC_Response: <REPO_ROOT>/artifacts\results\device_modeling\ac_response_deterministic_kan\predictions.csv has no derivative columns
- AC_Response: no saved prediction directory for symbolic_gated_kan
- AC_Response: no checkpoint for bayesian_kan_dropout
- AC_Response: no checkpoint for bayesian_kan_hmc
