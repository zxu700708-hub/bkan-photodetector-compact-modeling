# B-Spline / Finite-Difference / Bayesian Derivative Diagnostics

| task | method_family | method | split | count | derivative_mae | derivative_rmse | derivative_median_abs_error | derivative_medape | derivative_medape_nonzero | mean_derivative_std | reference_derivative_mean | reference_derivative_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AC_Response | Bayesian derivative | bayesian_kan_vi | test | 448.0 | 0.255922 | 0.310498 | 0.227159 | 0.999404 | 0.999404 | 0.0411972 | -0.331235 | 0.243102 |
| AC_Response | finite-difference reference | finite_difference_reference | test | 448.0 | 0 | 0 | 0 | 0 | 0 | nan | -0.331235 | 0.243102 |
| I_dark | B-spline/autograd | deterministic_kan | test | 240.0 | 0.00606205 | 0.0106919 | 0.00281839 | 0.0301214 | 0.0301214 | nan | -0.182156 | 0.20758 |
| I_dark | Bayesian derivative | bayesian_kan_vi | test | 240.0 | 0.0181267 | 0.0429513 | 0.00389405 | 0.0303311 | 0.0303311 | 0.018816 | -0.182156 | 0.20758 |
| I_dark | finite-difference reference | finite_difference_reference | test | 480.0 | 0 | 0 | 0 | 0 | 0 | nan | -0.182156 | 0.20758 |
| I_photo | B-spline/autograd | deterministic_kan | test | 256.0 | 0.00233438 | 0.00393595 | 0.00141514 | 0.0529062 | 0.0529062 | nan | -0.026774 | 0.0197313 |
| I_photo | Bayesian derivative | bayesian_kan_vi | test | 256.0 | 0.00190353 | 0.00376199 | 0.000795564 | 0.042516 | 0.042516 | 0.00349361 | -0.026774 | 0.0197313 |
| I_photo | finite-difference reference | finite_difference_reference | test | 512.0 | 0 | 0 | 0 | 0 | 0 | nan | -0.026774 | 0.0197313 |

Method family mapping:
- deterministic_kan and symbolic_gated_kan: B-spline/autograd derivative columns saved in predictions.csv.
- bayesian_kan_vi, bayesian_kan_dropout, bayesian_kan_hmc: Bayesian derivative recomputed from model checkpoints.
- finite_difference_reference: numerical reference from observed task curves.

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
