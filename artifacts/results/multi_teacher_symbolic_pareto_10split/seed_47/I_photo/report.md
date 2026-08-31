# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| I_photo | bkan_parameter_mean | 0.00431094 | reference_difference_only | 0.00045705 |
| I_photo | dkan | 0.00406858 | pass | 2.38419e-07 |
| I_photo | mlp_l | 0.00430128 | pass | 4.44089e-16 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| I_photo | tcad_direct | polynomial | 1 | 7 | 0.00738073 | 0.89464 | NA |
| I_photo | mlp_l | polynomial | 2 | 28 | 0.00452482 | 0.960401 | 0.00082039 |
| I_photo | mlp_l | additive_spline | 4 | 31 | 0.00430247 | 0.964197 | 0.000926882 |
| I_photo | mlp_l | additive_spline | 6 | 43 | 0.00428846 | 0.96443 | 0.000902699 |
| I_photo | tcad_direct | additive_spline | 8 | 55 | 0.00403552 | 0.968502 | NA |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
