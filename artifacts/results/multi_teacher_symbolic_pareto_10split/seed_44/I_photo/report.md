# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| I_photo | bkan_parameter_mean | 0.00423973 | reference_difference_only | 0.000875711 |
| I_photo | dkan | 0.00460413 | pass | 2.38419e-07 |
| I_photo | mlp_l | 0.00459854 | pass | 4.44089e-16 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| I_photo | bkan_parameter_mean | polynomial | 1 | 7 | 0.00755408 | 0.888032 | 0.00681427 |
| I_photo | bkan_parameter_mean | polynomial | 2 | 28 | 0.0046921 | 0.956802 | 0.00345995 |
| I_photo | bkan_parameter_mean | additive_spline | 4 | 31 | 0.00427506 | 0.96414 | 0.00279219 |
| I_photo | bkan_parameter_mean | additive_spline | 6 | 43 | 0.00410101 | 0.967 | 0.00255915 |
| I_photo | bkan_parameter_mean | additive_spline | 8 | 55 | 0.00373624 | 0.972609 | 0.0022549 |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
