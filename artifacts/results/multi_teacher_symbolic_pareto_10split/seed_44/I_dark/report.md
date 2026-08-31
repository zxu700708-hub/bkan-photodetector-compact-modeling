# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| I_dark | bkan_parameter_mean | 0.00310864 | reference_difference_only | 0.00165458 |
| I_dark | dkan | 0.0210633 | pass | 4.76837e-07 |
| I_dark | mlp_l | 0.0155621 | pass | 8.88178e-16 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| I_dark | bkan_parameter_mean | polynomial | 1 | 7 | 0.0472569 | 0.91764 | 0.0468956 |
| I_dark | tcad_direct | polynomial | 2 | 28 | 0.0206801 | 0.984228 | NA |
| I_dark | bkan_parameter_mean | additive_spline | 4 | 31 | 0.00514553 | 0.999024 | 0.00558824 |
| I_dark | bkan_parameter_mean | additive_spline | 6 | 43 | 0.00349882 | 0.999549 | 0.00435577 |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
