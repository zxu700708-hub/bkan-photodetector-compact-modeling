# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| I_dark | bkan_parameter_mean | 0.0331476 | reference_difference_only | 0.00291266 |
| I_dark | dkan | 0.0411734 | pass | 4.76837e-07 |
| I_dark | mlp_l | 0.0157914 | pass | 8.88178e-16 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| I_dark | bkan_parameter_mean | polynomial | 1 | 7 | 0.0478243 | 0.953028 | 0.0562022 |
| I_dark | tcad_direct | polynomial | 2 | 28 | 0.0208595 | 0.991064 | NA |
| I_dark | tcad_direct | polynomial | 3 | 84 | 0.0109075 | 0.997557 | NA |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
