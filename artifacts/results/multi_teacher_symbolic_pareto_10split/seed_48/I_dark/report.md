# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| I_dark | bkan_parameter_mean | 0.00360537 | reference_difference_only | 0.00137748 |
| I_dark | dkan | 0.0202483 | pass | 4.76837e-07 |
| I_dark | mlp_l | 0.0165701 | pass | 8.88178e-16 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| I_dark | tcad_direct | polynomial | 1 | 7 | 0.048449 | 0.926017 | NA |
| I_dark | tcad_direct | polynomial | 2 | 28 | 0.0212783 | 0.98573 | NA |
| I_dark | tcad_direct | additive_spline | 4 | 31 | 0.00580151 | 0.998939 | NA |
| I_dark | tcad_direct | additive_spline | 6 | 43 | 0.00427152 | 0.999425 | NA |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
