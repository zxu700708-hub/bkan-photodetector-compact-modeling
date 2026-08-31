# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| AC_Response | bkan_parameter_mean | 0.0548419 | reference_difference_only | 0.0289703 |
| AC_Response | dkan | 0.114104 | pass | 8.10623e-06 |
| AC_Response | mlp_l | 0.177759 | pass | 8.88178e-16 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| AC_Response | tcad_direct | polynomial | 1 | 4 | 0.916582 | 0.858986 | NA |
| AC_Response | tcad_direct | polynomial | 2 | 10 | 0.311816 | 0.98368 | NA |
| AC_Response | tcad_direct | additive_spline | 4 | 16 | 0.104905 | 0.998153 | NA |
| AC_Response | tcad_direct | additive_spline | 8 | 28 | 0.0985309 | 0.99837 | NA |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
