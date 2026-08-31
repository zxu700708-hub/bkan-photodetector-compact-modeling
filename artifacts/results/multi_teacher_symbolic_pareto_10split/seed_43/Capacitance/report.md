# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| Capacitance | bkan_parameter_mean | 5.75706e-17 | reference_difference_only | 1.15557e-16 |
| Capacitance | dkan | 1.74771e-16 | pass | 4.5528e-21 |
| Capacitance | mlp_l | 4.24624e-16 | pass | 4.97632e-21 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Capacitance | bkan_parameter_mean | polynomial | 1 | 8 | 1.99398e-15 | 0.631807 | 1.99408e-15 |
| Capacitance | bkan_parameter_mean | additive_spline | 4 | 36 | 2.90639e-16 | 0.992178 | 2.98646e-16 |
| Capacitance | bkan_parameter_mean | additive_spline | 6 | 50 | 2.53562e-16 | 0.994046 | 2.62155e-16 |
| Capacitance | bkan_parameter_mean | additive_spline | 8 | 64 | 2.47627e-16 | 0.994322 | 2.55737e-16 |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
