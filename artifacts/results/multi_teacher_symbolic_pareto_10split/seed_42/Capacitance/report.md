# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| Capacitance | bkan_parameter_mean | 2.03785e-16 | reference_difference_only | 3.42002e-17 |
| Capacitance | dkan | 4.61362e-16 | pass | 7.75565e-21 |
| Capacitance | mlp_l | 5.13158e-16 | pass | 5.66453e-21 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Capacitance | mlp_l | polynomial | 1 | 8 | 1.97702e-15 | 0.663704 | 1.89722e-15 |
| Capacitance | tcad_direct | additive_spline | 4 | 36 | 3.94583e-16 | 0.986604 | NA |
| Capacitance | tcad_direct | additive_spline | 6 | 50 | 3.71091e-16 | 0.988152 | NA |
| Capacitance | tcad_direct | additive_spline | 8 | 64 | 3.68533e-16 | 0.988314 | NA |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
