# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| Capacitance | bkan_parameter_mean | 4.83746e-17 | reference_difference_only | 3.91171e-17 |
| Capacitance | dkan | 1.76747e-16 | pass | 4.49986e-21 |
| Capacitance | mlp_l | 4.47821e-16 | pass | 4.60574e-21 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Capacitance | bkan_parameter_mean | polynomial | 1 | 8 | 2.00238e-15 | 0.647771 | 2.00574e-15 |
| Capacitance | bkan_parameter_mean | additive_spline | 4 | 36 | 2.96187e-16 | 0.992293 | 2.79201e-16 |
| Capacitance | bkan_parameter_mean | additive_spline | 6 | 50 | 2.57427e-16 | 0.994178 | 2.38968e-16 |
| Capacitance | bkan_parameter_mean | additive_spline | 8 | 64 | 2.51358e-16 | 0.99445 | 2.32537e-16 |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
