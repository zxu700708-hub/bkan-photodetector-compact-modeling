# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| Capacitance | bkan_parameter_mean | 5.77659e-17 | reference_difference_only | 2.72916e-17 |
| Capacitance | dkan | 2.81027e-16 | pass | 4.87044e-21 |
| Capacitance | mlp_l | 4.36851e-16 | pass | 4.2881e-21 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Capacitance | tcad_direct | polynomial | 1 | 8 | 2.02118e-15 | 0.638874 | NA |
| Capacitance | tcad_direct | additive_spline | 4 | 36 | 3.07399e-16 | 0.991647 | NA |
| Capacitance | tcad_direct | additive_spline | 6 | 50 | 2.71056e-16 | 0.993505 | NA |
| Capacitance | tcad_direct | additive_spline | 8 | 64 | 2.65266e-16 | 0.99378 | NA |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
