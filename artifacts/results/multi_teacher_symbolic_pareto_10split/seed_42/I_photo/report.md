# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| I_photo | bkan_parameter_mean | 0.00291166 | reference_difference_only | 0.00047063 |
| I_photo | dkan | 0.00330521 | pass | 2.38419e-07 |
| I_photo | mlp_l | 0.00364425 | pass | 4.44089e-16 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| I_photo | tcad_direct | polynomial | 1 | 7 | 0.00704376 | 0.902159 | NA |
| I_photo | dkan | polynomial | 2 | 28 | 0.00387855 | 0.970334 | 0.00355701 |
| I_photo | tcad_direct | additive_spline | 4 | 31 | 0.00366955 | 0.973445 | NA |
| I_photo | tcad_direct | additive_spline | 6 | 43 | 0.00366185 | 0.973557 | NA |
| I_photo | bkan_parameter_mean | additive_spline | 8 | 55 | 0.00335901 | 0.97775 | 0.00243202 |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
