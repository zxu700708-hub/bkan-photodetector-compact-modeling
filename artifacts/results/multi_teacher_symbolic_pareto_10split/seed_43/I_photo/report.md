# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| I_photo | bkan_parameter_mean | 0.00602676 | reference_difference_only | 0.000421979 |
| I_photo | dkan | 0.00652336 | pass | 2.38419e-07 |
| I_photo | mlp_l | 0.00583095 | pass | 4.44089e-16 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| I_photo | tcad_direct | polynomial | 1 | 7 | 0.00835433 | 0.873928 | NA |
| I_photo | mlp_l | polynomial | 2 | 28 | 0.00603627 | 0.934184 | 0.000884462 |
| I_photo | mlp_l | additive_spline | 4 | 31 | 0.00588754 | 0.937387 | 0.00104807 |
| I_photo | mlp_l | additive_spline | 6 | 43 | 0.0058463 | 0.938261 | 0.000922319 |
| I_photo | tcad_direct | additive_spline | 8 | 55 | 0.00547127 | 0.945928 | NA |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
