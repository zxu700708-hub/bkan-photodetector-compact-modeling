# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| I_photo | bkan_parameter_mean | 0.00641761 | reference_difference_only | 0.00126481 |
| I_photo | dkan | 0.00568284 | pass | 2.38419e-07 |
| I_photo | mlp_l | 0.00504808 | pass | 4.44089e-16 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| I_photo | mlp_l | polynomial | 1 | 7 | 0.00771443 | 0.884317 | 0.00597926 |
| I_photo | mlp_l | polynomial | 2 | 28 | 0.00525127 | 0.946397 | 0.000986919 |
| I_photo | mlp_l | additive_spline | 4 | 31 | 0.00514739 | 0.948497 | 0.00102001 |
| I_photo | mlp_l | additive_spline | 6 | 43 | 0.00507427 | 0.94995 | 0.000976614 |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
