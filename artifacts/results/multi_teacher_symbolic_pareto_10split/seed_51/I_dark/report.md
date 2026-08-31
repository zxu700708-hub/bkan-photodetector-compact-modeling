# Data-Budget-Matched Multi-Teacher Symbolic Export Pareto

All export students and TCAD-direct controls use identical training rows. Validation selects ridge alpha at each fixed complexity budget; calibration is untouched and test is evaluated only after freezing.

Scope: these common polynomial/additive-spline KAN-free response-surface exports isolate teacher choice under identical student families. They do not rerun or replace the task-specific mechanism-gated symbolic formulas in the main audit.

Runtime note: the frozen BKAN checkpoints were loaded under scikit-learn 1.8.0 and emitted an estimator-version warning for scaler objects serialized under 1.9.0. Reproduction should use the recorded environment or regenerate teachers.

## Teacher audit

| Task | Teacher | Teacher-TCAD RMSE | Replay status | Replay max abs |
| --- | --- | ---: | --- | ---: |
| I_dark | bkan_parameter_mean | 0.00902686 | reference_difference_only | 0.00855729 |
| I_dark | dkan | 0.0229226 | pass | 4.76837e-07 |
| I_dark | mlp_l | 0.0169224 | pass | 8.88178e-16 |

## RMSE/term Pareto front

| Task | Source | Family | Budget | Terms | Formula-TCAD RMSE | R2 | Formula-source RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| I_dark | tcad_direct | polynomial | 1 | 7 | 0.0482193 | 0.937545 | NA |
| I_dark | tcad_direct | polynomial | 2 | 28 | 0.0211568 | 0.987977 | NA |
| I_dark | tcad_direct | additive_spline | 4 | 31 | 0.00630265 | 0.998933 | NA |
| I_dark | tcad_direct | additive_spline | 6 | 43 | 0.00499115 | 0.999331 | NA |
| I_dark | tcad_direct | additive_spline | 8 | 55 | 0.00495178 | 0.999341 | NA |

The BKAN replay row compares a variational-parameter-mean deterministic KAN against the historical MC predictive mean and is therefore a reference difference, not a deterministic replay test. DKAN and MLP replay rows must pass.
