# Structured OOD Holdout UQ Rerun

## Protocol

- The B-KAN architecture is unchanged: width 8, grid 8, spline order 3, two output channels.
- Training, validation, conformal calibration, ID test, and OOD test are disjoint.
- Axis-tail OOD is assessed on independent test groups; condition-tail OOD holds out complete condition groups.
- Calibration is group-level normalized conformal with a no-shrink 1.96 floor.

## OOD uncertainty trajectory

| Task | Scenario | Seeds | OOD/ID std | OOD AUROC | OOD-ID RMSE | OOD-ID coverage (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| photo_current | length_high | 3 | 1.007 | 0.594 | 0.0029116 | -6.90 |
| photo_current | temperature_high | 3 | 0.851 | 0.159 | 0.0021882 | -1.89 |
| photo_current | trap_high | 3 | 0.971 | 0.579 | 0.0020049 | -0.46 |
| photo_current | voltage_extreme_reverse | 3 | 1.032 | 0.502 | 0.012571 | -35.42 |

Full per-domain calibration, proper scores, uncertainty decomposition, risk-ranking correlation, and saved predictions are in the accompanying CSV files.