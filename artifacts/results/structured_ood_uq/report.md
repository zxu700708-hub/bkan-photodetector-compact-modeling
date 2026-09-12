# Structured OOD Holdout UQ Rerun

## Protocol

- The B-KAN architecture is unchanged: width 8, grid 8, spline order 3, two output channels.
- Training, validation, conformal calibration, ID test, and OOD test are disjoint.
- Axis-tail OOD is assessed on independent test groups; condition-tail OOD holds out complete condition groups.
- Calibration is group-level normalized conformal with a no-shrink 1.96 floor.

## OOD uncertainty trajectory

| Task | Scenario | Seeds | OOD/ID std | OOD AUROC | OOD-ID RMSE | OOD-ID coverage (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ac_response | frequency_high | 3 | 2.339 | 0.938 | 1.8187 | -66.67 |
| ac_response | length_high | 3 | 1.066 | 0.586 | 0.040072 | 0.00 |
| ac_response | temperature_high | 3 | 1.200 | 0.655 | 0.007 | 0.00 |
| capacitance | frequency_high | 3 | 1.100 | 0.699 | 1.0552e-18 | -42.55 |
| capacitance | length_high | 3 | 1.293 | 0.785 | 5.1365e-19 | -4.31 |
| capacitance | temperature_high | 3 | 1.140 | 0.697 | 3.2755e-20 | 0.24 |
| capacitance | trap_high | 3 | 1.139 | 0.694 | 6.739e-20 | -0.04 |
| capacitance | voltage_extreme_reverse | 3 | 0.880 | 0.530 | 8.8227e-19 | -50.04 |
| dark_current | length_high | 3 | 1.101 | 0.695 | 0.0086516 | 0.00 |
| dark_current | temperature_high | 3 | 1.395 | 0.904 | 0.01416 | -12.64 |
| dark_current | trap_high | 3 | 1.106 | 0.676 | 0.0058991 | 2.01 |
| dark_current | voltage_extreme_reverse | 3 | 1.055 | 0.634 | 0.032985 | -50.52 |
| photo_current | length_high | 3 | 1.006 | 0.591 | 0.0029177 | -7.10 |
| photo_current | temperature_high | 3 | 0.850 | 0.158 | 0.0021939 | -1.89 |
| photo_current | trap_high | 3 | 0.971 | 0.580 | 0.0020066 | -0.46 |
| photo_current | voltage_extreme_reverse | 3 | 1.031 | 0.502 | 0.012627 | -35.94 |

Full per-domain calibration, proper scores, uncertainty decomposition, risk-ranking correlation, and saved predictions are in the accompanying CSV files.