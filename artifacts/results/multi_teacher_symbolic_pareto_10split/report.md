# Ten-Split Data-Budget-Matched Multi-Teacher Symbolic Audit

Seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51. Each seed uses the frozen matched 65/10/15/10 grouped partition. The corrected net-photocurrent task uses the dark-subtracted retrained checkpoints and frozen predictions.

The audit contains 960 fitted exports (96 task/source/family/budget cells across 10 splits). Every JSON evaluator and dense/derivative audit is required to remain finite.

Paired confidence intervals use the repeated-split variance factor `1/r + n_test/n_train`; Holm columns are reported both globally across all 72 teacher/direct contrasts and within each task. The splits reuse the same group pool and are not interpreted as independent datasets.

## Aggregate RMSE--term Pareto composition

Mean-split front source counts: `{"mlp_l": 5, "tcad_direct": 13}`.

## Multiplicity-corrected teacher/direct results

Global-Holm contrasts favoring a teacher: 0; favoring direct TCAD: 12.

| Task | Teacher | Family | Budget | Terms | Mean teacher RMSE | Mean direct RMSE | Corrected CI for teacher-direct | Global Holm p | W/T/L |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| AC_Response | bkan_parameter_mean | additive_spline | 4 | 16 | 0.111996 | 0.11093 | [-0.00116627, 0.00329937] | 1 | 3/0/7 |
| AC_Response | dkan | additive_spline | 4 | 16 | 0.115489 | 0.11093 | [0.00172184, 0.00739756] | 0.2832 | 0/0/10 |
| AC_Response | mlp_l | additive_spline | 4 | 16 | 0.180426 | 0.11093 | [0.0628383, 0.0761552] | 1.464e-07 | 0/0/10 |
| Capacitance | bkan_parameter_mean | additive_spline | 4 | 36 | 3.09653e-16 | 3.07804e-16 | [-2.14479e-18, 5.84239e-18] | 1 | 2/0/8 |
| Capacitance | dkan | additive_spline | 4 | 36 | 3.19018e-16 | 3.07804e-16 | [-7.00458e-18, 2.9433e-17] | 1 | 1/0/9 |
| Capacitance | mlp_l | additive_spline | 4 | 36 | 4.86097e-16 | 3.07804e-16 | [1.569e-16, 1.99685e-16] | 1.023e-06 | 0/0/10 |
| I_dark | bkan_parameter_mean | additive_spline | 4 | 31 | 0.00818978 | 0.00764557 | [-0.000255642, 0.00134406] | 1 | 2/0/8 |
| I_dark | dkan | additive_spline | 4 | 31 | 0.0087138 | 0.00764557 | [0.000100022, 0.00203644] | 1 | 0/0/10 |
| I_dark | mlp_l | additive_spline | 4 | 31 | 0.0178761 | 0.00764557 | [0.00775563, 0.0127055] | 0.0004054 | 0/0/10 |
| I_photo | bkan_parameter_mean | additive_spline | 4 | 31 | 0.00498746 | 0.00508222 | [-0.000417026, 0.000227518] | 1 | 6/0/4 |
| I_photo | dkan | additive_spline | 4 | 31 | 0.00502004 | 0.00508222 | [-0.000282624, 0.000158263] | 1 | 5/0/5 |
| I_photo | mlp_l | additive_spline | 4 | 31 | 0.00472304 | 0.00508222 | [-0.00106489, 0.000346523] | 1 | 7/0/3 |

The four-knot rows above are the exact-complexity comparison used in the main paper. Complete 72-contrast results, all 960 per-export rows, per-seed formulas, split manifests, and teacher replay records accompany this report.

## Teacher replay summary

| Task | Teacher | Splits | Mean teacher-TCAD RMSE | Max replay abs error | Statuses |
| --- | --- | ---: | ---: | ---: | --- |
| AC_Response | bkan_parameter_mean | 10 | 0.0647136 | 0.0392918 | reference_difference_only |
| AC_Response | dkan | 10 | 0.157495 | 7.03335e-05 | mismatch_allowed;pass |
| AC_Response | mlp_l | 10 | 0.174701 | 8.88178e-16 | pass |
| Capacitance | bkan_parameter_mean | 10 | 6.84291e-17 | 1.15557e-16 | reference_difference_only |
| Capacitance | dkan | 10 | 2.2122e-16 | 7.75565e-21 | pass |
| Capacitance | mlp_l | 10 | 4.43922e-16 | 5.82335e-21 | pass |
| I_dark | bkan_parameter_mean | 10 | 0.00757297 | 0.0127583 | reference_difference_only |
| I_dark | dkan | 10 | 0.0198487 | 9.53674e-07 | pass |
| I_dark | mlp_l | 10 | 0.0161239 | 8.88178e-16 | pass |
| I_photo | bkan_parameter_mean | 10 | 0.00503611 | 0.00336818 | reference_difference_only |
| I_photo | dkan | 10 | 0.00535539 | 2.38419e-07 | pass |
| I_photo | mlp_l | 10 | 0.00467397 | 4.44089e-16 | pass |
