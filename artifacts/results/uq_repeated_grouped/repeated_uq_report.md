# Repeated Grouped-Split UQ Report

Generated: 2026-07-02T19:38:36+08:00

## Design

- Run label: `confirmatory`
- Seeds completed: `42, 43, 44, 45, 46, 47, 48, 49, 50, 51`
- Split unit: complete condition curve
- Fractions (train/validation/calibration/test): `65%/10%/15%/10%`
- Minimum calibration curves required: `19`
- Validation selects checkpoints; calibration fits the conformal multiplier; test is used only for final metrics.

## Split Audit

- Manifests checked: `30`
- Overlapping condition groups: `0`
- Calibration minimum satisfied: `30/30`

## Aggregate Metrics

| Task | Seeds | RMSE mean | Cal. point PICP | Cal. curve coverage | 95% CI across seeds | Runs <95% | q mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AC_Response | 10 | 0.064635 | 99.71% | 98.12% | [95.97, 100.00] | 3/10 | 1.9600 |
| I_dark | 10 | 0.007692 | 99.38% | 99.38% | [97.96, 100.00] | 1/10 | 2.2951 |
| I_photo | 10 | 0.005024 | 99.49% | 99.38% | [97.96, 100.00] | 1/10 | 2.5473 |

The confidence interval above describes variation across repeated grouped splits. Per-seed Wilson intervals are available in `uq_metrics_by_seed.csv`; reused curves across repetitions mean they should not be interpreted as fully independent pooled observations.
