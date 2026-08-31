# Repeated Grouped-Split UQ Report

Generated: 2026-08-21T15:51:40+08:00

## Design

- Run label: `net_photocurrent_confirmatory`
- Bayesian inference method: `vi`
- Seeds completed: `42, 43, 44, 45, 46, 47, 48, 49, 50, 51`
- Split unit: independent group (condition curve; capacitance TCAD)
- Fractions (train/validation/calibration/test): `65%/10%/15%/10%`
- Minimum calibration groups required: `19`
- Validation selects checkpoints; calibration fits the conformal multiplier; test is used only for final metrics.

## Split Audit

- Manifests checked: `10`
- Overlapping condition groups: `0`
- Calibration minimum satisfied: `10/10`

## Aggregate Metrics

| Task | Seeds | RMSE mean | Cal. point PICP | Cal. group coverage | 95% CI across seeds | Runs <95% | q mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| I_photo | 10 | 0.005029 | 99.45% | 99.38% | [97.96, 100.00] | 1/10 | 2.5488 |

The confidence interval above describes variation across repeated grouped splits. Per-seed Wilson intervals are available in `uq_metrics_by_seed.csv`; reused groups across repetitions mean they should not be interpreted as fully independent pooled observations.
