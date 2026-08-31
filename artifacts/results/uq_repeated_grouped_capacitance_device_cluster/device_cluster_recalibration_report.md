# TCAD-Cluster Conformal Recalibration

Generated: 2026-08-04T16:45:21.090068+08:00

- Training checkpoints were reused; no model retraining was performed.
- Calibration unit: one complete TCAD response surface.
- Each calibration TCAD contributes one maximum normalized residual.
- Seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
- Calibration scores per seed: 24
- Calibration points per seed: 9600
- Test DEVICEs per seed: 16

## Aggregate

- Model: bayesian_kan_vi (vi)
- Raw point coverage: 98.64% ± 3.93%
- No-shrink point coverage: 98.85% ± 3.64%
- Raw TCAD coverage: 82.50% ± 12.08%
- No-shrink TCAD coverage: 98.75% ± 3.95%
- Conformal q_raw: 3.2780 ± 0.8643
- No-shrink q: 3.2780 ± 0.8643

## Raw / Standard / No-Shrink Comparison

| Variant | q | Point coverage | TCAD coverage | Subcurve coverage | MPIW |
|---|---:|---:|---:|---:|---:|
| Raw Gaussian | 1.9600 | 98.64% | 82.50% | 97.47% | 2.45192e-19 |
| Standard conformal | 3.2780 | 98.85% | 98.75% | 98.75% | 4.02758e-19 |
| No-shrink conformal | 3.2780 | 98.85% | 98.75% | 98.75% | 4.02758e-19 |

TCAD coverage has 6.25 percentage-point resolution per seed because the test split contains 16 DEVICEs.
