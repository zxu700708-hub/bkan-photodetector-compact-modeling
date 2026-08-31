# Ten-Split Matched Grouped Comparison

This is a frozen confirmatory comparison. All models use the same grouped splits; validation, calibration, and test condition groups are disjoint. Negative paired delta means lower RMSE for BKAN.

| Task | BKAN RMSE | Prespecified reference | Reference RMSE | Paired delta | Corrected 95% CI | Holm p | Median delta | W/T/L |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I_photo | 0.00502881 | Spline-Ridge | 0.00540984 | -0.000381024 | [-0.00112961, 0.000367559] | 0.8377 | -0.000589061 | 7/0/3 |

## Protocol

- Seeds: `42, 43, 44, 45, 46, 47, 48, 49, 50, 51`
- Models: `BKAN, DKAN, MLP-L, Poly3-Ridge, Spline-Ridge`
- Split fractions (train/validation/calibration/test): `65%/10%/15%/10%`
- Hyperparameters are frozen in `config.json`; no test-set tuning is performed.
- BKAN point predictions are reused from the frozen confirmatory UQ runs after manifest and target-order verification.
- Win/tie/loss uses paired RMSE with relative tolerance `1e-6 * scale`.
- The main contrast uses the prespecified Spline-Ridge reference for every task; it is not chosen from final-test performance.
- Corrected intervals use the repeated-split variance factor `1/r + test_fraction/train_fraction`; Holm correction covers all four BKAN-vs-baseline contrasts within each task.
- Naive split bootstrap intervals are retained only as descriptive audit columns in `paired_statistics.csv` and are not used for confirmatory claims.

Full model-wise summaries are in `metrics_summary.csv`; every seed/model test prediction is under `predictions/`.
