# Ten-Split Matched Grouped Comparison

This is a frozen confirmatory comparison. All models use the same grouped splits; validation, calibration, and test condition groups are disjoint. Negative paired delta means lower RMSE for BKAN.

| Task | BKAN RMSE | Prespecified reference | Reference RMSE | Paired delta | Corrected 95% CI | Holm p | Median delta | W/T/L |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I_dark | 0.00769246 | Spline-Ridge | 0.00623143 | 0.00146103 | [-0.00174683, 0.00466888] | 0.6595 | 3.55003e-05 | 5/0/5 |
| I_photo | 0.00502399 | Spline-Ridge | 0.00540899 | -0.000385005 | [-0.00112762, 0.00035761] | 0.8129 | -0.000591852 | 7/0/3 |
| AC_Response | 0.0646352 | Spline-Ridge | 0.111753 | -0.0471179 | [-0.0585856, -0.0356502] | 1.311e-05 | -0.05001 | 10/0/0 |
| Capacitance | 6.85373e-20 | Spline-Ridge | 2.72618e-19 | -2.04081e-19 | [-2.25294e-19, -1.82868e-19] | 1.291e-08 | -2.07435e-19 | 10/0/0 |

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
