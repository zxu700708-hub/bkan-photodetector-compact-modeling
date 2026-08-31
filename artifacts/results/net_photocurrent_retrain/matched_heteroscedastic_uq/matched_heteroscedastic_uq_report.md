# Matched Heteroscedastic UQ Baselines

## Protocol

- Exact frozen BKAN grouped manifests are replayed for every task and seed.
- All methods use a Gaussian mean/variance head with `softplus(logit)+1e-6` variance.
- Each MLP member copies the frozen BKAN realized epoch/update horizon, batch size, learning rate, warm-up, scheduler, and validation-NLL checkpointing. A second architecture-dependent early stop is disabled so the optimizer-update counts are exact.
- MC dropout uses 500 stochastic predictions and dropout rate 0.05; its total epoch/update budget matches the frozen BKAN split exactly.
- Deep ensemble was not run in this invocation. The optional runner uses 5 per-member-matched fits and therefore costs 5x the BKAN optimizer-update budget.
- Calibration uses the untouched calibration groups and the maximum normalized residual within each condition curve or complete TCAD.
- One training initialization is paired with each split, using `split_seed*1000+member_index`; as in the frozen BKAN run, split and initialization variability are not separately crossed.
- Gaussian NLL and CRPS score the unchanged raw predictive Gaussian; conformal scaling changes only interval diagnostics.

## Key means

| Task | Model | RMSE | Raw NLL | Raw CRPS | Raw group cov. | Cal. group cov. | Cal. MPIW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I_photo | bkan_vi_frozen | 0.00502881 | -3.49247 | 0.00362935 | 96.25% | 99.38% | 0.047705 |
| I_photo | mc_dropout_mlp | 0.00542514 | -3.32973 | 0.00413986 | 85.62% | 99.38% | 0.056514 |

Corrected paired results are in `paired_statistics.csv`. Positive `baseline_minus_bkan_mean` means the baseline is worse; repeated test splits reuse groups, so the Nadeau--Bengio-style correction is used instead of an independent-split t interval. `corrected_p_value_holm_global` adjusts the full prespecified task x baseline x endpoint family from this invocation; conclusions are based on estimates and corrected intervals, not p values alone.
