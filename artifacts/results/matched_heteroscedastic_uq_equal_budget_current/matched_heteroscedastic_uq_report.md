# Matched Heteroscedastic UQ Baselines

## Protocol

- Exact frozen BKAN grouped manifests are replayed for every task and seed.
- All methods use a Gaussian mean/variance head with `softplus(logit)+1e-6` variance.
- MLP fits copy the frozen batch size and learning rate. MC dropout and the optional higher-compute ensemble retain the source schedule; the equal-budget ensemble rescales phase boundaries and the LR schedule to each member's allocated horizon. A second architecture-dependent early stop is disabled so optimizer-update counts are exact.
- MC dropout uses 500 stochastic predictions and dropout rate 0.05; its total epoch/update budget matches the frozen BKAN split exactly.
- Equal-budget deep ensemble uses 5 independently initialized members and distributes the frozen BKAN epoch/update horizon across them; phase boundaries and the LR schedule are scaled by each member's epoch fraction, and total optimizer updates match BKAN exactly.
- Deep ensemble was not run in this invocation. The optional runner uses 5 per-member-matched fits and therefore costs 5x the BKAN optimizer-update budget.
- Calibration uses the untouched calibration groups and the maximum normalized residual within each condition curve or complete DEVICE.
- One training initialization is paired with each split, using `split_seed*1000+member_index`; as in the frozen BKAN run, split and initialization variability are not separately crossed.
- Gaussian NLL and CRPS score the unchanged raw predictive Gaussian; conformal scaling changes only interval diagnostics.

## Key means

| Task | Model | RMSE | Raw NLL | Raw CRPS | Raw group cov. | Cal. group cov. | Cal. MPIW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I_dark | bkan_vi_frozen | 0.00769246 | -2.89872 | 0.00594115 | 99.38% | 99.38% | 0.0967318 |
| I_dark | mc_dropout_mlp | 0.00479197 | -3.16329 | 0.00490089 | 100.00% | 100.00% | 0.0751742 |
| I_dark | equal_budget_deep_ensemble_mlp | 0.0139047 | -2.36332 | 0.0125667 | 93.75% | 99.38% | 0.197357 |
| I_photo | bkan_vi_frozen | 0.00502881 | -3.49247 | 0.00362935 | 96.25% | 99.38% | 0.047705 |
| I_photo | mc_dropout_mlp | 0.00542514 | -3.32973 | 0.00413986 | 85.62% | 99.38% | 0.056514 |
| I_photo | equal_budget_deep_ensemble_mlp | 0.00525927 | -3.32805 | 0.00402748 | 94.38% | 100.00% | 0.0529041 |
| AC_Response | bkan_vi_frozen | 0.0646352 | -1.22622 | 0.0375153 | 98.12% | 98.12% | 0.512704 |
| AC_Response | mc_dropout_mlp | 0.0907452 | -0.0919683 | 0.092763 | 100.00% | 100.00% | 1.4193 |
| AC_Response | equal_budget_deep_ensemble_mlp | 0.0745852 | -0.915921 | 0.0599085 | 75.00% | 91.88% | 0.841577 |
| Capacitance | bkan_vi_frozen | 6.85373e-17 | -36.5569 | 2.50733e-17 | 82.50% | 98.75% | 4.02758e-16 |
| Capacitance | mc_dropout_mlp | 5.44896e-17 | -35.1063 | 7.33215e-17 | 100.00% | 100.00% | 1.18495e-15 |
| Capacitance | equal_budget_deep_ensemble_mlp | 6.29154e-17 | -35.8393 | 6.76826e-17 | 81.88% | 98.75% | 1.16056e-15 |

Corrected paired results are in `paired_statistics.csv`. Positive `baseline_minus_bkan_mean` means the baseline is worse; repeated test splits reuse groups, so the Nadeau--Bengio-style correction is used instead of an independent-split t interval. `corrected_p_value_holm_global` adjusts the full prespecified task x baseline x endpoint family from this invocation; conclusions are based on estimates and corrected intervals, not p values alone.
