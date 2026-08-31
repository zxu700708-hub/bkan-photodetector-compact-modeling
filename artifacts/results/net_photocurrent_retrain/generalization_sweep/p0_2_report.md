# P0-2 Generalization Sweep — Training Data Fraction Experiment

## Purpose

Quantify how much training data is required for KAN vs MLP to reach a given accuracy level. Directly supports the "accelerating model development" narrative: KAN needs less data.

## Configuration

- Data: `artifacts\results\device_modeling\cleaned_data.csv`
- Tasks: `I_photo`
- Models: `dkan, bkan, mlp_n, mlp_l`
- Fractions: `0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0`
- Sampling seeds per fraction: `42, 43, 44`
- MLP epochs: `300`, lr=`0.002`
- D-KAN steps: `300`, grid=8, k=3, width=8
- B-KAN epochs: `300`, MC=`50`

## Summary Table (mean ± std across sampling seeds)

| task | model | fraction | n_seeds | rmse_target_mean | rmse_target_std | r2_target_mean | r2_target_std | mae_target_mean | n_train_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I_photo | bkan | 0.01 | 3 | 0.0231995 | 0.00122534 | -0.00995565 | 0.104435 | 0.0188232 | 17 |
| I_photo | bkan | 0.05 | 3 | 0.0200048 | 0.000484635 | 0.250696 | 0.0364908 | 0.0159485 | 89 |
| I_photo | bkan | 0.1 | 3 | 0.019495 | 0.000588928 | 0.288168 | 0.0425882 | 0.0157015 | 179 |
| I_photo | bkan | 0.25 | 3 | 0.012558 | 0.000323449 | 0.704699 | 0.0153393 | 0.00915386 | 448 |
| I_photo | bkan | 0.5 | 3 | 0.00672113 | 6.8128e-05 | 0.91546 | 0.00171353 | 0.00494385 | 896 |
| I_photo | bkan | 0.8 | 3 | 0.00457251 | 9.87958e-05 | 0.960858 | 0.00170169 | 0.00362757 | 1433 |
| I_photo | bkan | 1 | 3 | 0.00437567 | 0.000141477 | 0.964134 | 0.00233372 | 0.00348245 | 1792 |
| I_photo | dkan | 0.01 | 3 | 0.0210244 | 0.00109589 | 0.170611 | 0.0848552 | 0.0160303 | 17 |
| I_photo | dkan | 0.05 | 3 | 0.00872703 | 0.000492893 | 0.857028 | 0.0157826 | 0.00645428 | 89 |
| I_photo | dkan | 0.1 | 3 | 0.00609719 | 0.000746367 | 0.929392 | 0.0176275 | 0.00455231 | 179 |
| I_photo | dkan | 0.25 | 3 | 0.00437894 | 6.2883e-05 | 0.964111 | 0.00102618 | 0.00326972 | 448 |
| I_photo | dkan | 0.5 | 3 | 0.00383247 | 0.000145371 | 0.972476 | 0.00206195 | 0.00297567 | 896 |
| I_photo | dkan | 0.8 | 3 | 0.00351062 | 0.000321216 | 0.976745 | 0.00435247 | 0.00274025 | 1433 |
| I_photo | dkan | 1 | 3 | 0.00357539 | 0.000208436 | 0.975998 | 0.00275372 | 0.00284455 | 1792 |
| I_photo | mlp_l | 0.01 | 3 | 0.00864066 | 0.000450845 | 0.85991 | 0.0144088 | 0.0066184 | 17 |
| I_photo | mlp_l | 0.05 | 3 | 0.00525906 | 0.000194491 | 0.948174 | 0.00387532 | 0.0042211 | 89 |
| I_photo | mlp_l | 0.1 | 3 | 0.00479957 | 0.000150394 | 0.956852 | 0.00270899 | 0.00397547 | 179 |
| I_photo | mlp_l | 0.25 | 3 | 0.00462268 | 4.81503e-05 | 0.960008 | 0.000835806 | 0.00383465 | 448 |
| I_photo | mlp_l | 0.5 | 3 | 0.00455796 | 3.23594e-05 | 0.961123 | 0.000552359 | 0.00378262 | 896 |
| I_photo | mlp_l | 0.8 | 3 | 0.00454815 | 1.54377e-05 | 0.961291 | 0.000262674 | 0.0037805 | 1433 |
| I_photo | mlp_l | 1 | 3 | 0.00453595 | 1.04605e-05 | 0.961499 | 0.000177432 | 0.00375641 | 1792 |
| I_photo | mlp_n | 0.01 | 3 | 0.00913473 | 0.000956888 | 0.842143 | 0.0315065 | 0.00705231 | 17 |
| I_photo | mlp_n | 0.05 | 3 | 0.00544443 | 0.000149343 | 0.944491 | 0.00303924 | 0.00433163 | 89 |
| I_photo | mlp_n | 0.1 | 3 | 0.00484216 | 0.000135223 | 0.956091 | 0.00244726 | 0.00398307 | 179 |
| I_photo | mlp_n | 0.25 | 3 | 0.00463901 | 4.79616e-05 | 0.959725 | 0.000834405 | 0.0038399 | 448 |
| I_photo | mlp_n | 0.5 | 3 | 0.00455181 | 3.69052e-05 | 0.961227 | 0.000627036 | 0.00376675 | 896 |
| I_photo | mlp_n | 0.8 | 3 | 0.00453277 | 1.17005e-05 | 0.961553 | 0.00019831 | 0.0037573 | 1433 |
| I_photo | mlp_n | 1 | 3 | 0.00452035 | 7.5441e-06 | 0.961763 | 0.000127607 | 0.00372667 | 1792 |

## Key Findings

### I_photo
- **dkan**: 1% data R²=0.1706 → 100% data R²=0.9760, RMSE 0.0210 → 0.0036
- **bkan**: 1% data R²=-0.0100 → 100% data R²=0.9641, RMSE 0.0232 → 0.0044
- **mlp_n**: 1% data R²=0.8421 → 100% data R²=0.9618, RMSE 0.0091 → 0.0045
- **mlp_l**: 1% data R²=0.8599 → 100% data R²=0.9615, RMSE 0.0086 → 0.0045

## Output Files

- `generalization_rmse_vs_fraction.png` — RMSE vs data fraction curves
- `generalization_r2_vs_fraction.png` — R² vs data fraction curves
- `generalization_kan_advantage.png` — KAN/MLP RMSE ratio
- `metrics_by_run.csv` — Per-run raw metrics
- `metrics_summary.csv` — Aggregated (mean ± std across sampling seeds)

## Notes

- Train/test split is fixed at 70/30 (seed=42) across all fractions.
- Each fraction randomly subsamples the training set `n_seeds` times.
- Test set is always the full 30% held-out data.
- `rmse_target` is in the transformed target space (log10 for DC tasks).