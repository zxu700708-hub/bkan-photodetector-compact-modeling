# P0-2 Generalization Sweep — Training Data Fraction Experiment

## Purpose

Quantify how much training data is required for KAN vs MLP to reach a given accuracy level. Directly supports the "accelerating model development" narrative: KAN needs less data.

## Configuration

- Data: `<REPO_ROOT>/artifacts\results\device_modeling\cleaned_data.csv`
- Tasks: `Capacitance`
- Models: `dkan, bkan, mlp_n, mlp_l`
- Fractions: `0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0`
- Sampling seeds per fraction: `42, 43, 44`
- MLP epochs: `300`, lr=`0.002`
- D-KAN steps: `300`, grid=8, k=3, width=8
- B-KAN epochs: `300`, MC=`50`

## Summary Table (mean ± std across sampling seeds)

| task | model | fraction | n_seeds | rmse_target_mean | rmse_target_std | r2_target_mean | r2_target_std | mae_target_mean | n_train_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Capacitance | bkan | 0.01 | 3 | 2.01097e-18 | 1.31334e-19 | 0.630186 | 0.047072 | 1.09903e-18 | 448 |
| Capacitance | bkan | 0.05 | 3 | 3.7215e-19 | 5.77689e-20 | 0.987085 | 0.0040608 | 2.08374e-19 | 2240 |
| Capacitance | bkan | 0.1 | 3 | 2.05905e-19 | 1.3936e-20 | 0.996122 | 0.000534274 | 1.19807e-19 | 4480 |
| Capacitance | bkan | 0.25 | 3 | 1.32977e-19 | 1.4071e-20 | 0.998372 | 0.000350909 | 7.87745e-20 | 11200 |
| Capacitance | bkan | 0.5 | 3 | 9.65537e-20 | 2.65049e-21 | 0.99915 | 4.64511e-05 | 5.30479e-20 | 22400 |
| Capacitance | bkan | 0.8 | 3 | 8.67822e-20 | 5.44672e-21 | 0.999312 | 8.61177e-05 | 4.1693e-20 | 35840 |
| Capacitance | bkan | 1 | 3 | 6.72303e-20 | 1.1156e-20 | 0.999577 | 0.000144603 | 3.24671e-20 | 44800 |
| Capacitance | dkan | 0.01 | 3 | 3.40035e-19 | 8.48394e-20 | 0.988816 | 0.0054847 | 1.80423e-19 | 448 |
| Capacitance | dkan | 0.05 | 3 | 1.7714e-19 | 2.27866e-20 | 0.997095 | 0.000767174 | 1.02055e-19 | 2240 |
| Capacitance | dkan | 0.1 | 3 | 1.7729e-19 | 2.30275e-20 | 0.99709 | 0.000773317 | 1.02614e-19 | 4480 |
| Capacitance | dkan | 0.25 | 3 | 1.74747e-19 | 2.61265e-20 | 0.997157 | 0.000875416 | 1.01787e-19 | 11200 |
| Capacitance | dkan | 0.5 | 3 | 1.66847e-19 | 1.71475e-20 | 0.997438 | 0.000533498 | 9.6217e-20 | 22400 |
| Capacitance | dkan | 0.8 | 3 | 1.62537e-19 | 1.50576e-20 | 0.997574 | 0.000440635 | 9.27839e-20 | 35840 |
| Capacitance | dkan | 1 | 3 | 1.68636e-19 | 2.66006e-20 | 0.997346 | 0.000816222 | 9.80563e-20 | 44800 |
| Capacitance | mlp_l | 0.01 | 3 | 5.35249e-19 | 4.21078e-20 | 0.973751 | 0.00421858 | 3.64946e-19 | 448 |
| Capacitance | mlp_l | 0.05 | 3 | 4.67441e-19 | 4.10253e-20 | 0.97995 | 0.00358739 | 3.31298e-19 | 2240 |
| Capacitance | mlp_l | 0.1 | 3 | 4.51772e-19 | 4.10608e-20 | 0.981262 | 0.00348242 | 3.20554e-19 | 4480 |
| Capacitance | mlp_l | 0.25 | 3 | 4.43299e-19 | 2.6177e-20 | 0.982043 | 0.00215696 | 3.11606e-19 | 11200 |
| Capacitance | mlp_l | 0.5 | 3 | 4.45445e-19 | 3.1287e-20 | 0.981843 | 0.00259797 | 3.13563e-19 | 22400 |
| Capacitance | mlp_l | 0.8 | 3 | 4.4784e-19 | 2.91208e-20 | 0.98166 | 0.00242946 | 3.1525e-19 | 35840 |
| Capacitance | mlp_l | 1 | 3 | 4.47946e-19 | 2.96013e-20 | 0.981649 | 0.00247011 | 3.15715e-19 | 44800 |
| Capacitance | mlp_n | 0.01 | 3 | 6.4325e-19 | 7.4653e-21 | 0.962317 | 0.000875351 | 4.5296e-19 | 448 |
| Capacitance | mlp_n | 0.05 | 3 | 5.61008e-19 | 3.35604e-20 | 0.971238 | 0.0034767 | 4.10886e-19 | 2240 |
| Capacitance | mlp_n | 0.1 | 3 | 5.39936e-19 | 2.98653e-20 | 0.973372 | 0.00299363 | 3.95932e-19 | 4480 |
| Capacitance | mlp_n | 0.25 | 3 | 5.4097e-19 | 3.04762e-20 | 0.973267 | 0.00305952 | 3.95296e-19 | 11200 |
| Capacitance | mlp_n | 0.5 | 3 | 5.43482e-19 | 2.6116e-20 | 0.973042 | 0.0026153 | 3.98214e-19 | 22400 |
| Capacitance | mlp_n | 0.8 | 3 | 5.42235e-19 | 2.80379e-20 | 0.973155 | 0.002806 | 3.96758e-19 | 35840 |
| Capacitance | mlp_n | 1 | 3 | 5.40544e-19 | 2.85167e-20 | 0.97332 | 0.00284949 | 3.9577e-19 | 44800 |

## Key Findings

### Capacitance
- **dkan**: 1% data R²=0.9888 → 100% data R²=0.9973, RMSE 0.0000 → 0.0000
- **bkan**: 1% data R²=0.6302 → 100% data R²=0.9996, RMSE 0.0000 → 0.0000
- **mlp_n**: 1% data R²=0.9623 → 100% data R²=0.9733, RMSE 0.0000 → 0.0000
- **mlp_l**: 1% data R²=0.9738 → 100% data R²=0.9816, RMSE 0.0000 → 0.0000

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