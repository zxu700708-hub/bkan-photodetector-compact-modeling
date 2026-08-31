# Structured Generalization

## Purpose

Test extrapolation-like TCAD development scenarios by holding out physically meaningful regions rather than using only random interpolation splits.

## Configuration

- Data: `artifacts\results\device_modeling\cleaned_data.csv`
- Tasks: `I_photo`
- Models: `pdk_lut_knn, poly3_ridge, spline_ridge, rbf_nystroem, mlp_l, dkan, bkan`
- Holdout fraction: `0.2`
- Stochastic model seeds: `42, 43, 44`
- Deterministic models run once per holdout; stochastic models are summarized across seeds.
- B-KAN epochs / batch size: `300` / `32`

## Summary

| task | scenario | holdout_column | holdout_side | model | n_runs | rmse_target_mean | r2_target_mean | n_train_mean | n_test_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I_photo | voltage_extreme_reverse | light_voltage | low | LUT/kNN interpolation | 1 | 0.0086375 | -0.467856 | 1920 | 640 |
| I_photo | voltage_extreme_reverse | light_voltage | low | Polynomial response surface | 1 | 0.00885216 | -0.541722 | 1920 | 640 |
| I_photo | voltage_extreme_reverse | light_voltage | low | Spline response surface | 1 | 0.00730537 | -0.0500068 | 1920 | 640 |
| I_photo | voltage_extreme_reverse | light_voltage | low | RBF kernel ridge | 3 | 0.00579786 | 0.337822 | 1920 | 640 |
| I_photo | voltage_extreme_reverse | light_voltage | low | MLP-L | 3 | 0.00478013 | 0.550273 | 1920 | 640 |
| I_photo | voltage_extreme_reverse | light_voltage | low | D-KAN | 3 | 0.0165632 | -4.42261 | 1920 | 640 |
| I_photo | voltage_extreme_reverse | light_voltage | low | B-KAN | 3 | 0.017658 | -5.13773 | 1920 | 640 |
| I_photo | length_high | active_layer_length | high | LUT/kNN interpolation | 1 | 0.00887928 | 0.841628 | 2048 | 512 |
| I_photo | length_high | active_layer_length | high | Polynomial response surface | 1 | 0.0196382 | 0.225309 | 2048 | 512 |
| I_photo | length_high | active_layer_length | high | Spline response surface | 1 | 0.00931411 | 0.825736 | 2048 | 512 |
| I_photo | length_high | active_layer_length | high | RBF kernel ridge | 3 | 0.0295188 | -0.878624 | 2048 | 512 |
| I_photo | length_high | active_layer_length | high | MLP-L | 3 | 0.0050789 | 0.948072 | 2048 | 512 |
| I_photo | length_high | active_layer_length | high | D-KAN | 3 | 0.0094909 | 0.813628 | 2048 | 512 |
| I_photo | length_high | active_layer_length | high | B-KAN | 3 | 0.0112256 | 0.746017 | 2048 | 512 |
| I_photo | temperature_high | simulation_temperature | high | LUT/kNN interpolation | 1 | 0.00917403 | 0.841551 | 2048 | 512 |
| I_photo | temperature_high | simulation_temperature | high | Polynomial response surface | 1 | 0.0146774 | 0.594429 | 2048 | 512 |
| I_photo | temperature_high | simulation_temperature | high | Spline response surface | 1 | 0.00766704 | 0.889331 | 2048 | 512 |
| I_photo | temperature_high | simulation_temperature | high | RBF kernel ridge | 3 | 0.0201511 | 0.229397 | 2048 | 512 |
| I_photo | temperature_high | simulation_temperature | high | MLP-L | 3 | 0.00530109 | 0.947089 | 2048 | 512 |
| I_photo | temperature_high | simulation_temperature | high | D-KAN | 3 | 0.00857287 | 0.860646 | 2048 | 512 |
| I_photo | temperature_high | simulation_temperature | high | B-KAN | 3 | 0.00685369 | 0.911425 | 2048 | 512 |
| I_photo | trap_high | trap_assisted_recomb_A | high | LUT/kNN interpolation | 1 | 0.00856067 | 0.853981 | 2048 | 512 |
| I_photo | trap_high | trap_assisted_recomb_A | high | Polynomial response surface | 1 | 0.0191726 | 0.267583 | 2048 | 512 |
| I_photo | trap_high | trap_assisted_recomb_A | high | Spline response surface | 1 | 0.00370803 | 0.972604 | 2048 | 512 |
| I_photo | trap_high | trap_assisted_recomb_A | high | RBF kernel ridge | 3 | 0.0195188 | 0.22443 | 2048 | 512 |
| I_photo | trap_high | trap_assisted_recomb_A | high | MLP-L | 3 | 0.00400369 | 0.968061 | 2048 | 512 |
| I_photo | trap_high | trap_assisted_recomb_A | high | D-KAN | 3 | 0.00902324 | 0.830089 | 2048 | 512 |
| I_photo | trap_high | trap_assisted_recomb_A | high | B-KAN | 3 | 0.00649187 | 0.909396 | 2048 | 512 |

## Output Files

- `metrics_by_split.csv`
- `metrics_summary.csv`
- `structured_generalization_rmse.png`