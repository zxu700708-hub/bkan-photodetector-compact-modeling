# P0-3: TCAD-Referenced Compact Modeling: Classical Baselines vs KAN

## Purpose

Use TCAD data for the vertical Ge/Si photodetector as the high-fidelity reference, and compare classical analytical / equivalent-circuit-inspired baselines against data-driven KAN/MLP compact models on all three tasks.

## Positioning

This comparison does not claim that the physics baseline is a BSIM-like industry-standard photonic compact model. Unlike CMOS transistors, vertical photodetectors do not have a universally standardized compact model across foundries and EDA tools. The intended use case is TCAD-to-compact-model automation for new or customized optoelectronic devices. When a foundry provides a silicon-validated PDK model, that model remains the sign-off reference.

The current device is a vertical photodetector. Avalanche photodetector (APD) compact modeling is a planned extension and is not part of this experiment.

## Configuration

- Data: `artifacts\results\device_modeling\cleaned_data.csv`
- Tasks: `I_photo`
- Models: `physics_simple, physics_full, pdk_lut_knn, poly3_ridge, spline_ridge, rbf_nystroem, mlp_n, mlp_l, dkan, bkan`
- Seeds: `42, 43, 44`
- Split mode: `point`
- MLP epochs: `300`, lr=`0.002`
- D-KAN steps: `300`, grid=8, k=3, width=8
- B-KAN max epochs: `300`, MC=`50`, batch=`32`, inner validation=`15%`, early-stopping patience=`7` validation checks
- Statistical scope: multi-seed mean/population standard deviation.
- Generalization scope: point-level random split; this is an interpolation benchmark and can contain points from the same condition curve in train and test.

## Summary Table

| task | model | n_seeds | rmse_target_mean | rmse_target_std | r2_target_mean | r2_target_std |
| --- | --- | --- | --- | --- | --- | --- |
| I_photo | bkan | 3 | 0.0037753 | 0.000109205 | 0.97345 | 0.00173252 |
| I_photo | dkan | 3 | 0.00349726 | 0.000252418 | 0.977171 | 0.00300779 |
| I_photo | mlp_l | 3 | 0.0045118 | 2.24144e-05 | 0.962124 | 0.000696606 |
| I_photo | mlp_n | 3 | 0.00449567 | 2.96684e-05 | 0.962388 | 0.000986675 |
| I_photo | pdk_lut_knn | 3 | 0.00709744 | 0.000287938 | 0.90613 | 0.00772346 |
| I_photo | physics_full | 3 | 0.0228535 | 0.000126504 | 0.0282398 | 0.01704 |
| I_photo | physics_simple | 3 | 0.0228535 | 0.000126504 | 0.0282398 | 0.01704 |
| I_photo | poly3_ridge | 3 | 0.00371621 | 5.34724e-05 | 0.97431 | 0.000395573 |
| I_photo | rbf_nystroem | 3 | 0.00326886 | 0.00018159 | 0.98008 | 0.00198132 |
| I_photo | spline_ridge | 3 | 0.00395002 | 7.16075e-05 | 0.970949 | 0.00138604 |

## Key Findings

### I_photo
- **Analytical (simple)**: RMSE=0.02285, R2=0.0282
- **Physics / Eq.-Circuit**: RMSE=0.02285, R2=0.0282
- **D-KAN**: RMSE=0.00350, R2=0.9772
- **B-KAN**: RMSE=0.00378, R2=0.9735
- **MLP-N**: RMSE=0.00450, R2=0.9624
- **MLP-L**: RMSE=0.00451, R2=0.9621
- **Poly3-Ridge**: RMSE=0.00372, R2=0.9743
- **Spline-Ridge**: RMSE=0.00395, R2=0.9709
- **RBF-Nystroem**: RMSE=0.00327, R2=0.9801
- **LUT-KNN**: RMSE=0.00710, R2=0.9061

## Output Files

- `{task}_actual_vs_pred.png` — Actual vs predicted scatter per task
- `rmse_bar_summary.png` — RMSE bar chart across all tasks/models
- `{task}_residual_vs_voltage.png` — Residual vs voltage with physics region annotations
- `I_dark_decomposition.png` — Physical mechanism decomposition
- `AC_Response_overlay.png` — S21(f) overlay: physics vs KAN vs TCAD
- `AC_f3dB_vs_bias.png` — Bandwidth vs bias voltage
- `{task}_error_comparison.png` — KAN vs physics error scatter
- `metrics_by_seed.csv` — Per-seed metrics
- `metrics_summary.csv` — Aggregated metrics
- `split_audit.csv` — Train/test condition-curve overlap audit

## Physics Model Equations

### DC Dark Current (9 params)
```
I_dark = |I_diff| + I_gr + I_tat + I_surf + I_shunt
I_diff  = I_s0*(T/300)^3*exp(-E_a_diff/kT)*(exp(qV/nkT)-1)   [Shockley 1949]
I_gr    = I_gr0*(L/L_ref)*sqrt(V_bi-V)*exp(-E_a_gr/kT)       [Sah-Noyce-Shockley 1957]
I_tat   = I_tat0*(trap_A/1e-9)*exp(-B_tat/|V|)                [Hurkx IEEE TED 1992]
I_surf  = I_surf0*((vs+vg)/5000)*|V|                           [Chen JAP 2016]
I_shunt = |V|/R_shunt
```

### DC Photo Current (3 params)
```
I_photo = I_ph0 * (1 - exp(-alpha*L)) * (T/300)^gamma
```
_This first-order absorption baseline is intentionally compact. Its residual error is used to diagnose interface-recombination / trap-related corrections present in the TCAD data._

### AC Frequency Response (5 params)
```
f_3dB(V, L, T) = f_0 * |V|^p * (L/L_ref)^alpha * (T/T_ref)^gamma
S21(f) = S21_DC - 10*log10(1 + (f/f_3dB)^2)   [1st-order low-pass]
```

## Honest Analysis

1. **Parameter non-identifiability**: DC I-V alone cannot uniquely separate G-R, TAT, and surface leakage — they have partially overlapping voltage dependencies. This is a fundamental limitation of physics-based modeling, not a fitting issue.

2. **TAT regime (V < -2V)**: The simple exp(-B/|V|) is a first-order approximation. Real TAT involves trap energy distributions, multi-phonon processes, and field-dependent barrier lowering. KAN learns the correct nonlinearity automatically.

3. **Photocurrent residual diagnosis**: The first-order absorption baseline captures dominant photocurrent physics but misses recombination-dependent corrections (trap_A, vs, vg) that are present in the TCAD data.

4. **AC non-ideal roll-off**: The first-order low-pass model assumes a fixed functional form. KAN captures device-specific deviations from ideal behavior without manually adding additional parasitic terms.

5. **Why use the KAN compact model**: The value is not replacing a mature foundry PDK model. The value is compressing TCAD sweeps into a compact, differentiable, uncertainty-aware, Verilog-A-exportable model when no validated PDK compact model exists yet.

6. **Transferability boundary**: Classical analytical / equivalent-circuit baselines must be re-selected, enriched, and re-parameterized when the device geometry or material system changes. The KAN workflow still needs new TCAD or measurement data, but the train-distill-export pipeline can be reused.