# Direct active-microring device-metric comparison

All results use the frozen 80-structure repeated splits. BKAN-VI and DKAN share width 8, grid 8, and cubic splines; their output heads differ only as required for probabilistic versus deterministic prediction.

| Task | Model | Model-space RMSE | Raw RMSE | Raw MAE | Median APE (%) | R2 raw |
|---|---|---:|---:|---:|---:|---:|
| resonance_shift | BKAN-VI | 10.9537 ± nan | 10.9537 | 9.0904 | 1.19849 | 0.999844 |
| resonance_shift | DKAN | 10.4972 ± nan | 10.4972 | 7.48747 | 0.450253 | 0.999856 |
| resonance_shift | MLP-L | 12.6197 ± nan | 12.6197 | 10.1544 | 0.910782 | 0.999793 |
| resonance_shift | Spline-Ridge | 0.677571 ± nan | 0.677571 | 0.453145 | 0.0374085 | 0.999999 |
| resonance_shift | AutoPINN-adapted | 31.8929 ± nan | 31.8929 | 23.0385 | 1.25581 | 0.998675 |
| resonance_shift | Curve-LUT-PCHIP | 62.2255 ± nan | 62.2255 | 9.14093 | 0.0147766 | 0.994956 |
| resonance_shift | Ring SemiEmpirical-CM | 0.652149 ± nan | 0.652149 | 0.440307 | 0.0373568 | 0.999999 |
| quality_factor | BKAN-VI | 0.00605429 ± nan | 637.516 | 330.959 | 0.644778 | 0.994935 |
| quality_factor | DKAN | 0.00797853 ± nan | 787.453 | 485.522 | 1.03368 | 0.992273 |
| quality_factor | MLP-L | 0.00728148 ± nan | 619.174 | 472.388 | 1.17097 | 0.995223 |
| quality_factor | Spline-Ridge | 0.0126793 ± nan | 1099.49 | 778.626 | 1.5734 | 0.984936 |
| quality_factor | AutoPINN-adapted | 0.00631157 ± nan | 552.993 | 421.045 | 1.10637 | 0.996189 |
| quality_factor | Curve-LUT-PCHIP | 0.0326936 ± nan | 2961.55 | 2025.74 | 4.36948 | 0.890705 |
| quality_factor | Ring SemiEmpirical-CM | 0.010773 ± nan | 920.505 | 693.191 | 1.55814 | 0.989441 |
| drop_extinction | BKAN-VI | 0.12579 ± nan | 0.12579 | 0.0814012 | 0.132151 | 0.996484 |
| drop_extinction | DKAN | 0.1653 ± nan | 0.1653 | 0.119986 | 0.225332 | 0.993928 |
| drop_extinction | MLP-L | 0.149783 ± nan | 0.149783 | 0.120307 | 0.242437 | 0.995014 |
| drop_extinction | Spline-Ridge | 0.255342 ± nan | 0.255342 | 0.191061 | 0.322274 | 0.985511 |
| drop_extinction | AutoPINN-adapted | 0.128603 ± nan | 0.128603 | 0.105744 | 0.228823 | 0.996325 |
| drop_extinction | Curve-LUT-PCHIP | 0.652043 ± nan | 0.652043 | 0.486963 | 0.886926 | 0.905519 |
| drop_extinction | Ring SemiEmpirical-CM | 0.217083 ± nan | 0.217083 | 0.171783 | 0.320387 | 0.989528 |
| through_insertion | BKAN-VI | 0.0183592 ± nan | 7.09955e-06 | 4.54503e-06 | 1.37498 | 0.995943 |
| through_insertion | DKAN | 0.0139068 ± nan | 6.23568e-06 | 4.50284e-06 | 1.54805 | 0.99687 |
| through_insertion | MLP-L | 0.0130184 ± nan | 8.85398e-06 | 5.93705e-06 | 1.92759 | 0.993689 |
| through_insertion | Spline-Ridge | 0.0249221 ± nan | 1.59738e-05 | 1.03645e-05 | 3.22513 | 0.97946 |
| through_insertion | AutoPINN-adapted | 0.0136139 ± nan | 7.17269e-06 | 5.49884e-06 | 2.2508 | 0.995859 |
| through_insertion | Curve-LUT-PCHIP | 0.0731676 ± nan | 3.79206e-05 | 2.70678e-05 | 8.96724 | 0.884245 |
| through_insertion | Ring SemiEmpirical-CM | 0.021823 ± nan | 1.36397e-05 | 9.39491e-06 | 2.87046 | 0.985024 |
