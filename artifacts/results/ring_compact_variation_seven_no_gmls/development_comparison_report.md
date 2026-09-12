# Compact process/temperature microring: 7-model development comparison

This exploratory run uses only the 320-structure development partition. The 80 independently seeded frozen-confirmation structures remain unopened. The split unit is the nominal structure; all process, bias, and temperature rows from a structure stay together.

Lower is better. Bold marks the best development-test value and italics the second-best value.

| Model | Resonance-shift RMSE (pm) | Loaded-Q MdAPE (%) | Drop extinction MAE (dB) | Through IL MdAPE (%) | Average rank |
|---|---:|---:|---:|---:|---:|
| BKAN-VI | 11.0545 | **0.644778** | **0.0814012** | **1.37498** | 1.75 |
| DKAN | 10.0348 | *1.03368* | 0.119986 | *1.54805* | 2.50 |
| MLP-L | 11.8635 | 1.17097 | 0.120307 | 1.92759 | 4.00 |
| Spline-Ridge | *0.691123* | 1.5734 | 0.191061 | 3.22513 | 5.00 |
| AutoPINN-adapted | 31.0619 | 1.10637 | *0.105744* | 2.2508 | 3.75 |
| Curve-LUT-PCHIP | 62.2213 | 4.36948 | 0.486963 | 8.96724 | 7.00 |
| Ring SemiEmpirical-CM | **0.663969** | 1.55814 | 0.171783 | 2.87046 | 4.00 |

These are single preset-development-split results, not repeated-split inference or frozen-confirmation results.
