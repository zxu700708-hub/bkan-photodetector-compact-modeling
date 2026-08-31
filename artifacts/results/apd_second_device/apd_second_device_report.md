# APD Second-Device Matched Grouped Comparison

The independent split unit is `structure_group_id`. This is a device-specific retraining experiment, not zero-shot transfer.

## Mean Test Metrics

| Task | Model | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| APD_I_dark | BKAN-VI | 0.107135 ± 0.0192 | 0.0679454 | 0.991967 |
| APD_I_dark | DKAN | 0.127478 ± 0.0243 | 0.0936005 | 0.98858 |
| APD_I_dark | MLP-L | 0.279108 ± 0.0293 | 0.21099 | 0.946444 |
| APD_I_dark | Poly3-Ridge | 0.357621 ± 0.209 | 0.25069 | 0.886147 |
| APD_I_dark | Spline-Ridge | 0.142993 ± 0.00595 | 0.102888 | 0.986067 |
| APD_I_photo | BKAN-VI | 0.0483634 ± 0.00947 | 0.0315361 | 0.996422 |
| APD_I_photo | DKAN | 0.0810087 ± 0.0119 | 0.0546416 | 0.990123 |
| APD_I_photo | MLP-L | 0.258192 ± 0.00746 | 0.191625 | 0.90166 |
| APD_I_photo | Poly3-Ridge | 0.208287 ± 0.151 | 0.146929 | 0.906953 |
| APD_I_photo | Spline-Ridge | 0.0982156 ± 0.0082 | 0.0715811 | 0.985727 |
| APD_I_net | BKAN-VI | 0.0374651 ± 0.00622 | 0.0192893 | 0.997203 |
| APD_I_net | DKAN | 0.0682561 ± 0.00944 | 0.0405638 | 0.990769 |
| APD_I_net | MLP-L | 0.265604 ± 0.00875 | 0.200345 | 0.862662 |
| APD_I_net | Poly3-Ridge | 0.208278 ± 0.203 | 0.147241 | 0.844201 |
| APD_I_net | Spline-Ridge | 0.0495532 ± 0.00636 | 0.0336154 | 0.995162 |
| APD_M | BKAN-VI | 0.0378032 ± 0.00888 | 0.0185559 | 0.996711 |
| APD_M | DKAN | 0.0610976 ± 0.0133 | 0.0360916 | 0.991457 |
| APD_M | MLP-L | 0.26163 ± 0.013 | 0.197327 | 0.849843 |
| APD_M | Poly3-Ridge | 0.184095 ± 0.145 | 0.133049 | 0.884786 |
| APD_M | Spline-Ridge | 0.0494592 ± 0.00652 | 0.0333818 | 0.994572 |
| APD_V_M10 | BKAN-VI | 0.0210421 ± 0.00481 | 0.0173458 | -0.1861 |
| APD_V_M10 | DKAN | 0.0316835 ± 0.0235 | 0.023734 | -3.305 |
| APD_V_M10 | MLP-L | 0.0387921 ± 0.00972 | 0.0308355 | -3.7298 |
| APD_V_M10 | Poly3-Ridge | 0.240721 ± 0.248 | 0.147091 | -228.259 |
| APD_V_M10 | Spline-Ridge | 0.0503942 ± 0.0119 | 0.0397458 | -6.94536 |
| APD_V_br | BKAN-VI | 0.0372759 ± 0.0391 | 0.0225549 | -0.290892 |
| APD_V_br | DKAN | 0.157438 ± 0.265 | 0.0786326 | -177.792 |
| APD_V_br | MLP-L | 0.0740568 ± 0.0521 | 0.047279 | -12.7617 |
| APD_V_br | Poly3-Ridge | 0.584983 ± 0.865 | 0.379204 | -1164.45 |
| APD_V_br | Spline-Ridge | 0.0975324 ± 0.0402 | 0.0720217 | -18.4694 |

## Prespecified BKAN-VI vs Spline-Ridge Contrast

| Task | BKAN RMSE | Spline RMSE | Delta | Corrected 95% CI | Wins/Losses |
|---|---:|---:|---:|---:|---:|
| APD_I_dark | 0.107135 | 0.142993 | -0.0358581 | [-0.0566358, -0.0150805] | 9/1 |
| APD_I_photo | 0.0483634 | 0.0982156 | -0.0498522 | [-0.0647152, -0.0349892] | 10/0 |
| APD_I_net | 0.0374651 | 0.0495532 | -0.0120881 | [-0.0179703, -0.00620584] | 10/0 |
| APD_M | 0.0378032 | 0.0494592 | -0.0116561 | [-0.0202778, -0.00303436] | 9/1 |
| APD_V_M10 | 0.0210421 | 0.0503942 | -0.0293521 | [-0.0431946, -0.0155095] | 10/0 |
| APD_V_br | 0.0372759 | 0.0975324 | -0.0602565 | [-0.101539, -0.0189738] | 10/0 |

## Scope

- Included: dark current, illuminated current, net photocurrent, multiplication gain, gain-threshold voltage, and breakdown voltage.
- Excluded: AC response, C--V, terminal charge, spatial charge, and noise.
- The evidence supports second-device-class retraining within the same TCAD source, not foundry, measurement, or zero-shot transfer.