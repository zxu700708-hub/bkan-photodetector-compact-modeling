# Four-Task Symbolic and Direct-Export Audit

Only JSON formulas with no KAN inference path are treated as symbolic exports. All reported test sets are isolated from fitting and structural selection.

## Pure formula fidelity

| Task | Formula-network RMSE | Formula-TCAD RMSE | p95 / max TCAD error | Test / dense finite | Derivative finite | Terms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I_dark | 0.0190964 | 0.0163883 | 0.0341602 / 0.07499 | 1.000 / 1.000 | 1.000 | 36 |
| I_photo | 0.00261806 | 0.0034765 | 0.00713709 / 0.00800463 | 1.000 / 1.000 | 1.000 | 69 |
| AC_Response | 0.0974147 | 0.112706 | 0.191398 / 0.598024 | 1.000 / 1.000 | 1.000 | 10 |
| Capacitance | 5.01113e-19 | 5.03197e-19 | 1.11574e-18 / 2.93573e-18 | 1.000 / 1.000 | 1.000 | 24 |

## Distillation-seed stability

| Task | Term Jaccard mean (min) | Prediction disagreement RMSE mean (max) |
| --- | ---: | ---: |
| I_dark | 0.964 (0.946) | 0.00640635 (0.00766945) |
| I_photo | 0.827 (0.806) | 0.000927269 (0.00110566) |
| AC_Response | 0.879 (0.818) | 0.00296924 (0.00445386) |
| Capacitance | 0.947 (0.920) | 2.8973e-19 (4.15819e-19) |

## Direct response-surface exports

Poly3-Ridge and Spline-Ridge were refit on the frozen seed-42 grouped split and independently replayed from JSON. Exact per-task metrics, latency, formula length, Verilog-A line count, and structured holdouts are provided in the accompanying CSV files.

Important scope: the three stability seeds vary distillation optimization while keeping the registered teacher fixed. Dark current, net photocurrent, and capacitance use the q_beta parameter-mean deterministic KAN; AC uses the frozen 500-pass q_beta predictive mean. These runs measure conditional extraction stability, not q_beta-draw or model-seed structural stability.
