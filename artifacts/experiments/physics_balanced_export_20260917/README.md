# 物理锚定的暗电流、净光电流和端电荷折中方案

本目录保存论文使用的 post hoc development experiment。候选结构和选择规则是在历史测试结果已可见的情况下形成，因此结果不能作为独立确认。

## 固定方案

三个分组划分共同选择一套超参数。点值 RMSE 不超过登记基线的 105%，完整响应尾部允许 110%，导数指标不超过 105%；满足全部划分后，先取最小系数数，再取平均验证比值较低者。

| 任务 | 原复杂度 | 新复杂度 | 固定配置 |
| --- | ---: | ---: | --- |
| I_dark | 36 | 10 | teacher=0.25, derivative=0.1, ridge=0.0001 |
| I_photo | 54 | 8 | teacher=0.25, derivative=0.1, ridge=0.0001 |
| Q_terminal | 84 | 20 | teacher=0.0, derivative=3.0, ridge=1e-10 |

## 三划分测试汇总

| 任务 | RMSE 原→新 | RMSE 变化 | 完整响应 p95 RMSE 变化 | 导数指标变化 |
| --- | --- | ---: | ---: | ---: |
| I_dark | 0.015306 → 0.00541964 | -64.59% | -60.46% | -80.78% |
| I_photo | 0.00511357 → 0.00453925 | -11.23% | -13.89% | -95.30% |
| Q_terminal | 8.0218e-18 → 5.65685e-18 | -29.48% | -26.44% | -26.32% |

暗电流与净光电流三个划分的点值、完整响应尾部和物理电导误差全部下降。端电荷的 Q 指标三个划分全部下降；独立 dQ/dV 在两个划分下降、一个划分上升，三划分平均仍下降。

## 物理与部署边界

- 电流公式使用 log10(I/A)，由参考偏压条件幅值和在参考偏压严格为零的 C2 偏压修正组成；五个器件参数均保留显式主效应。
- 电荷公式严格满足 Q(0,p)=0，保留标称 V、V^2、V^3 与五个参数敏感项，并提供解析 dQ/dV。
- 所有公式通过 20,000 个电流包络点或 39,360 个电荷点的有限性、方向/正电容和解析导数检查。
- seed 42 公式是当前部署候选。其生成的 Verilog-A 已通过 Spectre 18.1 device-level 与 bounded circuit-level 验收；哈希绑定报告位于 `artifacts/results/physics_sparse_results_py36_20260921/`。这些结果仍不构成外推、新 TCAD 条件、测量或动态光端口验证。
- 电流选择使用非零 BKAN 教师权重；电荷选择为 TCAD 与独立导数直接拟合。这里不能把整体收益单独归因于 BKAN。

独立回放器 `replay.py` 仅依赖 NumPy/pandas；`independent_replay_audit.csv` 记录所有九个公式与冻结预测表的复现误差。

## 复现

在仓库根目录运行：

```powershell
python -B -X utf8 artifacts/experiments/physics_balanced_export_20260917/run_experiment.py --output artifacts/experiments/physics_balanced_export_20260917/development_rerun --seeds 42 43 44
python -B -X utf8 artifacts/experiments/physics_balanced_export_20260917/select_consensus.py --source artifacts/experiments/physics_balanced_export_20260917/development_rerun --output artifacts/experiments/physics_balanced_export_20260917/consensus_rerun
python -B -X utf8 artifacts/experiments/physics_balanced_export_20260917/test_experiment.py -v
python -B -X utf8 artifacts/experiments/physics_balanced_export_20260917/summarize.py
```

`consensus_3split_compact/` 保存逐任务、逐划分的冻结公式、预测和验证前沿；`aggregate.csv` 是三划分汇总；`replay.py` 可单独读取一个公式 JSON 和输入 CSV。输出目录必须不存在，以避免覆盖既有证据。
