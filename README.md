# KAN 光电器件紧凑建模代码与复现数据

本仓库对应论文《A Falsification-First Framework for Optoelectronic Compact Modeling with Kolmogorov--Arnold Networks》的公开代码与数据包。内容按论文当前实验边界整理，仅保留 Ge/Si 光电探测器、APD、主动微环、分组不确定性评估、符号导出和 current-plus-charge Verilog-A 验证所需文件。

## 快速复核

Python 3.10 及以上版本可直接运行完整性与冻结证据复核，不需要安装第三方依赖：

```bash
python verify_artifact.py --root . --output replay_output/verification_report.json
python replay_current_evidence.py --root . --output replay_output/current_evidence_replay.json
```

也可以使用平台入口：

```powershell
.\reproduce.ps1
```

```bash
bash reproduce.sh
```

成功时进程返回 `0`，并在 `replay_output/` 生成两份 JSON 报告。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `data/primary/` | 四个 Ge/Si 监督任务、condition design、terminal-Q 与独立 dQ/dV 对照 |
| `data/apd/` | APD 四个曲线任务和两个低方差电压指标 |
| `data/ring/` | 主动微环 development 数据、设计/工艺表和不含目标值的冻结确认 ID |
| `bkan/` | KAN/BKAN、任务数据接口、符号模型、terminal-Q 和最终 Verilog-A 相关源码 |
| `scripts/` | 当前七模型比较、UQ、OOD、符号导出、微环和发布审计入口 |
| `tests/` | 只针对当前论文路径的单元测试 |
| `artifacts/results/` | 当前论文使用的冻结配置、split、预测、指标、公式和验证报告 |
| `MANIFEST.sha256` | 发布树中每个受管文件的 SHA-256 |
| `artifact_metadata.json` | 版本、构建策略和逐文件来源哈希 |

本代码仓库不包含论文 TeX/PDF、Web UI、本地虚拟环境、研究笔记、原始控制台日志、HMC 失败研究、旧 proxy-charge 证据、smoke/trial 结果或已弃用的六/八模型比较。

## 当前实验对应关系

| 论文证据 | 冻结目录 |
| --- | --- |
| 五模型预设主比较 | `artifacts/results/matched_grouped_comparison/` |
| 七模型 Ge/Si PD 比较 | `artifacts/results/compact_framework_seven_models_pd/` |
| 七模型 APD device-specific refit | `artifacts/results/compact_framework_seven_models_apd/` |
| BKAN-VI、MC dropout 与 deep ensemble UQ | `artifacts/results/matched_heteroscedastic_uq_equal_budget_current/` |
| 结构化外推/OOD | `artifacts/results/structured_generalization_capacitance/`、`artifacts/results/structured_ood_uq/` |
| 受控多教师符号比较 | `artifacts/results/multi_teacher_symbolic_pareto_10split/` |
| current-plus-charge 数值与 Spectre 验证 | `artifacts/results/terminal_charge_model/`、`artifacts/results/corrected_composite_spectre/` |
| 主动微环单 development split | `artifacts/results/ring_compact_variation_seven_no_gmls/` |

完整机器可读索引见 `artifacts/results/evidence_audit/current_evidence_index.json`。

## 数据与划分边界

`data/` 是唯一推荐的训练数据入口。Ge/Si 曲线和表面使用 `_curve_id` 分组，APD 使用 `structure_group_id`，主动微环使用 `nominal_structure_id`。不得对同一完整响应对象做点级随机拆分。

主动微环结果是经 80 个既有高保真结构校准的 reduced-order synthetic development comparison。正文只报告 320 个 development 结构上的 seed-42 单划分点估计；80 个 independently seeded confirmation 结构仍未用于模型选择或报告，其公开文件只有 ID，不含响应目标。

## 软件与声明边界

商业 TCAD 求解器和 Cadence Spectre 不随仓库分发。无商业许可证时仍可复核 canonical 表、split、冻结预测、公式、源哈希和接受报告。本包支持论文限定的 in-envelope prediction、完整响应的经验校准、KAN-free formula replay 以及有界电气 simulator checks；不支持 measurement validity、可靠 OOD 告警、zero-shot device transfer、microring confirmation、动态 optical port、PDK/foundry sign-off 或 production deployment 声明。

更多信息见 `ARTIFACT_SCOPE.md`、`REPRODUCIBILITY.md` 和 `DATA_AVAILABILITY.md`。
