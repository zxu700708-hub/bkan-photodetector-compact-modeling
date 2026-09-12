# 复现说明

## 1. 完整性与发布范围检查

```bash
python verify_artifact.py --root . --output replay_output/verification_report.json
```

验证器执行以下检查：

1. 逐文件复核 `MANIFEST.sha256`；
2. 扫描本机绝对路径、凭据、访问令牌和私钥头；
3. 确认发布树不含 `paper/`、checkpoint、日志、cache、HMC、proxy、smoke/trial 和弃用比较；
4. 检查 18 张 canonical 数据表的行数、SHA-256、字段与微环 confirmation quarantine；
5. 检查主器件五模型预设比较和七模型探索性比较；
6. 检查 APD 七模型 device-specific refit 和零结构泄漏；
7. 检查微环 seed-42 development 比较的 28 行指标、28 份预测与 320-row split；
8. 检查 matched heteroscedastic UQ 的 120 行指标、80 组 baseline、72 个配对对比与零 split overlap；
9. 检查 960 个 JSON 与 960 个 Verilog-A common-family symbolic export；
10. 检查 terminal-Q 数值审计和 corrected current-plus-charge Spectre source binding。

## 2. 冻结结果重放

```bash
python replay_current_evidence.py --root . --output replay_output/current_evidence_replay.json
```

该入口使用 Python 标准库重新聚合正文三张预测表中的数值、matched UQ、受控符号比较、terminal-Q 与 Spectre 状态。它不重新训练、不访问微环确认目标，也不在 test set 上重新选择模型。

## 3. Python 环境与测试

核心验证和结果重放只依赖 Python 标准库。重新训练与单元测试使用：

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
```

CPU/GPU、PyTorch 与线性代数库版本可能造成训练结果最后几位差异，因此论文比较以冻结的 per-seed predictions、split manifests 和配对统计为准。

## 4. Canonical 数据与分组

`data/` 是唯一推荐的数据入口：

- Ge/Si：按 `_curve_id` 保持完整 bias curve、frequency curve 或 `16x25` surface；
- APD：按 `structure_group_id` 保持完整物理结构；
- active microring：按 `nominal_structure_id` 保持全部 45 个 process--bias--temperature 工况。

准确划分见相应 `artifacts/results/*/split_manifest*.csv`。微环 `ring_development_targets.csv` 只有 320 个 development 结构；`ring_frozen_confirmation_ids.csv` 不含任何响应目标。

## 5. 训练与重跑入口

当前主要入口包括：

```bash
python scripts/run_matched_grouped_comparison.py --help
python scripts/run_apd_second_device.py --help
python scripts/run_matched_heteroscedastic_uq.py --help
python scripts/run_multi_teacher_symbolic_pareto_10split.py --help
python scripts/run_ring_compact_variation_seven.py --help
python bkan/device_modeling/terminal_charge/train.py --help
```

训练脚本的历史冻结配置仍保留原始逻辑路径，但发布构建器已将本机绝对路径替换成 `<REPO_ROOT>`、`<FROZEN_APD_ROOT>` 或 `<FROZEN_RING_ROOT>`。重跑时应显式传入本地数据路径。

## 6. Spectre 边界

带 Cadence Spectre 18.1 的环境可以依据发布的 deck 和脚本重新运行。没有许可证的环境只能检查输入、源哈希、规范化报告和冻结数值。通过这些检查只表示已登记工作区间内的实现一致性，不表示 measurement、foundry 或 device qualification。
