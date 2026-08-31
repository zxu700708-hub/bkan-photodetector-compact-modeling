# 复现说明

## 1. 完整性与发布卫生检查

```bash
python verify_artifact.py --root . --output replay_output/verification_report.json
```

验证器执行以下检查：

1. 逐文件复核 `MANIFEST.sha256`；
2. 扫描本机绝对路径、凭据赋值、访问令牌和私钥头；
3. 确认代码发布版不包含 `paper/` 和原始 `*.log`；
4. 检查 grouped accuracy 的任务、seed、模型和表观电容物理单位；
5. 检查 14 张 canonical 数据表的行数、SHA-256 和字段清单；
6. 检查 matched heteroscedastic UQ 的指标、baseline、配对对比和 split overlap；
7. 检查 960 个 JSON 与 960 个 Verilog-A symbolic export；
8. 检查 corrected composite source 的 device/circuit 报告和 clean-rerun 状态。

## 2. 冻结结果重放

```bash
python replay_current_evidence.py --root . --output replay_output/current_evidence_replay.json
```

该入口从冻结结果重新聚合 grouped accuracy 与 matched UQ，并复核 symbolic export 和 Spectre source binding。对 `Capacitance`，程序读取 `artifacts/results/apparent_capacitance_correction/manifest.json` 中的 `scale_factor`，将历史 `rmse_target` 转换为 F 后再输出。它不重新训练，也不在测试集上重新选择模型。

## 3. Python 环境与测试

核心验证和重放只依赖 Python 标准库。完整测试使用：

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
```

CPU/GPU 和库版本可能造成训练结果最后几位差异，因此正式比较以冻结的 per-seed predictions、split manifests 和配对统计为准。

## 4. Canonical 数据与重新训练

`data/` 是唯一推荐的数据入口。主任务分组键为 `_curve_id`，APD 分组键为 `structure_group_id`；不得进行点级随机拆分。十组划分位于相应的 `artifacts/results/*/split_manifest*.csv`，完整训练配置位于结果目录的 `config.json` 和每个 seed 的运行元数据中。

## 5. Spectre 边界

带 Cadence Spectre 18.1 的环境可以依据发布的 deck 和脚本重新运行。没有许可证的环境只能检查输入、源哈希、规范化报告和冻结数值。通过这些检查只表示已登记工作区间内的实现一致性，不表示 measurement、foundry 或 device qualification。

## 6. 重建发布清单

整理发布树后运行：

```bash
python scripts/rebuild_release_metadata.py --root .
```

该命令拒绝包含 `paper/` 或 `*.log` 的发布树，并重新生成 `artifact_metadata.json` 与 `MANIFEST.sha256`。
