# BKAN 光电探测器建模代码与复现数据

本仓库提供与 BKAN 光电探测器紧凑模型研究配套的源代码、canonical 数据、冻结实验结果、符号公式、Verilog-A 源和哈希绑定的验证报告。RC10 是代码导向的公开发布候选，不包含论文正文、补充材料或投稿图源文件。

## 快速开始

Python 3.10 及以上版本可直接运行完整性与冻结证据复核：

```bash
python verify_artifact.py --root . --output replay_output/verification_report.json
python replay_current_evidence.py --root . --output replay_output/current_evidence_replay.json
```

也可以使用平台脚本：

```powershell
.\reproduce.ps1
```

```bash
bash reproduce.sh
```

成功时进程返回 `0`，并在 `replay_output/` 生成验证报告。`verify_artifact.py` 检查发布树、数据与冻结证据；`replay_current_evidence.py` 重新聚合当前出版数值，并对表观电容应用已登记的 `1000` 倍物理单位修正。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `data/` | Ge/Si 主任务、terminal-Q、strict dQ/dV 和 APD canonical 数据 |
| `bkan/` | KAN/BKAN、光电探测器、terminal-Q 和 Verilog-A 源码 |
| `scripts/` | 分组比较、UQ、统计审计、符号导出和发布元数据工具 |
| `tests/` | 当前代码、数据修正、UQ、符号导出和 terminal-charge 测试 |
| `artifacts/results/` | 冻结配置、划分、指标、公式、预测和接受报告 |
| `MANIFEST.sha256` | 发布树中每个受管文件的 SHA-256 |
| `artifact_metadata.json` | 版本、发布范围、文件计数和逐文件哈希记录 |

数据字段、分组单位和目标定义见 `data/README.md` 与 `data/DATA_DICTIONARY.csv`。软件和数据许可边界分别见 `LICENSE` 与 `DATA_LICENSE.md`。

## 测试

安装训练与测试依赖后运行：

```bash
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
```

测试不会调用商业 TCAD 或 Spectre；相关检查针对已发布的输入、导出结果、哈希绑定和数值后处理。

## 复现层级

1. `verify`：检查 SHA-256 清单、路径与凭据泄漏、canonical 数据、分组结果、matched UQ、符号导出和 Spectre source binding。
2. `replay`：从冻结指标和公式重新聚合出版数值，不重新训练，也不在测试集上重新选择模型。
3. `retrain`：从 `data/` 的 task-level 表重新训练模型；需要 `requirements-training.txt` 中的依赖。
4. 商业求解器重跑：需要用户自行取得 TCAD 和 Cadence Spectre 许可证，不属于本仓库的自动化测试范围。

## 发布与引用

当前候选版本为 `2026.08.31-rc10`，目标仓库为：

```text
https://github.com/zxu700708-hub/bkan-photodetector-compact-modeling
```

冻结标签计划使用 `artifact-v2026.08.31-rc10`。正式归档 DOI 与压缩包 SHA-256 应在实际生成后填写，不能预先编造。引用信息见 `CITATION.cff`，论文中的 Data and code availability 建议文本见 `DATA_AVAILABILITY.md`。

## 声明边界

本仓库支持限定工作区间内的预测、完整响应对象的校准审计、KAN-free 公式重放以及有界 Spectre device/circuit 验证。它不支持测量一致性、跨 foundry 有效性、可靠 OOD 告警、动态 optical port、PDK sign-off、device qualification 或 production deployment 声明。完整边界见 `ARTIFACT_SCOPE.md`。
