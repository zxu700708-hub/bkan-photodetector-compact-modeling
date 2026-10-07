# Verilog-A 文件与当前验证入口

## 当前净光电流版本：joint 54-term

`ge_si_photodetector_terminal_charge.va` 已替换为 seed 42 的 joint BKAN+TCAD 54-term 净光电流公式（22 个独立特征）。暗电流和终端电荷系数沿用原版本。独立回放及预检查已完成，器件级和电路级 Spectre 验证待重跑。

- 可直接传到 Cadence 主机的包：`../../../release/joint54_validation.zip`（仓库路径为 `release/joint54_validation.zip`）。
- 中文运行说明：[验证包 README](../../../release/joint54_validation/README.md)。
- 打包脚本：`scripts/prepare_joint54_validation.py`。
- 新版对比、源文件哈希与状态：`artifacts/results/joint54_deployment/`。
- 旧 69-term 文件备份：`artifacts/results/joint54_deployment/previous_69term/`。

上面的相对链接从本目录回到仓库根目录；代码路径均以仓库根目录为基准。解压验证包后，在包目录运行：

```bash
bash run_all.sh /absolute/path/joint54_results
```

该入口依次执行全部器件级和电路级测试，并生成验收报告。结果目录应在包外，且不得混入旧结果。

## 当前脚本

- `export_net_photocurrent.py`：从 KAN-free JSON 图导出净光电流 Verilog-A 函数并进行独立回放。
- `verify_terminal_charge_spectre.py`：当前 54-term 器件参考、preflight 与返回结果验收。
- `terminal_charge_circuit_validation.py`：冻结电路参考、生成测试网表、验收电路结果。打包时须显式提供新模型哈希，历史默认哈希仅供旧包复现。
- `run_spectre_terminal_charge.sh`：DC、AC 导纳及两种步长的瞬态测试。
- `run_spectre_terminal_charge_circuits.sh`：负载、TIA 与多实例测试。
- `run_spectre.sh`：默认转入终端电荷验证链。

## 独立 optical-SSAC 读出

`ge_si_ac_response_symbolic.va`、`testbench_ac_response_symbolic_ic618.scs` 和 `compare_ac_response_symbolic.py` 用于独立 dB 公式翻译验证，与两端动态电流支路分开。

## 历史模型

`ge_si_photodetector_charge_proxy.va`、`ge_si_pdet_fixed.va`、proxy 网表及 `_archive_legacy_20260710/` 保留用于历史复现。旧 proxy、旧 69-term 终端模型的验收报告均不能作为当前 54-term 源文件的仿真通过证据。
