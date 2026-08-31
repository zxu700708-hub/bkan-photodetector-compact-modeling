# Verilog-A 目录说明

本目录围绕当前 terminal-charge 实现和独立 optical-SSAC 公式检查组织，不再使用已退役的 proxy-charge 文件。

## 当前两端口电学模型

- `ge_si_photodetector_terminal_charge.va`：最终静态电流加 quasi-static terminal-charge 实现。
- `ge_si_pdet_fixed.va`：独立静态电流实现。
- `testbench_dc_terminal_charge_ic618.scs`：DC 检查。
- `testbench_ac_terminal_charge_ic618.scs`：AC-admittance 检查。
- `testbench_ac_bias_temp_terminal_charge_ic618.scs`：偏压与温度 AC 检查。
- `testbench_transient_terminal_charge_ic618.scs`：瞬态检查。
- `testbench_transient_terminal_charge_fine_ic618.scs`：细时间步瞬态检查。
- `verify_terminal_charge_spectre.py`：返回结果与冻结参考的独立复核。

有界负载、TIA 和多实例电路测试的可移植输入位于：

```text
artifacts/results/terminal_charge_circuit_validation/preflight_bundle/
```

## 独立 optical-SSAC 公式

- `ge_si_ac_response_symbolic.va`
- `testbench_ac_response_symbolic_ic618.scs`
- `compare_ac_response_symbolic.py`

这些文件只验证 optical-SSAC dB 公式的语言翻译，不构成两端口动态电学支路。

## 支持脚本与参考数据

- `ac_terminal_charge_reference.csv`
- `generate_ac_terminal_charge_reference.py`
- `generate_veriloga_current.py`
- `run_spectre_terminal_charge.sh`
- `run_spectre_terminal_charge_circuits.sh`
- `run_verification.py`
- `terminal_charge_circuit_validation.py`
- `validate_derivatives.py`
- `validate_generalization.py`
- `validate_kan_formulas.py`
- `compare_results.py`
- `kan_to_veriloga.py`

商业求解器输出不作为源码的一部分重新许可；仓库通过版本记录、规范化报告和 SHA-256 绑定其冻结证据。
