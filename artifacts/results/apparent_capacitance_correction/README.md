# 表观电容单位修正

canonical 目标定义为 `C_app = Im(Y_total)/(2*pi*f)`。历史表把 `vac_V` 记录为 `1`，而 TCAD 实际使用 `0.001 V`，因此所有由 admittance 派生的物理输出均乘以 `1000`。这是确定性的源单位修正，不是新的 TCAD 计算或模型拟合。

该目标不是 displacement-current capacitance，也不作为 terminal-charge 证据。`manifest.json` 记录源哈希、`scale_factor=1000.0` 以及各指标的精确变换规则；默认出版复算入口必须读取并应用该修正。
