# 数据入口

本目录是代码与数据发布包中唯一推荐的数据入口。所有表均由冻结数据生成，不重新运行 TCAD，也不重新选择模型。`dataset_manifest.json` 记录每张表的行数、字段、来源 SHA-256 和发布文件 SHA-256。

## 主 Ge/Si 数据

| 文件 | 内容 | 完整分组单位 |
| --- | --- | --- |
| `primary/ge_si_dark_current.csv` | 2,400 个暗电流点 | 完整 bias curve |
| `primary/ge_si_photo_current.csv` | 2,560 个配对净光电流点 | 配对完整 bias curve |
| `primary/ge_si_ac_response.csv` | 4,480 个 optical-SSAC 点 | 完整 frequency curve |
| `primary/ge_si_capacitance.csv` | 64,000 个表观电容点 | 完整 `16x25` bias-frequency surface |
| `primary/ge_si_terminal_charge.csv` | 2,080 个 quasi-static terminal-Q 点 | 完整 Q-V curve |
| `primary/ge_si_terminal_charge_dqdv_reference.csv` | 48 个 strict two-contact 对照点 | 独立 SSAC validation point |
| `primary/primary_condition_design.csv` | 160 个冻结器件条件 | device condition |
| `primary/primary_sampling_parameters.csv` | 五个采样参数的范围和分布 | parameter definition |

`ge_si_photo_current.csv` 中的目标是同一器件条件和同一偏压下的
`net_photocurrent = light_current - dark_current`。`ge_si_ac_response.csv` 是归一化光学小信号响应，不是电学 S 参数。`ge_si_capacitance.csv` 的目标定义是 `C_app = Im(Y_total)/(2*pi*f_Hz)`，不作为 terminal-charge 训练目标。

表观电容采用已修正的物理单位。历史 `vac_V` 元数据与实际 `0.001 V` 激励相差 `1000` 倍；修正规则和哈希见 `artifacts/results/apparent_capacitance_correction/manifest.json`。

## APD 第二器件数据

`apd/` 包含六张 task-ready 表，其中论文正文使用前四项：dark current、illuminated current、net photocurrent 和 multiplication gain。所有 APD 划分均以 `structure_group_id` 为独立单位；该实验表示 APD-specific retraining，不表示 zero-shot transfer。

## Split 与预测

canonical 数据本身不预先写入单一 split，因为论文使用十个 matched grouped splits。准确划分见：

```text
artifacts/results/matched_grouped_comparison/split_manifest.csv
artifacts/results/matched_heteroscedastic_uq_equal_budget_current/split_manifest_audit.csv
artifacts/results/apd_second_device/split_manifest.csv
artifacts/results/multi_teacher_symbolic_pareto_10split/split_manifest.csv
```

不得对行进行随机点级拆分；同一 `_curve_id` 或 `structure_group_id` 的所有点必须保持在同一分区。

## 数据来源边界

主 Ge/Si 数据可以从冻结 canonical 表精确重放。原始 160-condition sampling seed、主 campaign 的精确源码 revision 以及完整商业求解器 native archive 未恢复，因此不声称逐字节重建所有历史 TCAD solve。商业软件和许可证不随数据发布。

字段含义见 `DATA_DICTIONARY.csv`，许可边界见仓库根目录 `DATA_LICENSE.md`。
