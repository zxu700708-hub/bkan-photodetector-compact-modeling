# 论文数据入口

本目录是发布包中唯一推荐的数据入口。所有表均由冻结数据生成，不重新运行 TCAD，也不重新选择模型。`dataset_manifest.json` 记录每张表的行数、字段、来源 SHA-256 和发布文件 SHA-256。

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

## APD 第二器件数据

`apd/` 包含六张 task-ready 表，其中论文正文使用前四项：dark current、illuminated current、net photocurrent 和 multiplication gain。所有 APD 划分均以 `structure_group_id` 为独立单位；该实验表示 APD-specific retraining，不表示 zero-shot transfer。

## Active-microring 数据

| 文件 | 内容 | 完整分组单位 |
| --- | --- | --- |
| `ring/ring_development_targets.csv` | 320 个 development 结构的 14,400 个 process--bias--temperature 工况及四个直接拟合指标 | 同一 `nominal_structure_id` 的全部 45 行 |
| `ring/ring_nominal_structures.csv` | 400 个名义结构的设计与数据角色 | nominal structure |
| `ring/ring_process_realizations.csv` | 1,200 个工艺实现 | process realization |
| `ring/ring_frozen_confirmation_ids.csv` | 80 个独立生成的冻结确认结构 ID，不含目标值 | frozen confirmation structure |
| `ring/ring_public_manifest.json` | 生成设计、工艺变化、校准系数、源哈希与声明边界 | manifest |

微环表是经 80 个既有高保真结构校准的 reduced-order synthetic dataset，不是针对新样本重新执行的完整高保真求解器链，也不是测量数据。正文只报告 320 个 development 结构上的 preset seed-42 单划分点估计。公开包不提供 80 个确认结构的目标值，避免破坏后续 frozen-confirmation 评价。

## Split 与预测

canonical 数据本身不预先写入单一 split，因为论文使用十个 matched grouped splits。准确划分见：

```text
artifacts/results/matched_grouped_comparison/split_manifest.csv
artifacts/results/matched_heteroscedastic_uq_equal_budget_current/split_manifest_audit.csv
artifacts/results/apd_second_device/split_manifest.csv
artifacts/results/multi_teacher_symbolic_pareto_10split/split_manifest.csv
artifacts/results/compact_framework_seven_models_pd/split_manifest.csv
artifacts/results/compact_framework_seven_models_apd/split_manifest.csv
artifacts/results/ring_compact_variation_seven_no_gmls/split_manifest.csv
```

不得对行进行随机点级拆分；同一 `_curve_id` 或 `structure_group_id` 的所有点必须保持在同一分区。

## 数据来源边界

主 Ge/Si 数据可以从冻结 canonical 表精确重放。原始 160-condition sampling seed、主 campaign 的精确源码 revision 以及完整商业求解器 native archive 未恢复，因此不声称逐字节重建所有历史 TCAD solve。微环 development 表可从已登记的生成设计和校准系数复核，但完整高保真 anchor solver chain 不在本包内。商业软件和许可证不随数据发布。

字段含义见 `DATA_DICTIONARY.csv`，许可边界见仓库根目录 `DATA_LICENSE.md`。
