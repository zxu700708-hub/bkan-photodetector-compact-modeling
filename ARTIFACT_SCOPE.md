# 代码与数据发布范围

## 当前支持的声明

- 四个 Ge/Si 任务上的十个 matched grouped splits 及配对统计。
- BKAN-VI、heteroscedastic MC dropout 和 equal-update-budget ensemble 的登记比较。
- 完整 bias curve、frequency curve 和 bias-frequency surface 的校准审计。
- KAN-free symbolic formula 的独立 JSON replay、有限值和导数检查。
- corrected composite Verilog-A 源与 Spectre 18.1 device/circuit 报告之间的 SHA-256 绑定。
- APD 数据上 device-specific retraining 后的 workflow reuse。
- 主 Ge/Si、terminal-Q、strict dQ/dV 和 APD canonical task-level 数据。

## 不支持的外延

- 不支持零样本、共享权重、跨工艺或跨 foundry transfer。
- 不支持 measurement-domain validity。
- 不支持 predictive dispersion 是可靠 OOD alarm。
- 不支持动态 optical port、dispersive/trap-memory dynamics、PDK qualification 或 production deployment。
- 不把历史 proxy-charge 或 superseded Spectre 结果传播为当前 source 的证据。

## 发布边界

RC10 是代码与数据发布候选，不包含论文 LaTeX、补充材料、投稿图源文件、原始控制台日志、模型 checkpoint 或商业求解器返回压缩档。大体积 checkpoint 和原始返回档如需公开，应作为单独的 GitHub Release 或长期归档附件发布，并提供 SHA-256；不建议写入 Git 历史。

主 Ge/Si 数据从冻结的 160-row condition design 与 canonical TCAD 表开始复现。历史 sampling seed、完整商业求解器 native archive 和部分历史源码 revision 不可恢复，因此不声称逐字节重建全部历史 TCAD solve。
