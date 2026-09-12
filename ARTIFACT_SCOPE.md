# 代码与数据发布范围

## 当前支持的声明

- 四个 Ge/Si 任务上的十个 matched grouped splits、五模型预设比较与七模型探索性比较。
- APD 数据上七模型 device-specific refitting；不转移权重、normalizer 或符号表达式。
- BKAN-VI、heteroscedastic MC dropout 和 equal-update-budget deep ensemble 的完整响应 UQ 比较。
- 完整 bias curve、frequency curve 和 bias--frequency surface 的经验 no-shrink group conformal audit。
- 结构化边界外推失败与 predictive dispersion 不能作为可靠 OOD 告警的负面证据。
- 受控 960-formula 多教师比较、KAN-free symbolic replay、有限值和导数检查。
- independently fitted terminal-Q、最终 current-plus-charge Verilog-A 源及 Spectre 18.1 device/circuit 报告的 SHA-256 绑定。
- 经高保真结构校准的 reduced-order synthetic 主动微环数据上，单个预设 development split 的七模型点估计。
- 主 Ge/Si、terminal-Q、strict dQ/dV、APD 和主动微环 development canonical 数据。

## 不支持的外延

- 不支持 zero-shot、共享权重、跨工艺或跨 foundry transfer。
- 不支持 measurement-domain validity 或 fabricated-device agreement。
- 不支持 predictive dispersion 是可靠 OOD alarm。
- 不支持微环 frozen-confirmation 结论、重复划分显著性、UQ、符号导出、动态 optical port 或 circuit validation。
- 不支持 dispersive/trap-memory dynamics、PDK qualification、device qualification 或 production deployment。
- 不把历史 HMC、proxy-charge、smoke/trial、弃用六/八模型结果目录或 superseded Spectre 结果传播为当前证据。
- `scripts/run_ring_compact_variation_eight.py` 仅保留为七模型入口的共享代码依赖；当前入口排除 GMLS，且没有发布八模型结果。

## 发布内容边界

本仓库是 `public-code-core`：包含当前论文所需代码、canonical 数据、配置、split、汇总、冻结预测、公式、最终 Verilog-A 源和验证报告；排除论文源文件、模型 checkpoint、原始控制台日志、临时目录、商业求解器返回压缩档和与当前稿件无关的研究分支。

主动微环公开表只包含 320 个 development 结构的响应目标。80 个 frozen-confirmation 结构只发布 ID 与冻结策略，不发布目标值。

主 Ge/Si 数据从冻结的 160-row condition design 与 canonical TCAD 表开始复现。历史 sampling seed、完整商业求解器 native archive 和部分历史源码 revision 不可恢复，因此不声称逐字节重建全部历史 TCAD solve。
