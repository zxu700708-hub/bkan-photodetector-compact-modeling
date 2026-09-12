# TCAD 来源追溯边界

本目录区分已恢复证据、部分恢复证据和当前快照中不可恢复的历史信息。冻结的
160-condition 参数表及其采样参数表按原字节打包；160 个逐条件提取文件、保留的
Lumerical native sessions、后期源码快照和 canonical derivative artifacts 均在
CSV/JSON 清单中登记 SHA-256。

原始 sampling seed 和主 campaign 实际执行的精确 source-code revision 未能恢复。
只有 4 个 FDTD 与 4 个 DEVICE native projects 匹配主 campaign 日期，因此不能把
native archive 描述为完整。Mesh state 保存在已留存的 native projects 中；清单只
发布由日志提取并脱敏的 grid/vertex/element 摘要。商业二进制 projects 以及包含主机、
许可和本地路径信息的原始日志不纳入匿名 artifact。论文所报机器学习实验从打包的冻结
condition/canonical tables 开始复现，不声称能够重新生成每一次原始 TCAD solve 的逐字节
等价副本。
