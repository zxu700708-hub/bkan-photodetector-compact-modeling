# Data and code availability 建议文本

## 当前公开代码版本

```text
The source code, canonical datasets, grouped partitions, frozen predictions,
symbolic models, Verilog-A implementation, and hash-bound verification
materials supporting this study are versioned at
https://github.com/zxu700708-hub/bkan-photodetector-compact-modeling,
release artifact-v2026.09.13-rc11. The active-microring public table contains
the 320-structure development partition; the 80-structure confirmation targets
remain quarantined. Commercial TCAD and Cadence Spectre software are not
redistributed.
```

## 长期归档后使用

获得 DOI 并生成发布压缩包后，可改为：

```text
The data, source code, experiment configurations, grouped data partitions,
frozen predictions, symbolic models, Verilog-A implementation, and verification
materials supporting this study are archived at [ZENODO DOI]. The archived
release is artifact-v2026.09.13-rc11 and its archive SHA-256 is
[ARCHIVE SHA-256]. Commercial TCAD and circuit simulators are not redistributed.
```

只有在远端 tag、Zenodo DOI 和压缩包 SHA-256 实际生成后才能填写对应字段。当前本地 RC11 候选通过验证并不等同于已经完成远端发布。
