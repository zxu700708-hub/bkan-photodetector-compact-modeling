from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "consensus_3split_compact"
sys.path.insert(0, str(HERE))
import replay as independent_replay  # noqa: E402


def percent_change(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def main() -> None:
    rows = pd.read_csv(SOURCE / "metrics.csv")
    replay_rows = []
    for task in ("I_dark", "I_photo", "Q_terminal"):
        for seed in (42, 43, 44):
            directory = SOURCE / task / f"seed_{seed}"
            payload = json.loads(
                (directory / "selected_formula.json").read_text(encoding="utf-8")
            )
            table = pd.read_csv(directory / "test_predictions.csv")
            prediction, derivative = independent_replay.replay(payload, table)
            row = {
                "task": task,
                "seed": seed,
                "rows": len(table),
                "prediction_max_abs_error": float(
                    np.max(np.abs(prediction - table["selected"].to_numpy()))
                ),
            }
            if task == "Q_terminal":
                derivative_table = pd.read_csv(
                    directory / "derivative_test_predictions.csv"
                )
                _, derivative = independent_replay.replay(payload, derivative_table)
                derivative_reference = derivative_table["selected"].to_numpy()
            else:
                derivative_reference = table["selected_derivative"].to_numpy()
            row["derivative_max_abs_error"] = float(
                np.max(np.abs(derivative - derivative_reference))
            )
            replay_rows.append(row)
    pd.DataFrame(replay_rows).to_csv(
        HERE / "independent_replay_audit.csv", index=False
    )
    records = []
    for task, frame in rows.groupby("task", sort=False):
        record = {
            "task": task,
            "splits": len(frame),
            "baseline_count": int(frame["baseline_count"].iloc[0]),
            "selected_count": int(frame["selected_count"].iloc[0]),
        }
        for metric in [
            "rmse",
            "relative_p95",
            "group_rmse_p95",
            "conductance_nrmse",
            "conductance_relative_p95",
            "derivative_rmse",
            "derivative_p95",
        ]:
            old_column = f"baseline_test_{metric}"
            new_column = f"selected_test_{metric}"
            if old_column not in frame or frame[old_column].isna().all():
                continue
            old = float(frame[old_column].mean())
            new = float(frame[new_column].mean())
            record[f"baseline_{metric}"] = old
            record[f"selected_{metric}"] = new
            record[f"{metric}_change_pct"] = percent_change(new, old)
            record[f"{metric}_wins"] = int((frame[new_column] < frame[old_column]).sum())
        records.append(record)
    aggregate = pd.DataFrame(records)
    aggregate.to_csv(HERE / "aggregate.csv", index=False)
    configs = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))["selected_configs"]
    lines = [
        "# 物理锚定的暗电流、净光电流和端电荷折中方案",
        "",
        "本目录保存论文使用的 post hoc development experiment。候选结构和选择规则是在历史测试结果已可见的情况下形成，因此结果不能作为独立确认。",
        "",
        "## 固定方案",
        "",
        "三个分组划分共同选择一套超参数。点值 RMSE 不超过登记基线的 105%，完整响应尾部允许 110%，导数指标不超过 105%；满足全部划分后，先取最小系数数，再取平均验证比值较低者。",
        "",
        "| 任务 | 原复杂度 | 新复杂度 | 固定配置 |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in aggregate.to_dict(orient="records"):
        config = configs[row["task"]]
        lines.append(
            f"| {row['task']} | {row['baseline_count']} | {row['selected_count']} | "
            f"teacher={config['teacher_weight']}, derivative={config['derivative_weight']}, "
            f"ridge={config['ridge_alpha']} |"
        )
    lines.extend(
        [
            "",
            "## 三划分测试汇总",
            "",
            "| 任务 | RMSE 原→新 | RMSE 变化 | 完整响应 p95 RMSE 变化 | 导数指标变化 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate.to_dict(orient="records"):
        derivative = (
            row.get("conductance_nrmse_change_pct")
            if row["task"] != "Q_terminal"
            else row.get("derivative_rmse_change_pct")
        )
        lines.append(
            f"| {row['task']} | {row['baseline_rmse']:.6g} → {row['selected_rmse']:.6g} | "
            f"{row['rmse_change_pct']:.2f}% | {row['group_rmse_p95_change_pct']:.2f}% | "
            f"{derivative:.2f}% |"
        )
    lines.extend(
        [
            "",
            "暗电流与净光电流三个划分的点值、完整响应尾部和物理电导误差全部下降。端电荷的 Q 指标三个划分全部下降；独立 dQ/dV 在两个划分下降、一个划分上升，三划分平均仍下降。",
            "",
            "## 物理与部署边界",
            "",
            "- 电流公式使用 log10(I/A)，由参考偏压条件幅值和在参考偏压严格为零的 C2 偏压修正组成；五个器件参数均保留显式主效应。",
            "- 电荷公式严格满足 Q(0,p)=0，保留标称 V、V^2、V^3 与五个参数敏感项，并提供解析 dQ/dV。",
            "- 所有公式通过 20,000 个电流包络点或 39,360 个电荷点的有限性、方向/正电容和解析导数检查。",
            "- seed 42 公式是当前部署候选。其生成的 Verilog-A 已通过 Spectre 18.1 device-level 与 bounded circuit-level 验收；哈希绑定报告位于 `artifacts/results/physics_sparse_results_py36_20260921/`。这些结果仍不构成外推、新 TCAD 条件、测量或动态光端口验证。",
            "- 电流选择使用非零 BKAN 教师权重；电荷选择为 TCAD 与独立导数直接拟合。这里不能把整体收益单独归因于 BKAN。",
            "",
            "独立回放器 `replay.py` 仅依赖 NumPy/pandas；`independent_replay_audit.csv` 记录所有九个公式与冻结预测表的复现误差。",
            "",
            "## 复现",
            "",
            "在仓库根目录运行：",
            "",
            "```powershell",
            "python -B -X utf8 artifacts/experiments/physics_balanced_export_20260917/run_experiment.py --output artifacts/experiments/physics_balanced_export_20260917/development_rerun --seeds 42 43 44",
            "python -B -X utf8 artifacts/experiments/physics_balanced_export_20260917/select_consensus.py --source artifacts/experiments/physics_balanced_export_20260917/development_rerun --output artifacts/experiments/physics_balanced_export_20260917/consensus_rerun",
            "python -B -X utf8 artifacts/experiments/physics_balanced_export_20260917/test_experiment.py -v",
            "python -B -X utf8 artifacts/experiments/physics_balanced_export_20260917/summarize.py",
            "```",
            "",
            "`consensus_3split_compact/` 保存逐任务、逐划分的冻结公式、预测和验证前沿；`aggregate.csv` 是三划分汇总；`replay.py` 可单独读取一个公式 JSON 和输入 CSV。输出目录必须不存在，以避免覆盖既有证据。",
        ]
    )
    (HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
