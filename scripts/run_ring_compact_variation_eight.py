"""Run a model comparison on the compact process/temperature ring data.

Only the development partition is visible to this runner.  The separately
seeded frozen-confirmation structures remain quarantined until a later,
explicit confirmation run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_ring_direct_device_metrics as base  # noqa: E402
from compact_framework_baselines import (  # noqa: E402
    fit_predict_curve_lut,
    fit_predict_semiempirical_compact,
)

ORIGINAL_MODELS = tuple(base.MODELS)
ORIGINAL_TRAIN_ONE = base.train_one


DEFAULT_DATA_ROOT = Path(r"<FROZEN_RING_ROOT>/hifi_ring_80\results\ring_compact_variation_v1")
DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "ring_compact_variation_eight"
INPUT_COLUMNS = (
    "actual_core_width_nm",
    "actual_coupling_gap_nm",
    "actual_ring_radius_um",
    "actual_core_thickness_nm",
    "delta_material_index",
    "bias_v",
    "temperature_c",
)
MODELS = (
    "bkan",
    "dkan",
    "mlp_l",
    "spline_ridge",
    "gmls",
    "autopinn",
    "curve_lut",
    "semiempirical",
)
MODEL_LABELS = {
    **base.MODEL_LABELS,
    "curve_lut": "Curve-LUT-PCHIP",
    "semiempirical": "Ring SemiEmpirical-CM",
}
TASKS = {
    key: replace(task, input_cols=INPUT_COLUMNS)
    for key, task in base.TASKS.items()
}
TASKS["resonance_shift"] = replace(
    TASKS["resonance_shift"],
    y_bounds=(-10.0, 2500.0),
)


def load_dataset(data_root: Path):
    target_path = data_root / "training_tables" / "ring_compact_targets.csv"
    manifest_path = data_root / "dataset_manifest.json"
    confirmation_path = data_root / "frozen_confirmation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != "ring_compact_process_temperature_v1":
        raise RuntimeError("Unexpected compact ring dataset version")
    if manifest.get("claims", {}).get("downstream_model_performance_used_for_case_selection"):
        raise RuntimeError("Dataset manifest reports performance-based case selection")
    if not confirmation.get("frozen"):
        raise RuntimeError("Confirmation partition is not marked frozen")
    relative = "training_tables/ring_compact_targets.csv"
    if base.file_sha256(target_path) != manifest["files_sha256"][relative]:
        raise RuntimeError("Compact target table hash mismatch")
    frame = pd.read_csv(target_path)
    required = {
        "nominal_structure_id",
        "process_variant_id",
        "process_variant_index",
        "dataset_role",
        "split",
        *INPUT_COLUMNS,
        *(task.target_col for task in TASKS.values()),
    }
    if not required.issubset(frame.columns):
        raise KeyError(f"Missing compact ring columns: {sorted(required - set(frame.columns))}")
    if len(frame) != 18_000 or frame["nominal_structure_id"].nunique() != 400:
        raise RuntimeError("Unexpected complete compact ring dataset size")
    frozen_rows = frame[frame["dataset_role"].eq("frozen_confirmation")]
    if len(frozen_rows) != 3_600 or frozen_rows["nominal_structure_id"].nunique() != 80:
        raise RuntimeError("Frozen confirmation partition is incomplete")
    development = frame[frame["dataset_role"].eq("development")].copy()
    if len(development) != 14_400 or development["nominal_structure_id"].nunique() != 320:
        raise RuntimeError("Development partition is incomplete")
    development["structure_id"] = development["nominal_structure_id"].astype(str)
    development["split"] = development["split"].replace({"internal_test": "test"})
    numeric = [*INPUT_COLUMNS, *(task.target_col for task in TASKS.values())]
    if not np.isfinite(development[list(dict.fromkeys(numeric))].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite development input or target")
    group_splits = development[["structure_id", "split"]].drop_duplicates()
    if group_splits["structure_id"].duplicated().any():
        raise RuntimeError("A nominal structure crosses development splits")
    expected = {"train": 208, "validation": 32, "calibration": 48, "test": 32}
    if group_splits["split"].value_counts().to_dict() != expected:
        raise RuntimeError("Unexpected compact development split counts")
    split_rows = pd.concat(
        [group_splits.assign(seed=seed) for seed in range(42, 52)], ignore_index=True
    )
    provenance = {
        "metric_table": str(target_path),
        "metric_table_sha256": base.file_sha256(target_path),
        "source_manifest_sha256": base.file_sha256(manifest_path),
        "frozen_confirmation_manifest_sha256": base.file_sha256(confirmation_path),
        "visible_development_structures": 320,
        "quarantined_confirmation_structures": 80,
    }
    return development, split_rows, provenance


def split_frame(frame: pd.DataFrame, splits: pd.DataFrame, seed: int):
    assignment = splits[splits["seed"].eq(seed)].copy()
    if len(assignment) != 320 or assignment["structure_id"].nunique() != 320:
        raise RuntimeError(f"Incomplete preset split for training seed {seed}")
    work = frame.drop(columns="split").merge(
        assignment[["seed", "structure_id", "split"]],
        on="structure_id",
        how="left",
        validate="many_to_one",
    )
    order = ["structure_id", "process_variant_index", "temperature_c", "bias_v"]
    parts = tuple(
        work[work["split"].eq(name)].sort_values(order).reset_index(drop=True)
        for name in ("train", "validation", "calibration", "test")
    )
    expected_groups = (208, 32, 48, 32)
    if tuple(part["structure_id"].nunique() for part in parts) != expected_groups:
        raise RuntimeError("Preset development split changed")
    group_sets = [set(part["structure_id"]) for part in parts]
    if any(group_sets[i] & group_sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise RuntimeError("Nominal-structure leakage")
    return (*parts, assignment)


def train_one(model, train, validation, test, task, seed, device, args, model_dir):
    if model in ORIGINAL_MODELS:
        return ORIGINAL_TRAIN_ONE(
            model, train, validation, test, task, seed, device, args, model_dir
        )
    spec = base.baseline_spec(task)
    if model == "curve_lut":
        prediction, info = fit_predict_curve_lut(
            train, validation, test, spec, axis_col=task.axis_col
        )
        return prediction, None, info
    if model == "semiempirical":
        prediction, info = fit_predict_semiempirical_compact(
            train, validation, test, spec, axis_col=task.axis_col
        )
        return prediction, None, info
    raise ValueError(model)


def configure_base_module() -> None:
    base.__doc__ = __doc__
    base.DEFAULT_DATA_ROOT = DEFAULT_DATA_ROOT
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.INPUT_COLUMNS = INPUT_COLUMNS
    base.MODELS = MODELS
    base.MODEL_LABELS = MODEL_LABELS
    base.TASKS = TASKS
    base.load_dataset = load_dataset
    base.split_frame = split_frame
    base.train_one = train_one


def correct_output_metadata(output: Path, args) -> None:
    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "protocol": f"ring_compact_variation_{len(MODELS)}model_development_v1",
            "data_kind": "high_fidelity_calibrated_reduced_order_synthetic",
            "visible_partition": "development_only",
            "frozen_confirmation_opened": False,
            "independent_unit": "nominal_structure_id",
            "split_counts": {
                "train": 208,
                "validation": 32,
                "calibration": 48,
                "test": 32,
            },
            "input_columns": list(INPUT_COLUMNS),
            "claim_boundary": (
                "Exploratory development-partition comparison on a calibrated reduced-order synthetic "
                "dataset; not a frozen-confirmation result, new four-solver run, or measurement validation."
            ),
        }
    )
    config.setdefault("training", {}).update(
        {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "kl_weight": args.kl_weight,
            "mc_samples": args.mc_samples,
            "train_mc_samples": args.train_mc_samples,
            "validation_mc_samples": args.validation_mc_samples,
            "early_stopping_patience": args.early_stopping_patience,
        }
    )
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = pd.read_csv(output / "metrics_by_seed.csv")
    primary_columns = {
        "resonance_shift": "nonzero_bias_rmse_raw",
        "quality_factor": "median_ape_percent",
        "drop_extinction": "mae_raw",
        "through_insertion": "median_ape_percent",
    }
    display_names = {
        "resonance_shift": "Resonance-shift RMSE (pm)",
        "quality_factor": "Loaded-Q MdAPE (%)",
        "drop_extinction": "Drop extinction MAE (dB)",
        "through_insertion": "Through IL MdAPE (%)",
    }
    rows = []
    for model in MODELS:
        row = {"model": model, "model_label": MODEL_LABELS[model]}
        for task_key, metric_column in primary_columns.items():
            selected = metrics[
                metrics["task_key"].eq(task_key) & metrics["model"].eq(model)
            ]
            if len(selected) != 1:
                raise RuntimeError(f"Expected one result for {task_key}/{model}")
            row[task_key] = float(selected.iloc[0][metric_column])
        rows.append(row)
    comparison = pd.DataFrame(rows)
    for task_key in primary_columns:
        comparison[f"{task_key}_rank"] = comparison[task_key].rank(
            method="min", ascending=True
        ).astype(int)
    comparison["average_rank"] = comparison[
        [f"{task_key}_rank" for task_key in primary_columns]
    ].mean(axis=1)
    comparison.to_csv(output / "primary_comparison_table.csv", index=False)

    def formatted(value: float, rank: int) -> str:
        text = f"{value:.6g}"
        if rank == 1:
            return f"**{text}**"
        if rank == 2:
            return f"*{text}*"
        return text

    lines = [
        f"# Compact process/temperature microring: {len(MODELS)}-model development comparison",
        "",
        "This exploratory run uses only the 320-structure development partition. The 80 independently seeded frozen-confirmation structures remain unopened. The split unit is the nominal structure; all process, bias, and temperature rows from a structure stay together.",
        "",
        "Lower is better. Bold marks the best development-test value and italics the second-best value.",
        "",
        "| Model | " + " | ".join(display_names.values()) + " | Average rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        cells = []
        for task_key in primary_columns:
            cells.append(
                formatted(
                    float(getattr(row, task_key)),
                    int(getattr(row, f"{task_key}_rank")),
                )
            )
        lines.append(
            f"| {row.model_label} | " + " | ".join(cells) + f" | {row.average_rank:.2f} |"
        )
    lines.extend(
        [
            "",
            "These are single preset-development-split results, not repeated-split inference or frozen-confirmation results.",
        ]
    )
    (output / "development_comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    configure_base_module()
    args = base.resolve_args(base.parse_args())
    base.run(args)
    correct_output_metadata(args.output, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
