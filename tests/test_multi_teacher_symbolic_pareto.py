from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_multi_teacher_symbolic_pareto as pareto  # noqa: E402


def test_mark_pareto_groups_by_task_and_minimizes_both_axes() -> None:
    frame = pd.DataFrame(
        {
            "task": ["a", "a", "a", "a", "b"],
            "formula_to_tcad_rmse": [3.0, 2.0, 1.0, 2.5, 9.0],
            "terms": [1, 2, 4, 3, 100],
        }
    )
    flags = pareto.mark_pareto(frame)
    assert flags.tolist() == [True, True, True, False, True]


def test_fixed_budget_export_selects_validation_alpha_and_replays_json(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(80, 2))
    x_validation = rng.normal(size=(30, 2))
    y_train = 1.0 + 2.0 * x_train[:, 0] - 0.5 * x_train[:, 1] ** 2
    y_validation = 1.0 + 2.0 * x_validation[:, 0] - 0.5 * x_validation[:, 1] ** 2

    model, payload, evaluator, metadata = pareto._select_and_export(
        "polynomial",
        2,
        (1e-8, 1e-4, 1e-2),
        x_train,
        y_train,
        x_validation,
        y_validation,
        ("x0", "x1"),
        tmp_path,
    )

    expected = model.predict(x_validation).reshape(-1)
    replay = evaluator(payload, x_validation)
    assert np.max(np.abs(expected - replay)) < 1e-10
    assert metadata["terms"] == 6
    assert metadata["selected_alpha"] in {1e-8, 1e-4, 1e-2}
    assert (tmp_path / "formula.json").is_file()
    assert (tmp_path / "formula.va").is_file()


def test_metrics_reports_rmse_and_r2() -> None:
    result = pareto._metrics(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 3.0]),
        "audit",
    )
    assert np.isclose(result["audit_rmse"], np.sqrt(1.0 / 3.0))
    assert np.isclose(result["audit_r2"], 0.5)


def test_replay_audit_applies_registered_capacitance_scale(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions/seed_42/Capacitance/dkan"
    prediction_dir.mkdir(parents=True)
    legacy_actual = np.array([2.0e-17, 3.0e-17])
    legacy_prediction = np.array([2.1e-17, 2.9e-17])
    pd.DataFrame(
        {
            "actual_model_space": legacy_actual,
            "prediction_model_space": legacy_prediction,
        }
    ).to_csv(prediction_dir / "test_predictions.csv", index=False)
    args = SimpleNamespace(
        seed=42,
        matched_root=tmp_path,
        allow_teacher_replay_mismatch=False,
    )
    scaled_actual = legacy_actual * 1000.0
    scaled_prediction = legacy_prediction * 1000.0
    result = pareto._replay_audit(
        "Capacitance",
        "dkan",
        pd.DataFrame(index=range(2)),
        scaled_actual,
        scaled_prediction,
        args,
    )
    assert result["replay_status"] == "pass"
    assert result["frozen_scale_applied"] == 1000.0
    assert result["replay_max_abs_error"] == 0.0
