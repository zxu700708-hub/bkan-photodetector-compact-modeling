from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_multi_teacher_symbolic_pareto_10split as audit  # noqa: E402


def test_corrected_paired_statistics_uses_teacher_minus_direct_sign() -> None:
    result = audit.corrected_paired_statistics(
        np.array([-3.0, -2.0, -1.0, -2.0]), test_fraction=0.10, train_fraction=0.65
    )
    assert result["mean_delta_teacher_minus_direct"] == -2.0
    assert result["corrected_ci_high"] < 0.0
    assert result["teacher_wins"] == 4
    assert result["teacher_losses"] == 0


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    raw = pd.Series([0.04, 0.001, 0.02])
    adjusted = audit.holm_adjust(raw)
    order = np.argsort(raw.to_numpy())
    assert np.all(np.diff(adjusted.to_numpy()[order]) >= 0.0)
    assert np.all(adjusted.to_numpy() >= raw.to_numpy())


def test_photo_task_routes_to_corrected_roots(tmp_path: Path) -> None:
    args = SimpleNamespace(
        output=tmp_path / "out",
        data=tmp_path / "main.csv",
        net_photocurrent_data=tmp_path / "net.csv",
        capacitance_data=tmp_path / "cap.csv",
        matched_root=tmp_path / "matched",
        bkan_root=tmp_path / "bkan",
        capacitance_bkan_root=tmp_path / "cap_bkan",
        net_root=tmp_path / "net_root",
        split_fractions=(0.65, 0.10, 0.15, 0.10),
        poly_degrees=(1, 2, 3),
        spline_knots=(4, 6, 8),
        ridge_alphas=(1e-8, 1e-6),
        dense_points=100,
        latency_repeats=1,
        mlp_epochs=1,
        kan_steps=1,
        allow_teacher_replay_mismatch=False,
    )
    routed = audit._single_args(args, 42, "I_photo")
    assert routed.data == args.net_photocurrent_data
    assert routed.matched_root == args.net_root / "matched_grouped_comparison"
    assert routed.bkan_root == args.net_root / "uq_repeated_grouped"
    assert routed.output == args.output / "seed_42" / "I_photo"
