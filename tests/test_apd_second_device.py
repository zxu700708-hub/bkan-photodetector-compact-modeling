from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
if str(BKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BKAN_ROOT))

from device_modeling.apd.data import DEFAULT_APD_ROOT, load_apd_task
from device_modeling.photodetector import task_config as shared


def _spec(task):
    return shared.TaskSpec(
        key=task.key,
        name=task.name,
        result_subdir=task.key,
        target_col=task.target_col,
        axis_col=task.axis_col,
        input_candidates=task.input_cols,
        use_log_transform=task.use_log_transform,
        y_bounds=task.y_bounds,
    )


def test_apd_primary_task_shapes_and_targets():
    expected_rows = {
        "dark_current": 78 * 79,
        "photo_current": 156 * 80,
        "net_photocurrent": 156 * 80,
        "multiplication_gain": 156 * 80,
    }
    for key, count in expected_rows.items():
        frame, task = load_apd_task(key, DEFAULT_APD_ROOT)
        assert len(frame) == count
        assert frame["structure_group_id"].nunique() == 78
        assert frame[task.target_col].notna().all()
        assert np.isfinite(frame[task.target_col]).all()
        assert (frame[task.target_col] > 0.0).all()


def test_apd_fom_task_specific_filtering():
    threshold, _ = load_apd_task("gain_threshold_voltage", DEFAULT_APD_ROOT)
    breakdown, _ = load_apd_task("breakdown_voltage", DEFAULT_APD_ROOT)
    assert len(threshold) == 155
    assert len(breakdown) == 155
    assert "apd_20260818_00154" not in set(threshold["sample_id"])
    assert "apd_20260818_00028" not in set(breakdown["sample_id"])
    assert threshold["structure_group_id"].nunique() == 78
    assert breakdown["structure_group_id"].nunique() == 78


def test_apd_split_is_structure_disjoint():
    frame, task = load_apd_task("photo_current", DEFAULT_APD_ROOT)
    partitions = shared.split_task_dataframe(
        frame, _spec(task), seed=42, fractions=(0.65, 0.10, 0.15, 0.10)
    )
    group_sets = [set(part["structure_group_id"]) for part in partitions]
    assert [len(groups) for groups in group_sets] == [51, 8, 11, 8]
    assert len(set.union(*group_sets)) == 78
    for left in range(len(group_sets)):
        for right in range(left):
            assert group_sets[left].isdisjoint(group_sets[right])


def test_dark_current_uses_one_canonical_curve_per_structure():
    frame, _ = load_apd_task("dark_current", DEFAULT_APD_ROOT)
    assert frame.groupby("structure_group_id").size().eq(79).all()
    assert frame["bias_v"].lt(0.0).all()
