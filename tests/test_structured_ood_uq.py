from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_structured_ood_uq.py"
SPEC = importlib.util.spec_from_file_location("structured_ood_uq", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DummySpec:
    name = "dummy"
    axis_col = "axis"
    input_candidates = ("axis", "condition")


def frame():
    rows = []
    for group in range(40):
        for axis in range(10):
            rows.append({"_curve_id": str(group), "axis": axis, "condition": group, "target": axis + group})
    return pd.DataFrame(rows)


def test_axis_tail_is_unseen_and_test_groups_are_independent():
    scenario = MODULE.Scenario("dummy", "axis_high", "axis", "high", 0.2)
    parts, audit = MODULE.structured_partitions(frame(), DummySpec(), scenario, 42)
    assert audit["row_overlap"] == 0
    assert parts["train"]["axis"].max() < parts["ood_test"]["axis"].min()
    fit_groups = set(pd.concat([parts["train"], parts["validation"], parts["calibration"]])["_curve_id"])
    assert fit_groups.isdisjoint(set(parts["id_test"]["_curve_id"]))
    assert set(parts["id_test"]["_curve_id"]) == set(parts["ood_test"]["_curve_id"])


def test_condition_tail_holds_out_complete_groups():
    scenario = MODULE.Scenario("dummy", "condition_high", "condition", "high", 0.2)
    parts, _ = MODULE.structured_partitions(frame(), DummySpec(), scenario, 43)
    assert parts["train"]["condition"].max() < parts["ood_test"]["condition"].min()
    fit = set(pd.concat([parts["train"], parts["validation"], parts["calibration"], parts["id_test"]])["_curve_id"])
    assert fit.isdisjoint(set(parts["ood_test"]["_curve_id"]))
    assert np.isclose(len(parts["ood_test"]) / len(frame()), 0.2)
