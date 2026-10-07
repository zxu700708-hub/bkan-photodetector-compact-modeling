from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import replay as independent_replay


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("physics_balanced", HERE / "run_experiment.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class PhysicsBalancedTests(unittest.TestCase):
    def test_grouped_derivative_matches_quadratic(self):
        axis = np.tile(np.linspace(-3.0, 0.0, 9), 2)
        labels = np.repeat(["a", "b"], 9)
        values = axis**2 + np.repeat([0.0, 2.0], 9)
        derivative = module.grouped_axis_derivative(values, axis, labels)
        np.testing.assert_allclose(derivative, 2.0 * axis, atol=1e-12)

    def test_compensated_path_preserves_mandatory_columns(self):
        rng = np.random.default_rng(7)
        design = rng.normal(size=(200, 12))
        target = design @ rng.normal(size=12)
        path = module.base.elimination_path(
            design, target, 1e-8, [4, 6, 8, 12], mandatory=(0, 3, 7)
        )
        for coefficient in path.values():
            self.assertTrue(all(coefficient[index] != 0.0 for index in (0, 3, 7)))

    def test_saved_current_formula_is_reference_anchored(self):
        root = HERE / "consensus_3split_compact" / "I_photo" / "seed_42"
        if not (root / "selected_formula.json").is_file():
            self.skipTest("pilot output has not been generated")
        payload = json.loads((root / "selected_formula.json").read_text(encoding="utf-8"))
        if payload["schema"] == "unchanged_baseline":
            self.skipTest("pilot selected baseline fallback")
        table = pd.read_csv(root / "test_predictions.csv")
        reference = table[payload["input_order"]].copy()
        reference[payload["input_order"][0]] = payload["current_formula"]["reference_voltage"]
        _, derivative = module.current_formula_eval(payload, reference)
        prediction, _ = module.current_formula_eval(payload, reference)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.isfinite(derivative).all())
        formula = payload["current_formula"]
        active = set(formula["active_columns"])
        self.assertTrue(set(payload["selection"]["mandatory_columns"]).issubset(active))

    def test_saved_charge_formula_preserves_reference_and_coverage(self):
        root = HERE / "consensus_3split_compact" / "Q_terminal" / "seed_42"
        if not (root / "selected_formula.json").is_file():
            self.skipTest("pilot output has not been generated")
        payload = json.loads((root / "selected_formula.json").read_text(encoding="utf-8"))
        if payload["schema"] == "unchanged_baseline":
            self.skipTest("pilot selected baseline fallback")
        table = pd.read_csv(root / "test_predictions.csv")
        zero = table[payload["input_order"]].copy()
        zero["bias_v"] = 0.0
        charge, derivative = module.base.charge_sparse_eval(payload, zero)
        np.testing.assert_allclose(charge, 0.0, atol=1e-30)
        self.assertTrue(np.all(derivative > 0.0))
        active = {term["index"] for term in payload["terms"]}
        self.assertTrue(set(payload["selection"]["mandatory_columns"]).issubset(active))

    def test_all_consensus_exports_replay_frozen_predictions(self):
        root = HERE / "consensus_3split_compact"
        if not root.is_dir():
            self.skipTest("consensus output has not been generated")
        for task in ("I_dark", "I_photo", "Q_terminal"):
            for seed in (42, 43, 44):
                directory = root / task / f"seed_{seed}"
                payload = json.loads(
                    (directory / "selected_formula.json").read_text(encoding="utf-8")
                )
                table = pd.read_csv(directory / "test_predictions.csv")
                prediction, derivative = independent_replay.replay(payload, table)
                np.testing.assert_allclose(prediction, table["selected"], rtol=0.0, atol=2e-14)
                if task == "Q_terminal":
                    derivative_table = pd.read_csv(
                        directory / "derivative_test_predictions.csv"
                    )
                    _, derivative = independent_replay.replay(payload, derivative_table)
                    np.testing.assert_allclose(
                        derivative, derivative_table["selected"], rtol=0.0, atol=2e-27
                    )
                else:
                    np.testing.assert_allclose(
                        derivative,
                        table["selected_derivative"],
                        rtol=0.0,
                        atol=2e-9,
                    )


if __name__ == "__main__":
    unittest.main()
