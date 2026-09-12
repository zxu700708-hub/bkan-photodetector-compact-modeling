"""Checks for the method-level compact-model comparison baselines."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compact_framework_baselines import (  # noqa: E402
    GeneralizedMovingLeastSquaresRegressor,
    fit_predict_curve_lut,
    fit_predict_semiempirical_compact,
    train_autopinn_adapted,
)


class CompactFrameworkBaselineTests(unittest.TestCase):
    def test_curve_lut_interpolates_complete_curves(self):
        rows = []
        for condition in (0.0, 1.0):
            for axis in np.linspace(-1.0, 1.0, 9):
                rows.append(
                    {"axis": axis, "condition": condition, "target": 3.0 * axis + 2.0 * condition}
                )
        train = pd.DataFrame(rows)
        validation = pd.DataFrame(
            {
                "axis": np.linspace(-0.9, 0.9, 7),
                "condition": 0.25,
            }
        )
        validation["target"] = 3.0 * validation["axis"] + 0.5
        test = pd.DataFrame(
            {
                "axis": np.linspace(-0.8, 0.8, 5),
                "condition": 0.5,
            }
        )
        test["target"] = 3.0 * test["axis"] + 1.0
        spec = SimpleNamespace(
            name="synthetic_curve",
            input_cols=("axis", "condition"),
            target_col="target",
            use_log_transform=False,
        )
        prediction, info = fit_predict_curve_lut(
            train, validation, test, spec, axis_col="axis"
        )
        np.testing.assert_allclose(prediction, test["target"], rtol=0.0, atol=1e-12)
        self.assertEqual(info["method_variant"], "curve_object_pchip_lut_condition_idw")
        self.assertEqual(info["stored_curves"], 2)

    def test_semiempirical_compact_is_reproducible_and_finite(self):
        def make_frame(conditions):
            rows = []
            for condition in conditions:
                for axis in np.linspace(-8.0, 0.0, 17):
                    reverse = -axis / 8.0
                    rows.append(
                        {
                            "axis": axis,
                            "condition": condition,
                            "target": 0.3 + 0.2 * condition + 0.7 * reverse + 0.5 * reverse**2,
                        }
                    )
            return pd.DataFrame(rows)

        train = make_frame((1.0, 2.0, 4.0, 5.0))
        validation = make_frame((3.0,))
        test = make_frame((3.5,))
        spec = SimpleNamespace(
            name="synthetic_compact",
            input_cols=("axis", "condition"),
            target_col="target",
            use_log_transform=False,
        )
        first, first_info = fit_predict_semiempirical_compact(
            train, validation, test, spec, axis_col="axis"
        )
        second, second_info = fit_predict_semiempirical_compact(
            train, validation, test, spec, axis_col="axis"
        )
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
        self.assertEqual(first_info["selected_alpha"], second_info["selected_alpha"])
        self.assertEqual(
            first_info["method_variant"],
            "ridge_calibrated_semiempirical_compact_feature_library",
        )

    def test_gmls_recovers_a_quadratic_surface(self):
        axis = np.linspace(-1.0, 1.0, 13)
        x1, x2 = np.meshgrid(axis, axis)
        x = np.column_stack((x1.ravel(), x2.ravel()))
        y = 0.7 + 0.4 * x[:, 0] - 0.2 * x[:, 1] + 0.3 * x[:, 0] * x[:, 1]
        query = np.array([[-0.45, 0.15], [0.25, -0.35], [0.6, 0.5]])
        expected = (
            0.7 + 0.4 * query[:, 0] - 0.2 * query[:, 1]
            + 0.3 * query[:, 0] * query[:, 1]
        )
        model = GeneralizedMovingLeastSquaresRegressor(min_neighbors=30)
        model.fit(x, y)
        np.testing.assert_allclose(model.predict(query), expected, rtol=2e-5, atol=2e-5)

    def test_autopinn_adaptation_is_reproducible_and_finite(self):
        axis = np.linspace(-1.0, 1.0, 60)
        frame = pd.DataFrame(
            {
                "axis": axis,
                "condition": np.cos(axis),
                "target": 1.2 - 0.8 * axis + 0.05 * np.sin(2.0 * axis),
            }
        )
        train = frame.iloc[:36].reset_index(drop=True)
        validation = frame.iloc[36:48].reset_index(drop=True)
        test = frame.iloc[48:].reset_index(drop=True)
        spec = SimpleNamespace(
            input_cols=("axis", "condition"),
            target_col="target",
            use_log_transform=False,
        )
        kwargs = dict(
            axis_col="axis",
            monotonic_sign=-1,
            seed=17,
            device=torch.device("cpu"),
            epochs=40,
            candidates=((8, 8), (12, 8)),
            collocation_size=16,
            patience=3,
        )
        first, first_info = train_autopinn_adapted(
            train, validation, test, spec, **kwargs
        )
        second, second_info = train_autopinn_adapted(
            train, validation, test, spec, **kwargs
        )
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
        self.assertEqual(first_info["selected_hidden"], second_info["selected_hidden"])
        self.assertEqual(first_info["method_variant"],
                         "autopinn_adapted_smooth_monotone_architecture_search")


if __name__ == "__main__":
    unittest.main()
