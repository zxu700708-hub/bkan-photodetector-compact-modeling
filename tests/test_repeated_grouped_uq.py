import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TEST_OUTPUT = ROOT / "tmp" / "test_repeated_grouped_uq"
TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
BKAN_ROOT = ROOT / "bkan"
SCRIPTS = ROOT / "scripts"
for path in (BKAN_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from device_modeling.photodetector.task_config import (  # noqa: E402
    TASKS,
    split_task_dataframe,
    task_group_labels,
)
from run_repeated_grouped_uq import (  # noqa: E402
    aggregate_metrics,
    wilson_interval,
)
from device_modeling.photodetector.bayes_modeler_impl import (  # noqa: E402
    BayesKANDeviceModeler,
)
from device_modeling.photodetector.run_research import (  # noqa: E402
    conformal_group_columns,
)


def synthetic_dark_curves(n_curves=160, points_per_curve=4):
    rows = []
    voltages = np.linspace(-3.0, -0.1, points_per_curve)
    for curve_id in range(n_curves):
        for voltage in voltages:
            rows.append(
                {
                    "_curve_id": f"curve_{curve_id:03d}",
                    "dark_voltage": voltage,
                    "trap_assisted_recomb_A": 0.8e-9
                    + (curve_id % 8) * 0.1e-9,
                    "active_layer_length": 10.0 + (curve_id % 10) * 10.0,
                    "simulation_temperature": 280.0
                    + (curve_id % 5) * 10.0,
                    "dark_current": 1.0e-8
                    * (1.0 + curve_id / n_curves)
                    * (1.0 + abs(voltage)),
                }
            )
    return pd.DataFrame(rows)


def synthetic_capacitance_clusters(scale=1.0, n_devices=24):
    rows = []
    for device in range(n_devices):
        device_value = float(device) / max(n_devices - 1, 1)
        for bias in (-3.0, -1.0):
            for frequency in (0.1, 1.0, 10.0):
                target = scale * (
                    2.0
                    + 0.1 * bias
                    + 0.01 * np.log10(frequency)
                    + 0.05 * device_value
                )
                rows.append(
                    {
                        "_curve_id": f"capacitance_sample_{device}",
                        "bias_v": bias,
                        "log_frequency_ghz": np.log10(frequency),
                        "device_value": device_value,
                        "capacitance_F": target,
                    }
                )
    return pd.DataFrame(rows)


def calibrated_dummy_modeler(frame, scale, output_dir):
    modeler = BayesKANDeviceModeler(
        task_name="Capacitance",
        results_dir=output_dir,
        device="cpu",
    )
    modeler.model = object()
    modeler.input_cols = ["bias_v", "log_frequency_ghz", "device_value"]
    modeler.output_col = "capacitance_F"
    modeler.use_log_transform = False
    modeler.scaler_y = StandardScaler().fit(frame[["capacitance_F"]])

    def predict_with_uncertainty(values):
        values = np.asarray(values, dtype=np.float64)
        target = scale * (
            2.0
            + 0.1 * values[:, 0]
            + 0.01 * values[:, 1]
            + 0.05 * values[:, 2]
        )
        residual = scale * (0.01 + 0.02 * values[:, 2])
        std = np.full(len(values), scale * 0.01, dtype=np.float64)
        return {
            "log_mean": target + residual,
            "log_std": std,
        }

    modeler.predict_with_uncertainty = predict_with_uncertainty
    modeler.calibrate_uncertainty(
        df_calib=frame,
        target_picp=95.0,
        min_z=None,
        calibration_unit="curve",
        group_cols=["_curve_id"],
    )
    return modeler


class RepeatedGroupedUQTests(unittest.TestCase):
    def test_65_10_15_10_split_is_curve_disjoint(self):
        spec = TASKS["dark_current"]
        frame = synthetic_dark_curves()
        partitions = split_task_dataframe(
            frame,
            spec,
            seed=42,
            fractions=(0.65, 0.10, 0.15, 0.10),
        )
        curve_sets = [
            set(task_group_labels(partition, spec))
            for partition in partitions
        ]

        self.assertEqual([len(values) for values in curve_sets], [104, 16, 24, 16])
        self.assertEqual(sum(map(len, partitions)), len(frame))
        for left in range(len(curve_sets)):
            for right in range(left):
                self.assertFalse(curve_sets[left] & curve_sets[right])

    def test_default_split_remains_backward_compatible(self):
        spec = TASKS["dark_current"]
        partitions = split_task_dataframe(
            synthetic_dark_curves(),
            spec,
            seed=42,
        )
        counts = [
            task_group_labels(partition, spec).nunique()
            for partition in partitions
        ]
        self.assertEqual(counts, [112, 16, 16, 16])

    def test_invalid_split_fractions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            split_task_dataframe(
                synthetic_dark_curves(),
                TASKS["dark_current"],
                seed=42,
                fractions=(0.65, 0.10, 0.10, 0.10),
            )

    def test_wilson_interval_reflects_small_curve_count(self):
        low, high = wilson_interval(87.5, 16)
        self.assertLess(low, 87.5)
        self.assertGreater(high, 87.5)
        self.assertGreater(high - low, 20.0)

    def test_conformal_quantile_selects_the_exact_finite_sample_rank(self):
        scores = np.arange(1.0, 25.0)
        # ceil((24 + 1) * 0.90) = 23, so the exact order statistic is 23.
        self.assertEqual(
            BayesKANDeviceModeler._conformal_quantile(scores, 90.0),
            23.0,
        )

    def test_aggregate_metrics_reports_across_seed_interval(self):
        metrics = pd.DataFrame(
            {
                "seed": [42, 43, 44],
                "task": ["AC_Response"] * 3,
                "model": ["bayesian_kan_vi"] * 3,
                "rmse_model_space": [0.08, 0.09, 0.10],
                "calibrated_curve_coverage_95": [87.5, 93.75, 100.0],
                "calibration_z": [1.0, 1.2, 1.4],
            }
        )
        summary = aggregate_metrics(metrics).iloc[0]
        self.assertEqual(int(summary["n_seeds"]), 3)
        self.assertAlmostEqual(
            float(summary["calibrated_curve_coverage_95_mean"]),
            93.75,
        )
        self.assertLess(
            float(summary["calibrated_curve_coverage_95_ci95_low"]),
            93.75,
        )
        self.assertGreater(
            float(summary["calibrated_curve_coverage_95_ci95_high"]),
            93.75,
        )

    def test_capacitance_conformal_uses_one_score_per_device(self):
        frame = synthetic_capacitance_clusters()
        self.assertEqual(
            conformal_group_columns(frame, TASKS["capacitance"]),
            ["_curve_id"],
        )
        modeler = calibrated_dummy_modeler(frame, 1.0, TEST_OUTPUT)
        self.assertEqual(modeler._calib_score_count, 24)
        self.assertEqual(modeler._calib_group_cols, ["_curve_id"])

    def test_conformal_scores_are_target_scale_invariant(self):
        frame_unit = synthetic_capacitance_clusters(scale=1.0)
        frame_femto = synthetic_capacitance_clusters(scale=1.0e-18)
        unit_modeler = calibrated_dummy_modeler(
            frame_unit,
            1.0,
            TEST_OUTPUT,
        )
        femto_modeler = calibrated_dummy_modeler(
            frame_femto,
            1.0e-18,
            TEST_OUTPUT,
        )
        self.assertAlmostEqual(
            unit_modeler._calib_raw_quantile,
            femto_modeler._calib_raw_quantile,
            places=4,
        )
        self.assertLess(femto_modeler._calib_sigma_floor, 1.0e-12)

    def test_duplicating_points_does_not_inflate_cluster_count(self):
        frame = synthetic_capacitance_clusters()
        duplicated = pd.concat([frame, frame], ignore_index=True)
        original = calibrated_dummy_modeler(frame, 1.0, TEST_OUTPUT)
        repeated = calibrated_dummy_modeler(duplicated, 1.0, TEST_OUTPUT)
        self.assertEqual(original._calib_score_count, 24)
        self.assertEqual(repeated._calib_score_count, 24)
        self.assertAlmostEqual(
            original._calib_raw_quantile,
            repeated._calib_raw_quantile,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
