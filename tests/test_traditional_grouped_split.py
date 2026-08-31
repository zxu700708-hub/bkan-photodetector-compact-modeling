import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_EXPERIMENTS = ROOT / "bkan" / "simulation" / "paper_experiments"
if str(PAPER_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(PAPER_EXPERIMENTS))

from p0_3_traditional_comparison import (  # noqa: E402
    DarkCurrentParams,
    TASKS,
    split_audit_row,
    split_task_data,
    task_group_columns,
)


def synthetic_photo_curves(n_curves=20, points_per_curve=4):
    rows = []
    voltages = np.linspace(-3.0, -0.1, points_per_curve)
    for curve_id in range(n_curves):
        for voltage in voltages:
            rows.append(
                {
                    "light_voltage": voltage,
                    "trap_assisted_recomb_A": 0.8e-9
                    + (curve_id % 5) * 0.1e-9,
                    "ge_sio2_recomb_velocity": 1000.0 + curve_id * 10.0,
                    "ge_si_recomb_velocity": 2000.0 + curve_id * 11.0,
                    "active_layer_length": 10.0 + (curve_id % 10) * 10.0,
                    "simulation_temperature": 280.0
                    + (curve_id % 4) * 10.0,
                    "light_current": 1.0e-3
                    * (1.0 + curve_id / n_curves)
                    * (1.0 + 0.01 * abs(voltage)),
                }
            )
    return pd.DataFrame(rows)


class TraditionalGroupedSplitTests(unittest.TestCase):
    def test_dark_physics_full_declares_nine_free_and_two_fixed_parameters(self):
        manifest = DarkCurrentParams.manifest()

        self.assertEqual(manifest["free_parameter_count"], 9)
        self.assertEqual(len(DarkCurrentParams.names), 9)
        self.assertEqual(len(DarkCurrentParams.bounds_natural), 9)
        self.assertEqual(DarkCurrentParams.initial_guess().shape, (9,))
        self.assertEqual(
            manifest["fixed_parameters"],
            {"E_a_diff": 0.66, "gamma_surf": 1.0},
        )
        self.assertIn("E_a_gr", DarkCurrentParams.names)
        self.assertNotIn("E_a_diff", DarkCurrentParams.names)

        unpacked = DarkCurrentParams.unpack(DarkCurrentParams.initial_guess())
        self.assertEqual(len(unpacked), 11)
        self.assertEqual(unpacked[-2:], (0.66, 1.0))

    def test_dark_physics_full_rejects_wrong_free_parameter_count(self):
        with self.assertRaisesRegex(ValueError, "exactly 9 fitted parameters"):
            DarkCurrentParams.unpack(np.zeros(11))

    def test_group_split_keeps_complete_condition_curves_disjoint(self):
        spec = TASKS["I_photo"]
        frame = synthetic_photo_curves()
        train, test = split_task_data(frame, spec, seed=42, split_mode="group")
        group_cols = list(task_group_columns(spec))

        train_groups = set(map(tuple, train[group_cols].drop_duplicates().to_numpy()))
        test_groups = set(map(tuple, test[group_cols].drop_duplicates().to_numpy()))
        self.assertEqual(len(train_groups), 14)
        self.assertEqual(len(test_groups), 6)
        self.assertFalse(train_groups & test_groups)

        audit = split_audit_row(train, test, spec, seed=42, split_mode="group")
        self.assertEqual(audit["overlap_groups"], 0)
        self.assertEqual(audit["train_curves"], 14)
        self.assertEqual(audit["test_curves"], 6)

    def test_point_split_remains_backward_compatible(self):
        spec = TASKS["I_photo"]
        frame = synthetic_photo_curves()
        train, test = split_task_data(frame, spec, seed=42, split_mode="point")
        self.assertEqual(len(train), 56)
        self.assertEqual(len(test), 24)


if __name__ == "__main__":
    unittest.main()
