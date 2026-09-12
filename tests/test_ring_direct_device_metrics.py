from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_ring_direct_device_metrics import TASKS, split_frame, task_metrics


class RingDirectDeviceMetricTests(unittest.TestCase):
    def test_split_frame_keeps_structures_intact(self) -> None:
        rows = []
        assignments = []
        names = ["train", "validation", "calibration", "test"]
        counts = [52, 8, 12, 8]
        cursor = 0
        for name, count in zip(names, counts):
            for index in range(cursor, cursor + count):
                structure = f"s{index:02d}"
                assignments.append({"seed": 42, "structure_id": structure, "split": name})
                for bias in (0.0, -1.0):
                    rows.append({"structure_id": structure, "bias_v": bias, "split": "legacy"})
            cursor += count
        train, validation, calibration, test, _ = split_frame(
            pd.DataFrame(rows), pd.DataFrame(assignments), 42
        )
        self.assertEqual(
            [part["structure_id"].nunique() for part in (train, validation, calibration, test)],
            counts,
        )
        self.assertFalse(set(train["structure_id"]) & set(test["structure_id"]))

    def test_shift_nonzero_metric_excludes_zero_bias_denominator(self) -> None:
        task = TASKS["resonance_shift"]
        test = pd.DataFrame(
            {
                "structure_id": ["a", "a", "b", "b"],
                "resonance_shift_pm": [0.0, 10.0, 0.0, 20.0],
            }
        )
        values = task_metrics(test, task, np.asarray([100.0, 11.0, -100.0, 22.0]))
        self.assertAlmostEqual(values["nonzero_bias_rmse_raw"], np.sqrt(2.5))
        self.assertAlmostEqual(values["median_ape_percent"], 10.0)


if __name__ == "__main__":
    unittest.main()
