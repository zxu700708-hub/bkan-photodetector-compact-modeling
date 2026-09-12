from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_ring_third_device import RingTask, deterministic_limit, prediction_metrics


class RingThirdDeviceTests(unittest.TestCase):
    def test_curve_limit_preserves_both_sampling_regions(self) -> None:
        frame = pd.DataFrame(
            {
                "curve_id": ["a"] * 200,
                "structure_id": ["s"] * 200,
                "bias_v": [0.0] * 200,
                "curve_row_index": np.arange(200),
                "sample_reason": ["resonance_dense"] * 120 + ["off_resonance_coarse"] * 80,
            }
        )
        limited = deterministic_limit(frame, 40)
        self.assertLessEqual(len(limited), 40)
        self.assertEqual(set(limited["sample_reason"]), {"resonance_dense", "off_resonance_coarse"})

    def test_weighted_metrics_are_structure_macro_auditable(self) -> None:
        task = RingTask("through", "Ring_through_dB", "through_db", "Through")
        frame = pd.DataFrame(
            {
                "through_db": [0.0, 0.0, 0.0, 0.0],
                "structure_id": ["a", "a", "b", "b"],
                "curve_id": ["a0", "a0", "b0", "b0"],
                "sample_reason": ["resonance_dense", "off_resonance_coarse"] * 2,
                "represented_full_grid_rows": [1.0, 9.0, 1.0, 9.0],
            }
        )
        result = prediction_metrics(frame, task, np.array([1.0, 0.0, 3.0, 0.0]))
        self.assertAlmostEqual(result["rmse_target"], np.sqrt(0.5))
        self.assertAlmostEqual(result["resonance_region_rmse_db"], np.sqrt(5.0))
        self.assertTrue(np.isfinite(result["structure_macro_rmse_db"]))


if __name__ == "__main__":
    unittest.main()
