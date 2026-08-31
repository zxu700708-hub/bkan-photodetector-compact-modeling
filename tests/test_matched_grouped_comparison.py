import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_matched_grouped_comparison import (  # noqa: E402
    corrected_repeated_split_statistics,
    paired_bootstrap_interval,
    paired_statistics,
)


class MatchedGroupedComparisonTests(unittest.TestCase):
    def test_paired_bootstrap_is_reproducible(self):
        delta = np.array([-0.2, -0.1, 0.0, 0.1])
        first = paired_bootstrap_interval(delta, replicates=2000, seed=17)
        second = paired_bootstrap_interval(delta, replicates=2000, seed=17)
        self.assertEqual(first, second)
        self.assertLess(first[0], first[1])

    def test_corrected_interval_is_wider_than_naive_standard_error(self):
        delta = np.array([-0.2, -0.1, -0.15, -0.05, -0.12])
        low, high, corrected_se, p_value = corrected_repeated_split_statistics(delta)
        naive_se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        self.assertGreater(corrected_se, naive_se)
        self.assertLess(low, high)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_paired_statistics_uses_prespecified_reference_and_holm(self):
        rows = []
        values = {
            "bkan": [0.8, 0.9, 1.0],
            "dkan": [1.0, 1.1, 1.2],
            "mlp_l": [0.7, 0.8, 0.9],
            "poly3_ridge": [1.3, 1.4, 1.5],
            "spline_ridge": [1.1, 1.2, 1.3],
        }
        for model, metrics in values.items():
            for seed, rmse in zip((42, 43, 44), metrics):
                rows.append(
                    {"task": "I_dark", "seed": seed, "model": model, "rmse_target": rmse}
                )
        pairwise, selected = paired_statistics(
            pd.DataFrame(rows), replicates=2000, seed=19
        )
        self.assertEqual(len(pairwise), 4)
        self.assertEqual(selected.iloc[0]["baseline"], "spline_ridge")
        self.assertIn("not selected", selected.iloc[0]["selection_rule"])
        self.assertEqual(int(selected.iloc[0]["wins"]), 3)
        self.assertIn("corrected_p_value_holm", pairwise.columns)
        self.assertTrue(
            np.all(
                pairwise["corrected_p_value_holm"]
                >= pairwise["corrected_p_value_raw"]
            )
        )
        dkan = pairwise[pairwise["baseline"] == "dkan"].iloc[0]
        self.assertEqual(int(dkan["wins"]), 3)
        self.assertAlmostEqual(float(dkan["paired_delta_mean"]), -0.2)

    def test_incomplete_pairing_is_rejected(self):
        frame = pd.DataFrame(
            {
                "task": ["I_dark"] * 4,
                "seed": [42, 42, 42, 42],
                "model": ["bkan", "dkan", "mlp_l", "poly3_ridge"],
                "rmse_target": [1.0, 1.1, 1.2, 1.3],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "Incomplete matched model set"):
            paired_statistics(frame, replicates=1000)


if __name__ == "__main__":
    unittest.main()
