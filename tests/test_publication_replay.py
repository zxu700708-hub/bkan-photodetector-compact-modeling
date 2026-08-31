import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replay_current_evidence import replay_grouped  # noqa: E402


class PublicationReplayTests(unittest.TestCase):
    def test_capacitance_rmse_uses_registered_physical_scale(self):
        report = replay_grouped(ROOT)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["capacitance_scale_applied"], 1000.0)
        row = next(
            item
            for item in report["summary"]
            if item["task"] == "Capacitance" and item["model"] == "bkan"
        )
        self.assertEqual(row["unit"], "F")
        self.assertAlmostEqual(row["rmse_mean"], 6.853730269380608e-17, places=28)


if __name__ == "__main__":
    unittest.main()
