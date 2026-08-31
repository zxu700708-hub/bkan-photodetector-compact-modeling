import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BKAN_ROOT = ROOT / "bkan"
if str(BKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BKAN_ROOT))

from device_modeling.photodetector.data_ingestion import (  # noqa: E402
    AC_RESPONSE_DB_COLUMN,
    canonicalize_model_columns,
)
from device_modeling.photodetector.task_config import (  # noqa: E402
    NET_PHOTOCURRENT_COLUMN,
    TASKS,
    apply_dark_current_floor,
    prepare_task_dataframe,
)


class PrimaryPhotocurrentTargetTests(unittest.TestCase):
    def test_photo_task_uses_dark_subtracted_net_current(self):
        frame = pd.DataFrame(
            {
                "dark_voltage": [-3.0, -2.0],
                "light_voltage": [-3.0, -2.0],
                "dark_current": [2.0e-7, 3.0e-7],
                "light_current": [2.02e-4, 3.03e-4],
            }
        )
        cleaned = apply_dark_current_floor(frame)

        self.assertEqual(TASKS["photo_current"].target_col, NET_PHOTOCURRENT_COLUMN)
        np.testing.assert_allclose(
            cleaned[NET_PHOTOCURRENT_COLUMN],
            frame["light_current"] - frame["dark_current"],
        )
        self.assertEqual(cleaned.attrs["photocurrent_target_audit"]["paired_rows"], 2)

    def test_misaligned_dark_and_illuminated_biases_are_rejected(self):
        frame = pd.DataFrame(
            {
                "dark_voltage": [-3.0],
                "light_voltage": [-2.9],
                "dark_current": [2.0e-7],
                "light_current": [2.0e-4],
            }
        )
        with self.assertRaisesRegex(ValueError, "misaligned"):
            apply_dark_current_floor(frame)

    def test_prepared_net_photocurrent_has_stable_curve_ids(self):
        rows = []
        for condition in range(10):
            for voltage in (-3.0, -2.0):
                rows.append(
                    {
                        "dark_voltage": voltage,
                        "light_voltage": voltage,
                        "dark_current": 1.0e-7,
                        "light_current": 2.0e-4 + condition * 1.0e-6,
                        "active_layer_length": 35.0 + condition,
                    }
                )
        cleaned = apply_dark_current_floor(pd.DataFrame(rows))
        prepared, _ = prepare_task_dataframe(cleaned, TASKS["photo_current"])

        self.assertIn("_curve_id", prepared.columns)
        self.assertEqual(prepared["_curve_id"].nunique(), 10)

    def test_ac_metadata_identifies_optical_ssac_and_contact_current(self):
        frame = pd.DataFrame(
            {
                "frequency_ghz": [0.1, 1.0, 10.0],
                "bandwidth": [0.0, -1.0, -3.0],
            }
        )
        canonical = canonicalize_model_columns(frame)

        self.assertIn(AC_RESPONSE_DB_COLUMN, canonical.columns)
        self.assertIn("optical-generation SSAC", canonical.attrs["ac_target_semantics"])
        self.assertIn("contact-current", canonical.attrs["ac_target_semantics"])


if __name__ == "__main__":
    unittest.main()
