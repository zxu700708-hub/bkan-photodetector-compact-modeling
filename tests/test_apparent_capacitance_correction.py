import importlib.util
import math
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/apply_apparent_capacitance_correction.py"
ROOT = SCRIPT.parents[1]
BKAN_ROOT = ROOT / "bkan"
if str(BKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BKAN_ROOT))
SPEC = importlib.util.spec_from_file_location("cap_correction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

from device_modeling.photodetector.task_config import (  # noqa: E402
    DEFAULT_CAPACITANCE_DATA,
    load_capacitance_data,
)


class ApparentCapacitanceCorrectionTest(unittest.TestCase):
    def test_corrected_frame_scales_admittance_and_target(self):
        frame = pd.DataFrame(
            {
                "sample_id": [0],
                "bias_v": [-3.0],
                "frequency_hz": [1.0e8],
                "Y_real_S": [2.0e-11],
                "Y_imag_S": [2.0 * math.pi * 1.0e8 * 3.0e-17],
                "Y_abs_S": [4.0e-8],
                "capacitance_F": [3.0e-17],
                "conductance_S": [2.0e-11],
                "vac_V": [1.0],
            }
        )
        corrected = MODULE.corrected_frame(frame)
        self.assertEqual(corrected.loc[0, "vac_V"], 1.0e-3)
        self.assertEqual(corrected.loc[0, "legacy_capacitance_F"], 3.0e-17)
        self.assertAlmostEqual(corrected.loc[0, "capacitance_F"], 3.0e-14)
        self.assertAlmostEqual(corrected.loc[0, "Y_real_S"], 2.0e-8)

    def test_formula_json_scales_output_scale(self):
        payload = MODULE.correct_formula_payload({"output_scale": 2e-17})
        self.assertAlmostEqual(payload["output_scale"], 2e-14)
        self.assertEqual(payload["target_symbol"], "C_app")

    def test_formula_json_scales_nested_spline_coefficients(self):
        payload = MODULE.correct_formula_payload(
            {
                "intercept": 2e-17,
                "features": [{"coefficients": [[1e-18, 2e-18]]}],
            }
        )
        self.assertAlmostEqual(payload["intercept"], 2e-14)
        self.assertAlmostEqual(payload["features"][0]["coefficients"][0][1], 2e-15)

    def test_default_loader_uses_only_corrected_apparent_capacitance(self):
        frame = load_capacitance_data()
        self.assertEqual(DEFAULT_CAPACITANCE_DATA.resolve(), (ROOT / "artifacts/results/apparent_capacitance_correction/primary_ge_si_apparent_capacitance.csv").resolve())
        self.assertEqual(len(frame), 64_000)
        self.assertEqual(frame["sample_id"].nunique(), 160)
        self.assertTrue((frame["vac_V"] == 1.0e-3).all())
        self.assertTrue(frame["target_definition"].str.startswith("C_app=").all())


if __name__ == "__main__":
    unittest.main()
