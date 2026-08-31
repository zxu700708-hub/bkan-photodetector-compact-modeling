import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_symbolic_export_audit import (  # noqa: E402
    _eval_poly,
    _eval_pp,
    _write_poly_va,
    _write_spline_va,
)


class SymbolicExportAuditTests(unittest.TestCase):
    def test_polynomial_export_replay(self):
        payload = {
            "x_mean": [1.0, 2.0], "x_scale": [2.0, 4.0],
            "powers": [[1, 0], [0, 2], [1, 1]],
            "coefficients": [2.0, -1.0, 0.5], "intercept": 3.0,
        }
        raw = np.array([[3.0, 6.0], [1.0, 2.0]])
        result = _eval_poly(payload, raw)
        self.assertTrue(np.allclose(result, [4.5, 3.0]))

    def test_piecewise_polynomial_clamps_domain(self):
        feature = {
            "domain": [0.0, 2.0], "breaks": [0.0, 1.0, 2.0],
            # f(x)=x on [0,1], f(x)=2x-1 on [1,2]
            "coefficients": [[1.0, 2.0], [0.0, 1.0]],
        }
        result = _eval_pp(feature, np.array([-1.0, 0.5, 1.5, 3.0]))
        self.assertTrue(np.allclose(result, [0.0, 0.5, 2.0, 3.0]))

    def test_poly_veriloga_declares_inputs(self):
        payload = {
            "inputs": ["bias_v"], "x_mean": [0.0], "x_scale": [1.0],
            "powers": [[1]], "coefficients": [2.0], "intercept": 1.0,
        }
        with patch.object(Path, "write_text") as write_text:
            _write_poly_va(payload, Path("poly.va"))
        source = write_text.call_args.args[0]
        self.assertIn("parameter real x0=0.0; // bias_v", source)
        self.assertIn("V(out) <+ y;", source)

    def test_spline_veriloga_has_upper_boundary_fallback(self):
        payload = {
            "inputs": ["frequency_ghz"], "x_mean": [0.0], "x_scale": [1.0],
            "intercept": 0.0,
            "features": [{
                "domain": [0.0, 2.0], "breaks": [0.0, 1.0, 2.0],
                "coefficients": [[1.0, 2.0], [0.0, 1.0]],
            }],
        }
        with patch.object(Path, "write_text") as write_text:
            _write_spline_va(payload, Path("spline.va"))
        source = write_text.call_args.args[0]
        self.assertIn("parameter real x0=0.0; // frequency_ghz", source)
        self.assertIn("else begin", source)
        self.assertNotIn("else if (z < 2", source)


if __name__ == "__main__":
    unittest.main()
