"""Focused checks for the frozen sparse-current/charge Spectre candidate."""

import ast
import importlib.util
import json
import math
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "bkan/device_modeling/veriloga"
sys.path.insert(0, str(MODULE_DIR))
import physics_sparse_validation as validation  # noqa: E402


class PhysicsSparseValidationTests(unittest.TestCase):
    def test_portable_scripts_parse_as_python_36(self):
        for name in ("physics_sparse_validation.py", "physics_sparse_replay.py"):
            ast.parse((MODULE_DIR / name).read_text(encoding="utf-8"), feature_version=(3, 6))

    def test_portable_replay_matches_frozen_predictions(self):
        replay = validation.load_replay()
        for task in validation.TASKS:
            for seed in (42, 43, 44):
                directory = validation.FORMULA_DIR / task / "seed_{}".format(seed)
                payload = json.loads((directory / "selected_formula.json").read_text(encoding="utf-8"))
                frame = pd.read_csv(directory / "test_predictions.csv")
                prediction, derivative = replay.replay(payload, frame)
                np.testing.assert_allclose(prediction, frame["selected"], rtol=0, atol=2e-14)
                if task == "Q_terminal":
                    derivative_frame = pd.read_csv(directory / "derivative_test_predictions.csv")
                    _, derivative = replay.replay(payload, derivative_frame)
                    np.testing.assert_allclose(derivative, derivative_frame["selected"], rtol=0, atol=2e-27)
                else:
                    np.testing.assert_allclose(derivative, frame["selected_derivative"], rtol=0, atol=2e-9)

    def test_rendered_equations_match_independent_replay(self):
        formulas = validation.load_formulas()
        source = validation.render_model(formulas)
        self.assertIn("ddt(q_terminal)", source)
        self.assertNotIn("--", source)
        replay = validation.load_replay()
        voltage = np.linspace(-3.0, 0.0, 73)
        for condition in validation.CONDITIONS:
            frame = validation.condition_frame(voltage, 305.0, condition)
            arguments = {name: frame[name].to_numpy(float) for name in frame}
            arguments.update(sparse_pos=lambda x: np.maximum(x, 0.0), pow=np.power)
            for task, name in (("I_dark", "sparse_dark"), ("I_photo", "sparse_photo"),
                               ("Q_terminal", "sparse_charge")):
                match = re.search(r"\b{} = (.*?);".format(name), source, re.S)
                self.assertIsNotNone(match)
                rendered = eval(match.group(1), {"__builtins__": {}}, arguments)
                expected, _ = replay.replay(formulas[task], frame)
                if task != "Q_terminal":
                    expected = np.power(10.0, expected)
                np.testing.assert_allclose(rendered, expected, rtol=3e-13, atol=1e-29)

    def test_device_acceptance_requires_full_dc_and_ac_grids(self):
        current = validation.SymbolicCurrentReference()
        charge = validation.ChargeReference()
        dc_payloads = {}
        grid = np.linspace(-3.0, 0.0, 61)
        for index, temperature in enumerate(validation.TEMPERATURE_VALUES):
            path = Path("temperature_sweep-{:03d}_dc_sparse.dc".format(index))
            dc_payloads[path] = {"Vb_dc": grid.astype(complex)}
            for trace, light, condition in (("Vdark:p", 0, "nominal"),
                                            ("Vlight:p", 1, "nominal"),
                                            ("Vcorner:p", 1, "corner")):
                dc_payloads[path][trace] = -current.evaluate(grid, temperature, light, condition).astype(complex)
        with patch.object(validation, "parse_psf", side_effect=lambda path: dc_payloads[path]):
            self.assertTrue(validation.validate_dc(list(dc_payloads), current)["passed"])
            shortened = {path: dict(values) for path, values in dc_payloads.items()}
            first = next(iter(shortened))
            shortened[first] = {key: value[:7] for key, value in shortened[first].items()}
            with patch.object(validation, "parse_psf", side_effect=lambda path: shortened[path]):
                self.assertFalse(validation.validate_dc(list(shortened), current)["passed"])

        ac_payloads = {}
        frequencies = validation.FREQUENCIES
        for b, bias in enumerate(validation.BIAS_VALUES):
            for t, temperature in enumerate(validation.TEMPERATURE_VALUES):
                path = Path("bias_sweep-{:03d}_temperature_sweep-{:03d}_ac_sparse.ac".format(b, t))
                ac_payloads[path] = {"freq": frequencies.astype(complex)}
                for trace, light, condition in (("Vac_dark:p", 0, "nominal"),
                                                ("Vac_light:p", 1, "nominal"),
                                                ("Vac_corner:p", 1, "corner")):
                    didv = current.derivative([bias], temperature, light, condition)[0]
                    _, dqdv = charge.evaluate(validation.condition_frame([bias], temperature, condition))
                    ac_payloads[path][trace] = -validation.VAC_MAG * (
                        didv + 1j * 2.0 * math.pi * frequencies * dqdv[0])
        with patch.object(validation, "parse_psf", side_effect=lambda path: ac_payloads[path]):
            self.assertTrue(validation.validate_ac(list(ac_payloads), current, charge)["passed"])
            shortened = {path: dict(values) for path, values in ac_payloads.items()}
            first = next(iter(shortened))
            shortened[first] = {key: value[:3] for key, value in shortened[first].items()}
            with patch.object(validation, "parse_psf", side_effect=lambda path: shortened[path]):
                self.assertFalse(validation.validate_ac(list(shortened), current, charge)["passed"])

    def test_clamped_charge_has_zero_outside_tangent(self):
        charge = validation.ChargeReference()
        frame = validation.condition_frame([-3.1, -3.0, 0.0, 0.1])
        values, derivative = charge.evaluate(frame)
        self.assertEqual(derivative[0], 0.0)
        self.assertEqual(derivative[-1], 0.0)
        self.assertGreater(derivative[1], 0.0)
        self.assertGreater(derivative[2], 0.0)
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[2], values[3])

    def test_returned_spectre_reports_bind_selected_candidate(self):
        evidence = ROOT / "artifacts/results/physics_sparse_results_py36_20260921"
        device = json.loads(
            (evidence / "device/acceptance_report.json").read_text(encoding="utf-8")
        )
        circuit = json.loads(
            (evidence / "circuits/circuit_acceptance_report.json").read_text(encoding="utf-8")
        )
        device_source = evidence / "device/bundle" / validation.MODEL_NAME
        circuit_source = evidence / "circuits/bundle" / validation.MODEL_NAME
        model_hash = validation.sha256(device_source)

        self.assertEqual(device["status"], "passed")
        self.assertEqual(circuit["status"], "passed")
        self.assertTrue(device["checks"]["source_binding"])
        self.assertTrue(circuit["checks"]["source_binding"]["passed"])
        self.assertEqual(device["model_sha256"], model_hash)
        self.assertEqual(circuit["checks"]["source_binding"]["expected_sha256"], model_hash)
        self.assertEqual(circuit["checks"]["source_binding"]["returned_sha256"], model_hash)
        self.assertEqual(validation.sha256(circuit_source), model_hash)


if __name__ == "__main__":
    unittest.main()
