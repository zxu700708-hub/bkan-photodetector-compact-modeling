from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "bkan" / "device_modeling" / "veriloga"
sys.path.insert(0, str(MODULE_DIR))
import verify_terminal_charge_spectre as verifier  # noqa: E402


class TerminalChargeSpectrePreflightTests(unittest.TestCase):
    def test_verifier_source_parses_without_writing_bytecode(self) -> None:
        ast.parse(
            (MODULE_DIR / "verify_terminal_charge_spectre.py").read_text(
                encoding="utf-8"
            ),
            feature_version=(3, 6),
        )

    def test_exact_replacement_source_matches_frozen_artifact(self) -> None:
        self.assertEqual(
            verifier.sha256(verifier.MODEL_SOURCE),
            verifier.sha256(verifier.ARTIFACT_SOURCE),
        )
        source = verifier.MODEL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("module ge_si_photodetector_terminal_charge", source)
        self.assertIn("ddt(q_terminal)", source)
        self.assertNotIn("q_proxy", source)

    def test_all_current_decks_are_portable_and_replacement_only(self) -> None:
        for name in verifier.DECK_NAMES:
            text = (MODULE_DIR / name).read_text(encoding="utf-8")
            self.assertIn(
                'ahdl_include "ge_si_photodetector_terminal_charge.va"', text
            )
            self.assertNotIn("/home/", text)
            self.assertNotIn("charge_proxy", text)
            self.assertNotIn("ge_si_pdet_fixed", text)

    def test_default_runner_dispatches_to_full_terminal_charge_chain(self) -> None:
        wrapper = (MODULE_DIR / "run_spectre.sh").read_text(encoding="utf-8")
        dispatch = wrapper.index("run_spectre_terminal_charge.sh")
        historical_target = wrapper.index('VA_MODEL="ge_si_pdet_fixed.va"')
        self.assertLess(dispatch, historical_target)
        runner = (MODULE_DIR / "run_spectre_terminal_charge.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "testbench_dc_terminal_charge_ic618.scs",
            "testbench_ac_bias_temp_terminal_charge_ic618.scs",
            "testbench_transient_terminal_charge_ic618.scs",
            "testbench_transient_terminal_charge_fine_ic618.scs",
            "acceptance_report.json",
        ):
            self.assertIn(token, runner)
        self.assertNotIn("spectre -ahdl_check", runner)
        self.assertIn("implicit AHDL compilation during DC", runner)
        self.assertNotIn("charge_proxy", runner)

    def test_current_reference_does_not_load_retired_charge_model(self) -> None:
        reference = verifier.SymbolicCurrentReference()
        values = reference.evaluate([-3.0, -1.0, 0.0], 300.0)
        derivative = reference.derivative([-3.0, -1.0, 0.0], 300.0)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.isfinite(derivative).all())
        source = (MODULE_DIR / "verify_terminal_charge_spectre.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("ProxyChargeReference", source)
        self.assertNotIn("charge_model/models.json", source)

    def test_upper_boundary_derivative_matches_spectre_equality_branch(self) -> None:
        current = verifier.SymbolicCurrentReference()
        step = 1.0e-5
        expected = (
            current._evaluate_unclipped([step], 300.0)[0]
            - current._evaluate_unclipped([0.0], 300.0)[0]
        ) / step
        actual = current.derivative([0.0], 300.0, step=step)[0]
        self.assertAlmostEqual(actual, expected, places=12)

        backward = (
            current._evaluate_unclipped([0.0], 300.0)[0]
            - current._evaluate_unclipped([-step], 300.0)[0]
        ) / step
        self.assertGreater(abs(actual - backward), 7.0e-4)

    def test_preflight_artifact_is_pending_not_a_simulator_pass(self) -> None:
        path = (
            ROOT
            / "artifacts"
            / "results"
            / "terminal_charge_spectre"
            / "preflight_manifest.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "ready_to_run")
        self.assertEqual(payload["simulator_execution_status"], "pending")
        self.assertFalse(payload["legacy_proxy_evidence_transfer"])
        self.assertEqual(payload["references"]["dc_rows"], 183)
        self.assertEqual(payload["references"]["ac_rows"], 1815)


class PsfAsciiParserTests(unittest.TestCase):
    def test_parser_handles_real_and_complex_traces(self) -> None:
        path = ROOT / "tmp_terminal_charge_psf_parser_test.ac"
        path.write_text(
            '\n'.join(
                [
                    "HEADER",
                    "VALUE",
                    '"freq" 1.000000e+06',
                    '"Vac:p" (1.0e-08 -2.0e-12)',
                    '"freq" 2.000000e+06',
                    '"Vac:p" (1.1e-08 -4.1e-12)',
                    "END",
                ]
            ),
            encoding="utf-8",
        )
        try:
            parsed = verifier.parse_psf_ascii(path)
        finally:
            path.unlink()
        np.testing.assert_allclose(parsed["freq"].real, [1.0e6, 2.0e6])
        np.testing.assert_allclose(parsed["Vac:p"].imag, [-2.0e-12, -4.1e-12])

    def test_dc_and_ac_verifiers_accept_exact_independent_replay(self) -> None:
        root = Path("synthetic")
        current = verifier.SymbolicCurrentReference()
        charge = verifier.terminal_reference()

        dc_files = []
        dc_payloads = {}
        for temperature_index, temperature in enumerate(verifier.TEMPERATURE_VALUES):
            path = Path(
                f"temperature_sweep-{temperature_index:03d}_dc_terminal_charge.dc"
            )
            bias = np.linspace(verifier.V_MIN, verifier.V_MAX, 7)
            dc_files.append(path)
            dc_payloads[path] = {
                "Vb_dc": bias.astype(np.complex128),
                "Vdc:p": (-current.evaluate(bias, temperature)).astype(np.complex128),
            }

        with patch.object(verifier, "indexed_files", return_value=dc_files), patch.object(
            verifier, "parse_psf_ascii", side_effect=lambda path: dc_payloads[path]
        ):
            dc = verifier.verify_dc(root)

        frequencies = np.asarray([1.0e6, 1.0e9, 1.0e12])
        ac_files = []
        ac_payloads = {}
        for bias_index, bias in enumerate(verifier.BIAS_VALUES):
            for temperature_index, temperature in enumerate(
                verifier.TEMPERATURE_VALUES
            ):
                path = Path(
                    f"bias_sweep-{bias_index:03d}_temperature_sweep-"
                    f"{temperature_index:03d}_ac_terminal_charge.ac"
                )
                didv = current.derivative([bias], temperature)[0]
                _, dqdv = charge.evaluate(verifier.condition_frame([bias], temperature))
                source_current = -verifier.VAC_MAG * (
                    didv + 1j * 2.0 * np.pi * frequencies * dqdv[0]
                )
                ac_files.append(path)
                ac_payloads[path] = {
                    "freq": frequencies.astype(np.complex128),
                    "Vac:p": source_current.astype(np.complex128),
                }

        with patch.object(verifier, "indexed_files", return_value=ac_files), patch.object(
            verifier, "parse_psf_ascii", side_effect=lambda path: ac_payloads[path]
        ):
            ac = verifier.verify_ac(root)

        self.assertTrue(dc["passed"], dc)
        self.assertTrue(ac["passed"], ac)
        self.assertEqual(dc["files"], 3)
        self.assertEqual(ac["files"], 15)


if __name__ == "__main__":
    unittest.main()
