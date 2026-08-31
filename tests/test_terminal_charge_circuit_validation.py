from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "bkan" / "device_modeling" / "veriloga"
BUNDLE = (
    ROOT
    / "artifacts"
    / "results"
    / "terminal_charge_circuit_validation"
    / "preflight_bundle"
)
sys.path.insert(0, str(MODULE_DIR))
import terminal_charge_circuit_validation as circuit  # noqa: E402


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]
    }


def write_psf(path: Path, columns: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["HEADER", "VALUE"]
    length = len(next(iter(columns.values())))
    for index in range(length):
        for name, values in columns.items():
            value = complex(values[index])
            if value.imag:
                lines.append(f'"{name}" ({value.real:.17e} {value.imag:.17e})')
            else:
                lines.append(f'"{name}" {value.real:.17e}')
    lines.append("END")
    path.write_text("\n".join(lines), encoding="utf-8")


class TerminalChargeCircuitPreflightTests(unittest.TestCase):
    def test_source_and_runner_are_static_parseable(self) -> None:
        ast.parse(
            (MODULE_DIR / "terminal_charge_circuit_validation.py").read_text(
                encoding="utf-8"
            ),
            feature_version=(3, 6),
        )
        # MODEL_SHA256 belongs to the frozen, previously exercised bundle.
        # The corrected net-photocurrent source must be re-bundled and rerun;
        # treating the old hash as current would transfer stale evidence.
        self.assertNotEqual(circuit.sha256(circuit.MODEL), circuit.MODEL_SHA256)
        runner = (
            MODULE_DIR / "run_spectre_terminal_charge_circuits.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("bias_load", runner)
        self.assertIn("tia_ac", runner)
        self.assertIn("tia_transient", runner)
        for count in circuit.MULTI_COUNTS:
            self.assertIn(f"multi_{count:03d}", runner)
        self.assertNotIn("-ahdl_check", runner)
        self.assertIn("terminal_charge_circuit_validation.py\" verify", runner)
        self.assertNotIn("sys.version_info >= (3, 10)", runner)
        self.assertNotIn("remote_numeric_verification_skipped", runner)
        self.assertNotIn("from __future__ import annotations\nimport hashlib", runner)

    def test_current_generated_tia_decks_do_not_save_invalid_isource_branch(self) -> None:
        self.assertNotIn("Iinj:p", circuit.DECKS["testbench_tia_ac_terminal_charge_ic618.scs"])
        self.assertNotIn("Iinj:p", circuit.DECKS["testbench_tia_transient_terminal_charge_ic618.scs"])

    def test_preflight_bundle_has_frozen_scope_and_complete_references(self) -> None:
        manifest = json.loads(
            (BUNDLE / "preflight_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["model_sha256"], circuit.MODEL_SHA256)
        self.assertEqual(manifest["status"], "ready_for_remote_spectre")
        self.assertFalse(manifest["optical_port_validation"])
        self.assertFalse(manifest["device_qualification"])
        reference = BUNDLE / "reference"
        self.assertEqual(len(read_csv(reference / "bias_load_reference.csv")["source_V"]), 57)
        self.assertEqual(len(read_csv(reference / "tia_ac_reference.csv")["frequency_Hz"]), 101)
        self.assertEqual(len(read_csv(reference / "tia_transient_reference.csv")["time_s"]), 5001)
        self.assertEqual(len(read_csv(reference / "multi_instance_reference.csv")["frequency_Hz"]), 303)

    def test_decks_are_portable_and_use_only_current_terminal_q_source(self) -> None:
        decks = sorted(BUNDLE.glob("*.scs"))
        self.assertEqual(len(decks), 6)
        for deck in decks:
            text = deck.read_text(encoding="utf-8")
            self.assertIn(
                'ahdl_include "ge_si_photodetector_terminal_charge.va"', text
            )
            self.assertNotIn("/home/", text)
            self.assertNotIn("charge_proxy", text)

    def test_reference_operating_points_stay_in_characterized_voltage_range(self) -> None:
        bias = read_csv(BUNDLE / "reference" / "bias_load_reference.csv")
        self.assertGreaterEqual(float(np.min(bias["pd_voltage_V"])), -3.0)
        self.assertLessEqual(float(np.max(bias["pd_voltage_V"])), 0.0)
        operating = json.loads(
            (BUNDLE / "reference" / "tia_operating_point.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertGreaterEqual(operating["pd_voltage_dc_V"], -3.0)
        self.assertLessEqual(operating["pd_voltage_dc_V"], 0.0)


class TerminalChargeCircuitSyntheticAcceptanceTests(unittest.TestCase):
    def test_exact_frozen_reference_replay_passes_end_to_end(self) -> None:
        scratch = (
            ROOT
            / "artifacts"
            / "results"
            / "terminal_charge_circuit_validation"
            / "test_scratch"
        )
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = scratch / uuid.uuid4().hex
        temporary.mkdir()
        try:
            run_root = temporary / "run"
            shutil.copytree(BUNDLE, run_root / "bundle")
            results = run_root / "results"
            results.mkdir(parents=True)
            reference = run_root / "bundle" / "reference"

            bias = read_csv(reference / "bias_load_reference.csv")
            write_psf(
                results / "bias_load.raw" / "bias_load.dc",
                {
                    "Vsweep": bias["source_V"],
                    "pd": bias["pd_anode_V"],
                    "Vdrive:p": bias["source_branch_current_A"],
                },
            )
            ac = read_csv(reference / "tia_ac_reference.csv")
            write_psf(
                results / "tia_ac.raw" / "tia_ac.ac",
                {
                    "freq": ac["frequency_Hz"],
                    "in": ac["input_real_V"] + 1j * ac["input_imag_V"],
                    "out": ac["output_real_V"] + 1j * ac["output_imag_V"],
                },
            )
            transient = read_csv(reference / "tia_transient_reference.csv")
            write_psf(
                results / "tia_transient.raw" / "tia_transient.tran",
                {
                    "time": transient["time_s"],
                    "in": transient["input_V"],
                    "out": transient["output_V"],
                },
            )
            multi = read_csv(reference / "multi_instance_reference.csv")
            for count in circuit.MULTI_COUNTS:
                subset = multi["instances"] == count
                write_psf(
                    results / f"multi_{count:03d}.raw" / f"multi_{count:03d}.ac",
                    {
                        "freq": multi["frequency_Hz"][subset],
                        "Vstim:p": multi["source_current_real_A"][subset]
                        + 1j * multi["source_current_imag_A"][subset],
                    },
                )
                (results / f"multi_{count:03d}.runtime.txt").write_text(
                    "real 1.0\nuser 0.8\nsys 0.2\n", encoding="utf-8"
                )
            labels = ["bias_load", "tia_ac", "tia_transient"] + [
                f"multi_{count:03d}" for count in circuit.MULTI_COUNTS
            ]
            for label in labels:
                (results / f"{label}.log").write_text(
                    "spectre completes with 0 errors, 0 warnings.\n",
                    encoding="utf-8",
                )
            output = run_root / "circuit_acceptance_report.json"
            code = circuit.verify(run_root, output)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(code, 0, report)
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["failed_checks"])
            self.assertIn("device_qualification", report["claims_not_supported"])
        finally:
            shutil.rmtree(temporary)


if __name__ == "__main__":
    unittest.main()
