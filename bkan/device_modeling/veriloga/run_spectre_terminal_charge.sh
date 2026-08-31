#!/usr/bin/env bash
set -euo pipefail

# Full Spectre validation for the exact replacement terminal-charge source.
# Usage:
#   source <cadence-environment>
#   ./run_spectre_terminal_charge.sh [output-directory]
#
# The output directory must not already contain files.  A run is accepted only
# after implicit AHDL compilation during DC, DC/AC admittance, two transient
# step sizes, and the independent Python verifier all pass.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${1:-${SCRIPT_DIR}/terminal_charge_spectre_results}"

if ! command -v spectre >/dev/null 2>&1; then
  echo "[ERROR] spectre is unavailable; source the Cadence environment first." >&2
  exit 2
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "[ERROR] Python is required for independent reference checks." >&2
  exit 2
fi

if [ -e "${OUTPUT_ROOT}" ] && [ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "[ERROR] Refusing to mix a new run with existing files: ${OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/bundle" "${OUTPUT_ROOT}/results"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

MODEL=ge_si_photodetector_terminal_charge.va
DECKS=(
  testbench_dc_terminal_charge_ic618.scs
  testbench_ac_bias_temp_terminal_charge_ic618.scs
  testbench_transient_terminal_charge_ic618.scs
  testbench_transient_terminal_charge_fine_ic618.scs
)

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_terminal_charge_spectre.py" preflight \
  --output "${OUTPUT_ROOT}/preflight_manifest.json"

cp "${SCRIPT_DIR}/${MODEL}" "${OUTPUT_ROOT}/bundle/${MODEL}"
for deck in "${DECKS[@]}"; do
  cp "${SCRIPT_DIR}/${deck}" "${OUTPUT_ROOT}/bundle/${deck}"
done
cp "${SCRIPT_DIR}/verify_terminal_charge_spectre.py" "${OUTPUT_ROOT}/bundle/verify_terminal_charge_spectre.py"

spectre -version >"${OUTPUT_ROOT}/results/spectre_version.txt" 2>&1

pushd "${OUTPUT_ROOT}/bundle" >/dev/null

# Spectre 18.1 does not expose the newer -ahdl_check command-line option.
# Loading this exact deck necessarily compiles or loads the exact AHDL module;
# a zero-error DC completion is therefore the compilation/execution evidence.

spectre -64 +aps -format psfascii \
  testbench_dc_terminal_charge_ic618.scs \
  -raw "${OUTPUT_ROOT}/results/dc.raw" \
  2>&1 | tee "${OUTPUT_ROOT}/results/dc.log"

spectre -64 +aps -format psfascii \
  testbench_ac_bias_temp_terminal_charge_ic618.scs \
  -raw "${OUTPUT_ROOT}/results/ac.raw" \
  2>&1 | tee "${OUTPUT_ROOT}/results/ac.log"

spectre -64 +aps -format psfascii \
  testbench_transient_terminal_charge_ic618.scs \
  -raw "${OUTPUT_ROOT}/results/transient.raw" \
  2>&1 | tee "${OUTPUT_ROOT}/results/transient.log"

spectre -64 +aps -format psfascii \
  testbench_transient_terminal_charge_fine_ic618.scs \
  -raw "${OUTPUT_ROOT}/results/transient_fine.raw" \
  2>&1 | tee "${OUTPUT_ROOT}/results/transient_fine.log"

popd >/dev/null

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_terminal_charge_spectre.py" verify \
  --results "${OUTPUT_ROOT}/results" \
  --output "${OUTPUT_ROOT}/acceptance_report.json"

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
rows = []
for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "sha256_manifest.json"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append({"path": path.relative_to(root).as_posix(), "sha256": digest, "bytes": path.stat().st_size})
(root / "sha256_manifest.json").write_text(json.dumps({"files": rows}, indent=2), encoding="utf-8")
PY

echo "[PASS] Current replacement terminal-Q Spectre evidence: ${OUTPUT_ROOT}/acceptance_report.json"
