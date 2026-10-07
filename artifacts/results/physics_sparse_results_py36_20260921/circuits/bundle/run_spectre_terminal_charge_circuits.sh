#!/usr/bin/env bash
set -euo pipefail

# Run inside the generated bundle on a host with Spectre 18.1 or newer.
# Usage: bash run_spectre_terminal_charge_circuits.sh /absolute/new/run_directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${1:?Provide an empty absolute output directory}"

if ! command -v spectre >/dev/null 2>&1; then
  echo "[ERROR] spectre is unavailable; source the Cadence environment first." >&2
  exit 2
fi
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "[ERROR] Python is required for returned-result verification." >&2
  exit 2
fi
if [ -e "${RUN_ROOT}" ] && [ -n "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "[ERROR] Refusing to mix a new run with existing files: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/bundle" "${RUN_ROOT}/results"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
cp -R "${SCRIPT_DIR}/." "${RUN_ROOT}/bundle/"
spectre -version >"${RUN_ROOT}/results/spectre_version.txt" 2>&1

run_deck() {
  local label="$1"
  local deck="$2"
  pushd "${RUN_ROOT}/bundle" >/dev/null
  spectre -64 +aps -format psfascii "${deck}" -raw "${RUN_ROOT}/results/${label}.raw" 2>&1 | tee "${RUN_ROOT}/results/${label}.log"
  popd >/dev/null
}

run_timed_deck() {
  local label="$1"
  local deck="$2"
  pushd "${RUN_ROOT}/bundle" >/dev/null
  { time -p spectre -64 +aps -format psfascii "${deck}" -raw "${RUN_ROOT}/results/${label}.raw" 2>&1 | tee "${RUN_ROOT}/results/${label}.log"; } 2>"${RUN_ROOT}/results/${label}.runtime.txt"
  popd >/dev/null
}

run_deck bias_load testbench_bias_load_terminal_charge_ic618.scs
run_deck tia_ac testbench_tia_ac_terminal_charge_ic618.scs
run_deck tia_transient testbench_tia_transient_terminal_charge_ic618.scs
run_timed_deck multi_001 testbench_multi_instance_001_terminal_charge_ic618.scs
run_timed_deck multi_010 testbench_multi_instance_010_terminal_charge_ic618.scs
run_timed_deck multi_100 testbench_multi_instance_100_terminal_charge_ic618.scs

"${PYTHON_BIN}" "${RUN_ROOT}/bundle/terminal_charge_circuit_validation.py" verify \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/circuit_acceptance_report.json"

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
rows = []
for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "sha256_manifest.json"):
    rows.append({
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    })
(root / "sha256_manifest.json").write_text(json.dumps({"files": rows}, indent=2), encoding="utf-8")
PY

echo "[PASS] Circuit-level report: ${RUN_ROOT}/circuit_acceptance_report.json"
