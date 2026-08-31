#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ROOT}/replay_output"
python3 "${ROOT}/verify_artifact.py" --root "${ROOT}" --output "${ROOT}/replay_output/verification_report.json"
python3 "${ROOT}/replay_current_evidence.py" --root "${ROOT}" --output "${ROOT}/replay_output/current_evidence_replay.json"
printf 'Verification report: %s\n' "${ROOT}/replay_output/verification_report.json"
printf 'Current-evidence replay: %s\n' "${ROOT}/replay_output/current_evidence_replay.json"
