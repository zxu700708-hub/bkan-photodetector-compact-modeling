$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Join-Path $root "replay_output\verification_report.json"
New-Item -ItemType Directory -Force (Split-Path -Parent $out) | Out-Null
python (Join-Path $root "verify_artifact.py") --root $root --output $out
$replay = Join-Path $root "replay_output\current_evidence_replay.json"
python (Join-Path $root "replay_current_evidence.py") --root $root --output $replay
Write-Host "Verification report: $out"
Write-Host "Current-evidence replay: $replay"
