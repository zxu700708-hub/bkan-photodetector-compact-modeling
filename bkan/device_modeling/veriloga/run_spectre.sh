#!/bin/bash
set -euo pipefail

# Canonical entry point.  The historical fixed/proxy commands below are kept
# only as source history and are unreachable: their evidence cannot transfer to
# the replacement terminal-Q module.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_spectre_terminal_charge.sh" "$@"

# ============================================================
# KAN Verilog-A → IC618 Spectre 一键验证脚本
# ============================================================
# 用法:
#   chmod +x run_spectre.sh
#   ./run_spectre.sh              # 完整验证
#   ./run_spectre.sh --quick      # 仅 DC dark current
#   ./run_spectre.sh --full       # 含 AC + transient
#
# 前提: 需先 source Cadence 环境
#   source /opt/cadence/IC618/tools/cshrc   (根据实际安装路径调整)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  KAN Verilog-A IC618 Verification"
echo "============================================"

# ── 检查 Cadence 环境 ──
if ! command -v spectre &>/dev/null; then
    echo ""
    echo "[ERROR] spectre 未找到，请先 source Cadence 环境:"
    echo "  bash: source /opt/cadence/installs/IC618/tools/cshrc"
    echo "  或:   source ~/.bashrc_cadence"
    echo ""
    exit 1
fi

echo "[OK] spectre: $(which spectre)"
spectre -version 2>&1 | head -3

# ── Step 1: Verilog-A 语法检查 ──
echo ""
echo "── Step 1/4: Verilog-A Syntax Check ──"

VA_MODEL="ge_si_pdet_fixed.va"

if spectre -ahdl_check "$VA_MODEL" 2>&1; then
    echo "[PASS] $VA_MODEL syntax OK"
else
    echo "[FAIL] $VA_MODEL has syntax errors — aborting"
    exit 1
fi

# ── Step 2: DC Verification ──
echo ""
echo "── Step 2/4: DC Sweep Simulation ──"

DC_TB="testbench_dc_ic618.scs"

spectre +aps -format sst2 "$DC_TB" -o dc_result 2>&1 | tee dc_run.log

# Check for convergence
echo ""
echo "── Convergence Summary ──"
grep -i -E "converge|fail|newton|error" dc_run.log | tail -30 || echo "  (no issues flagged)"

FAILS=$(grep -c -i "failed.*converge\|convergence.*failed" dc_run.log || true)
if [ "$FAILS" -gt 0 ]; then
    echo "[WARN] $FAILS convergence failure(s) detected — see dc_run.log"
else
    echo "[PASS] No convergence failures"
fi

# ── Step 3: Extract & Compare ──
echo ""
echo "── Step 3/4: Extract & Compare Results ──"

python3 compare_results.py dc_result.raw spectre_reference.txt 2>&1 || \
python compare_results.py dc_result.raw spectre_reference.txt 2>&1 || \
echo "[WARN] Automatic comparison skipped (Python not available or parse error)"

# ── Step 4: Report ──
echo ""
echo "── Step 4/4: Output Files ──"
echo "  Simulation log:     dc_run.log"
echo "  Raw data:           dc_result.raw"
echo "  Verilog-A model:    $VA_MODEL"
echo ""
echo "============================================"
echo "  Verification Complete"
echo "============================================"
echo ""
echo "Manual check: look at dc_run.log for Newton iteration counts"
echo "Compare key bias points:"
echo "  V=-5.0V: I_dark ≈ 4.299e-07 A"
echo "  V=-3.0V: I_dark ≈ 3.863e-07 A"
echo "  V=-1.0V: I_dark ≈ 2.682e-07 A"
echo "  V= 0.0V: I_dark ≈ 9.304e-08 A"
