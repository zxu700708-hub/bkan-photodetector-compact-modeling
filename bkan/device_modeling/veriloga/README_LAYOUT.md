# Verilog-A directory layout

This directory is organized around the final verification artifacts used by the paper.

## Active standalone AC-response readout

- `ge_si_ac_response_symbolic.va`
- `testbench_ac_response_symbolic_ic618.scs`
- `compare_ac_response_symbolic.py`

These files implement and verify the standalone AC-response dB formula readout. This readout is a formula-translation check only; it is not the two-terminal dynamic branch.

Returned Cadence/Spectre evidence is stored under:

- `result/return_ac_response_symbolic_numeric_pass/ac_response_symbolic_numeric_pass_full.tar.gz`

## Active proxy-charge compact-model verification

- `ge_si_photodetector_charge_proxy.va`
- `ge_si_pdet_fixed.va`
- `testbench_dc_ic618.scs`
- `testbench_dc_bias_temp_proxy_ic618.scs`
- `testbench_ac_bias_temp_proxy_ic618.scs`
- `testbench_transient_proxy_ic618.scs`
- `compare_results_proxy.py`
- `compare_ac_charge_proxy.py`
- `summarize_results.py`
- `verification_summary.txt`

Spectre outputs and returned archives are kept under `result/`.

## Supporting scripts and references

- `charge_proxy_models.json`
- `ac_charge_proxy_reference.csv`
- `spectre_reference.txt`
- `generate_veriloga_current.py`
- `generate_ac_charge_proxy_reference.py`
- `run_spectre.sh`
- `run_verification.py`
- `validate_derivatives.py`
- `validate_generalization.py`
- `validate_kan_formulas.py`
- `compare_results.py`
- `kan_to_veriloga.py`

## Legacy / exploratory archive

Older Verilog-A modules, early testbenches, raw DC dumps, and temporary Python cache files have been moved to:

- `_archive_legacy_20260710/`

Nothing in this archive is deleted; it is retained for provenance and recovery.
