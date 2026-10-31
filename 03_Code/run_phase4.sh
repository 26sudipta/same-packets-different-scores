#!/usr/bin/env bash
# Phase 4 run order (RQ1). From the workspace root:  bash 03_Code/run_phase4.sh
# Needs the frozen Phase 3 instrument in 04_Results/instrument/ (03_Code/run_phase3.sh).
# About 4 minutes on the laptop; peak memory about 4 GB.
set -euo pipefail
PY=.venv/bin/python
LOG=04_Results/phase4/logs
mkdir -p "$LOG"
$PY 03_Code/agreement.py > "$LOG/agreement.log" 2>&1
$PY 03_Code/fig_flows_per_conv.py > "$LOG/fig_flows_per_conv.log" 2>&1
# run_all.sh sets PHASE4_NUMBERS=0: make_numbers.py also reads the Phase 5 to 8 results, which
# a clean run has not written yet at this point; it runs at the end there
if [ "${PHASE4_NUMBERS:-1}" = 1 ]; then
  $PY 03_Code/make_numbers.py > "$LOG/make_numbers.log" 2>&1
fi
$PY -m unittest 03_Code/tests/test_agreement.py
