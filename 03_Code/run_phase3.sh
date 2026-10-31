#!/usr/bin/env bash
# Phase 3 run order (step 3.10). From the workspace root:  bash 03_Code/run_phase3.sh
# One day at a time to stay inside 14 GB of RAM; peak about 3 to 6 GB per day.
set -euo pipefail
PY=.venv/bin/python
LOG=04_Results/phase3/logs
mkdir -p "$LOG"
for d in Monday Tuesday Wednesday Thursday Friday; do
  $PY 03_Code/gap_sweep.py CICIDS2017 "$d" > "$LOG/gap_sweep_CICIDS2017_$d.log" 2>&1
  $PY 03_Code/build_instrument.py CICIDS2017 "$d" > "$LOG/build_CICIDS2017_$d.log" 2>&1
done
for d in 2015-01-22 2015-02-18; do
  $PY 03_Code/gap_sweep.py UNSW-NB15 "$d" > "$LOG/gap_sweep_UNSW-NB15_$d.log" 2>&1
  $PY 03_Code/build_instrument.py UNSW-NB15 "$d" > "$LOG/build_UNSW-NB15_$d.log" 2>&1
done
$PY 03_Code/unsw_clock_offset.py > "$LOG/unsw_clock_offset.log" 2>&1
$PY 03_Code/hand_check.py CICIDS2017 Tuesday
$PY 03_Code/hand_check.py UNSW-NB15 2015-02-18
$PY 03_Code/hand_check.py UNSW-NB15 2015-01-22 --focus cic_unsw
$PY 03_Code/phase3_summary.py
$PY -m unittest discover -s 03_Code/tests
