#!/usr/bin/env bash
# Phase 5 run order (RQ2). From the workspace root:  bash 03_Code/run_phase5.sh [stage]
# Stages: data, tune, grid, analysis, all (default). Needs the frozen Phase 3
# instrument (03_Code/run_phase3.sh). Settings: 04_Results/phase5/settings.json.
# The grid checkpoints per cell: rerunning skips finished cells; delete
# 04_Results/phase5/cells and preds for a clean run. About 12 hours on the laptop.
set -euo pipefail
PY=.venv/bin/python
LOG=04_Results/phase5/logs
STAGE=${1:-all}
mkdir -p "$LOG"
if [[ $STAGE == data || $STAGE == all ]]; then
  $PY 03_Code/phase5_data.py CICIDS2017 > "$LOG/data_CICIDS2017.log" 2>&1
  $PY 03_Code/phase5_data.py UNSW-NB15 > "$LOG/data_UNSW-NB15.log" 2>&1
fi
if [[ $STAGE == tune || $STAGE == grid || $STAGE == all ]]; then
  $PY 03_Code/phase5_grid.py tune CICIDS2017 >> "$LOG/tune.log" 2>&1
  $PY 03_Code/phase5_grid.py tune UNSW-NB15 >> "$LOG/tune.log" 2>&1
fi
if [[ $STAGE == grid || $STAGE == all ]]; then
  $PY 03_Code/phase5_grid.py run UNSW-NB15 >> "$LOG/grid.log" 2>&1
  $PY 03_Code/phase5_grid.py run CICIDS2017 >> "$LOG/grid.log" 2>&1
fi
if [[ $STAGE == analysis || $STAGE == all ]]; then
  $PY 03_Code/phase5_analysis.py > "$LOG/analysis.log" 2>&1
  $PY 03_Code/phase5_sanity.py > "$LOG/sanity.log" 2>&1
  $PY -m unittest 03_Code/tests/test_phase5.py
fi
echo "Exit status 0"
