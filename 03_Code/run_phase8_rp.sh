#!/bin/bash
# Phase 8, last runner: the R5 cells with Exp_RP as the authors' CRiSIS-2022 (protocol
# amendment 1), then the R5 scoring again and the verdict tables. Starts after
# run_phase8_rest.sh is done.
# Watch:  watch -n 30 bash 03_Code/phase8_progress.sh
cd "$(dirname "$0")/.."
L=04_Results/phase8/logs
PY=.venv/bin/python
say() { echo "$(date '+%F %T') $*" >> $L/rp.log; }
step() {
  local n=$1; shift
  say "start $n"
  "$@" >> $L/$n.log 2>&1 && say "done $n" || say "FAILED $n"
}
until grep -q "ALL DONE" $L/rest.log 2>/dev/null; do sleep 60; done
step flow_R5_rp      $PY 03_Code/phase8_flow.py R5
step conv_R5_rp      $PY 03_Code/phase8_conv.py R5
step decomp_flow_R5  $PY 03_Code/phase8_decomp.py R5
step decomp_conv_R5  $PY 03_Code/phase8_decomp.py R5 --stage conv
step verdicts_sched  $PY 03_Code/phase8_verdicts.py
step verdicts_fold   $PY 03_Code/phase8_verdicts.py --variant fold
say "ALL DONE"
