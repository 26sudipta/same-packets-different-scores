#!/bin/bash
# Phase 8: R3 (RD vs RPD) after the pandas 3 text-column fix of 28 Oct, then its
# decomposition and the verdict tables again. The Exp_RP files were not obtained
# (gfetch6.log), so R3rp, the RP map and the 8.4 R3 cells stay out (protocol amendment 1).
# Start:  setsid nohup systemd-inhibit --what=sleep:idle bash 03_Code/run_phase8_r3.sh &
cd "$(dirname "$0")/.."
L=04_Results/phase8/logs
PY=.venv/bin/python
step() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $L/r3.log
  "$@" >> $L/$name.log 2>&1 && echo "$(date '+%F %T') done $name" >> $L/r3.log \
    || echo "$(date '+%F %T') FAILED $name" >> $L/r3.log
}
step flow_R3          $PY 03_Code/phase8_flow.py R3
step decomp_flow_R3   $PY 03_Code/phase8_decomp.py R3
step verdicts_sched   $PY 03_Code/phase8_verdicts.py --variant sched
step verdicts_fold    $PY 03_Code/phase8_verdicts.py --variant fold
echo "$(date '+%F %T') ALL DONE" >> $L/r3.log
