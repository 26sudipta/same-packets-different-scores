#!/bin/bash
# Phase 8, steps 8.3 and 8.4 after the per-flow rebuilds of run_phase8.sh.
# Start:  setsid nohup systemd-inhibit --what=sleep:idle bash 03_Code/run_phase8_rest.sh &
# Watch:  watch -n 30 bash 03_Code/phase8_progress.sh
# Runs one heavy job at a time: it starts when run_phase8.sh is only waiting for the
# CRiSIS-2022 downloads, and does the R3 parts after run_phase8.sh is done.
cd "$(dirname "$0")/.."
L=04_Results/phase8/logs
PY=.venv/bin/python
say() { echo "$(date '+%F %T') $*" >> $L/rest.log; }
step() {   # name, then the command
  local n=$1; shift
  say "start $n"
  "$@" >> $L/$n.log 2>&1 && say "done $n" || say "FAILED $n"
}
until grep -q "waiting for the CRiSIS\|ALL DONE" $L/run.log 2>/dev/null; do sleep 60; done
step conv_R5_sched $PY 03_Code/phase8_conv.py R5
step conv_R1_sched $PY 03_Code/phase8_conv.py R1
step conv_R2_sched $PY 03_Code/phase8_conv.py R2
step conv_R2_fold  $PY 03_Code/phase8_conv.py R2 --variant fold
step conv_R4_sched $PY 03_Code/phase8_conv.py R4
step decomp_flow   $PY 03_Code/phase8_decomp.py R1 R2 R4 R5
step decomp_conv   $PY 03_Code/phase8_decomp.py R1 R2 R4 R5 --stage conv
step decomp_conv_fold $PY 03_Code/phase8_decomp.py R2 --stage conv --variant fold
say "waiting for run_phase8.sh (R3 needs the CRiSIS-2022 files)"
until grep -q "ALL DONE" $L/run.log 2>/dev/null; do sleep 60; done
step rp_map        $PY 03_Code/phase8_rp_map.py
step conv_R3_sched $PY 03_Code/phase8_conv.py R3
step conv_R3_fold  $PY 03_Code/phase8_conv.py R3 --variant fold
step decomp_flow_R3 $PY 03_Code/phase8_decomp.py R3
step decomp_conv_R3 $PY 03_Code/phase8_decomp.py R3 --stage conv
step decomp_conv_R3_fold $PY 03_Code/phase8_decomp.py R3 --stage conv --variant fold
say "ALL DONE"
