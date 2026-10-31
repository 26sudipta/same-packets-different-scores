#!/bin/bash
# Phase 8: the Exp_RP steps of run_all.sh (protocol amendment 4), run in the working copy
# once the CRiSIS-2022 Exp_RP files arrived (30 Oct 2026). Same commands and order as the
# Phase 8 section of run_all.sh with RP = 5; cells already written are skipped by the
# scripts, so only the RP cells are new. R3rp waits for Exp_R Thursday (blob 9847d24e).
# Start:  setsid nohup systemd-inhibit --what=sleep:idle bash 03_Code/run_phase8_rpfill.sh > /dev/null 2>&1 &
# Watch:  watch -n 10 bash 03_Code/phase8_rpfill_progress.sh
cd "$(dirname "$0")/.."
set -u
PY=.venv/bin/python
F=04_Results/phase8/rpfill
mkdir -p $F/steps $F/logs
LOG=$F/run.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
step() {   # name, then the command; stops on the first failure
  local n=$1; shift
  [ -f $F/steps/$n.done ] && return 0
  say "start $n"
  local t0=$(date +%s)
  if "$@" > $F/logs/$n.log 2>&1; then
    echo $(( $(date +%s) - t0 )) > $F/steps/$n.done
    say "done $n ($(( $(date +%s) - t0 )) s)"
  else
    say "FAILED $n (see $F/logs/$n.log)"
    exit 1
  fi
}
[ -f $F/start_time ] || date +%s > $F/start_time
DATA=$($PY -c 'import sys; sys.path.insert(0, "03_Code"); import releases as R; print(R.DATA)')
RP=$(ls "$DATA"/CRiSIS2022_RP/*-WorkingHours.csv 2>/dev/null | wc -l)
[ "$RP" -eq 5 ] || { say "FAILED: only $RP of 5 Exp_RP files"; exit 1; }
say "CRiSIS-2022 Exp_RP files present: $RP of 5"
step p8_flow_R5   $PY 03_Code/phase8_flow.py R5
step p8_rp_map    $PY 03_Code/phase8_rp_map.py
step p8_conv_R5   $PY 03_Code/phase8_conv.py R5
RT="$DATA/CRiSIS2022_R/Thursday-WorkingHours.csv"
if [ ! -f $F/steps/p8_flow_R3rp.done ]; then
  say "waiting for Exp_R Thursday"
  until [ -f "$RT" ] && [ "$(git hash-object "$RT")" = 9847d24e7cf5169ccff61d52441a3781c42bb3b4 ]; do sleep 60; done
  say "Exp_R Thursday present, blob id matches"
fi
step p8_flow_R3rp $PY 03_Code/phase8_flow.py R3rp
step p8_conv_R3   $PY 03_Code/phase8_conv.py R3
step p8_conv_R3f  $PY 03_Code/phase8_conv.py R3 --variant fold
# R3 in a process of its own: with RP and R its four arms ran the shared process out of
# memory after R1 and R2 (30 Oct, earlyoom). Each claim draws from its own generator, so
# the split does not change any number.
# No R4 here: the working copy holds the 27 Oct instrument (nine arms), and R4 was
# frozen in run 3 with ten (23efff5); RP does not touch R4. (The 30 Oct pass of this
# step did include R4; its R4 files were put back from git.)
step p8_decomp_flow  $PY 03_Code/phase8_decomp.py R1 R2 R5
step p8_decomp_flow_R3 /usr/bin/time -v $PY 03_Code/phase8_decomp.py R3
step p8_decomp_conv  $PY 03_Code/phase8_decomp.py R1 R2 R5 --stage conv
step p8_decomp_convf $PY 03_Code/phase8_decomp.py R2 --stage conv --variant fold
step p8_decomp_conv_R3  $PY 03_Code/phase8_decomp.py R3 --stage conv
step p8_decomp_conv_R3f $PY 03_Code/phase8_decomp.py R3 --stage conv --variant fold
step p8_verdicts      $PY 03_Code/phase8_verdicts.py --variant sched
step p8_verdicts_fold $PY 03_Code/phase8_verdicts.py --variant fold
step numbers    $PY 03_Code/make_numbers.py
step figures    $PY 03_Code/make_figures.py
step tests      $PY -m unittest discover -s 03_Code/tests
date +%s > $F/end_time
say "ALL DONE"
