#!/bin/bash
# Phase 11 (review response) runs, one heavy job at a time. From the workspace root:
#
#   setsid nohup systemd-inhibit --what=sleep:idle bash 03_Code/run_phase11.sh > /dev/null 2>&1 &
#   watch -n 30 bash 03_Code/phase11_progress.sh
#
# Uses the cache of freeze run 3, so every run reads the arm tables the frozen results came
# from. A restart skips the steps already done (04_Results/phase11/steps/), and inside a step
# the grid code skips finished cells.
cd "$(dirname "$0")/.."
set -u
export NIDS_CACHE_ROOT="${NIDS_CACHE_ROOT:-/home/sudipta/Research/AI-Driven Cyber Threat Detection and Log Analysis/Datasets/_cache_freeze3}"
PY=.venv/bin/python
F=04_Results/phase11
mkdir -p $F/steps $F/logs
LOG=$F/run.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
step() {
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
say "run_phase11.sh at commit $(git rev-parse --short HEAD), caches $NIDS_CACHE_ROOT"
step payload_check   $PY 03_Code/phase11_payload.py check
step payload_run     $PY 03_Code/phase11_payload.py run
step ttl_run         $PY 03_Code/phase11_ttl.py
step models_tune_cic $PY 03_Code/phase11_models.py tune CICIDS2017
step models_run_cic  $PY 03_Code/phase11_models.py run CICIDS2017
step models_tune_unsw $PY 03_Code/phase11_models.py tune UNSW-NB15
step models_run_unsw $PY 03_Code/phase11_models.py run UNSW-NB15
step full_build_cic  $PY 03_Code/phase11_full.py build CICIDS2017
step full_run_cic    $PY 03_Code/phase11_full.py run CICIDS2017
step full_build_unsw $PY 03_Code/phase11_full.py build UNSW-NB15
step full_run_unsw   $PY 03_Code/phase11_full.py run UNSW-NB15
if [ -f 03_Code/phase11_instrument_loo.py ]; then
  step loo_cic       $PY 03_Code/phase11_instrument_loo.py CICIDS2017
  step loo_unsw      $PY 03_Code/phase11_instrument_loo.py UNSW-NB15
fi
say "all steps done"
