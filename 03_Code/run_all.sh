#!/bin/bash
# Step 9.4, the freeze: every result of Phases 2 to 8, the paper's tables and figures and the
# unit tests, from the raw releases, in one run. From the workspace root of a clean clone:
#
#   NIDS_CACHE_ROOT=$PWD/_freeze_cache setsid nohup systemd-inhibit --what=sleep:idle \
#       bash 03_Code/run_all.sh > /dev/null 2>&1 &
#   watch -n 30 bash 03_Code/freeze_progress.sh
#
# NIDS_CACHE_ROOT must name an empty folder, so that no cache of an earlier run is read
# (03_Code/releases.py). The raw releases are read from NIDS_DATA (default: the Datasets folder
# in releases.py). At the first start 03_Code/freeze_clear.py removes every committed result
# (04_Results, 05_Paper/tables, 05_Paper/figures, apart from a short list of fixed inputs and
# records), so that each one is written again by this run; it refuses unless the checkout is a
# fresh clone with no change. The last step checks that every removed file was written again.
# A restart after a failure skips the steps already done (04_Results/freeze/steps/). One heavy
# job at a time: peak memory is about 8.5 GB (Phase 8 R3). About 12 hours on the laptop
# (i5-11300H, 14 GB).
cd "$(dirname "$0")/.."
set -u
PY=.venv/bin/python
F=04_Results/freeze
mkdir -p $F/steps $F/logs
LOG=$F/run.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
step() {   # name, then the command; stops the run on the first failure
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

if [ ! -f $F/start_time ]; then
  [ -n "${NIDS_CACHE_ROOT:-}" ] || { echo "set NIDS_CACHE_ROOT to an empty folder" >&2; exit 2; }
  mkdir -p "$NIDS_CACHE_ROOT"
  if [ -n "$(ls -A "$NIDS_CACHE_ROOT")" ]; then
    echo "the cache folder is not empty; this would not be a clean run" >&2; exit 2
  fi
  # every committed result goes, so that this run writes each one again (checkpoints included)
  $PY 03_Code/freeze_clear.py > $F/logs/clear.log 2>&1 || { cat $F/logs/clear.log >&2; exit 2; }
  date +%s > $F/start_time
  echo "$NIDS_CACHE_ROOT" > $F/cache_root
fi
export NIDS_CACHE_ROOT=$(cat $F/cache_root)
DATA=$($PY -c 'import sys; sys.path.insert(0, "03_Code"); import releases as R; print(R.DATA)')
say "run_all.sh at commit $(git rev-parse --short HEAD), data $DATA, caches $NIDS_CACHE_ROOT"

# Environment and inputs
step env        bash -c "$PY --version; uv pip freeze --python $PY; git rev-parse HEAD; git status --short"
step hashes     $PY 03_Code/freeze_hashes.py
# Phase 2, release inventory and checks
step p2_inventory $PY 03_Code/inventory.py
step p2_counts    $PY 03_Code/compare_counts.py
step p2_clock     $PY 03_Code/clock_check.py
# Phase 3, the instrument (also runs the unit tests)
step p3         bash 03_Code/run_phase3.sh
# Phase 4, RQ1 (agreement, Figure 1, RQ1 tables)
# (make_numbers.py runs once at the end, when every result it reads exists)
step p4         env PHASE4_NUMBERS=0 bash 03_Code/run_phase4.sh
# Phase 5, RQ2 (the same commands as run_phase5.sh all, one step each)
step p5_data_cic  $PY 03_Code/phase5_data.py CICIDS2017
step p5_data_unsw $PY 03_Code/phase5_data.py UNSW-NB15
step p5_tune_cic  $PY 03_Code/phase5_grid.py tune CICIDS2017
step p5_tune_unsw $PY 03_Code/phase5_grid.py tune UNSW-NB15
step p5_grid_unsw $PY 03_Code/phase5_grid.py run UNSW-NB15
step p5_grid_cic  $PY 03_Code/phase5_grid.py run CICIDS2017
step p5_analysis  $PY 03_Code/phase5_analysis.py
step p5_sanity    $PY 03_Code/phase5_sanity.py
step p5_tests     $PY -m unittest 03_Code/tests/test_phase5.py
step p5_repro     bash 03_Code/phase5_repro.sh
# Phase 6, RQ3
step p6         $PY 03_Code/phase6_rq3.py
# Phase 7, RQ4
step p7_extras  $PY 03_Code/phase7_extras.py
step p7_fold    $PY 03_Code/phase7_retrain.py fold CICIDS2017
step p7_own     $PY 03_Code/phase7_retrain.py own CICIDS2017
step p7_rq4     $PY 03_Code/phase7_rq4.py
step p7_resid   $PY 03_Code/phase7_residual.py
step p7_omega   $PY 03_Code/phase7_omega.py
# Phase 8, RQ5 (the steps of run_phase8.sh, run_phase8_rest.sh, run_phase8_rp.sh and
# run_phase8_r3.sh that produced the frozen results, without their waits). The Exp_RP steps run
# only when the files are in Datasets/CRiSIS2022_RP (protocol amendment 4).
RP=$(ls "$DATA"/CRiSIS2022_RP/*-WorkingHours.csv 2>/dev/null | wc -l)
say "CRiSIS-2022 Exp_RP files present: $RP of 5"
for c in R1 R5 R2 R4 R3; do step p8_flow_$c $PY 03_Code/phase8_flow.py $c; done
if [ "$RP" -eq 5 ]; then
  step p8_flow_R3rp $PY 03_Code/phase8_flow.py R3rp
  step p8_rp_map    $PY 03_Code/phase8_rp_map.py
fi
step p8_conv_R5   $PY 03_Code/phase8_conv.py R5
step p8_conv_R1   $PY 03_Code/phase8_conv.py R1
step p8_conv_R2   $PY 03_Code/phase8_conv.py R2
step p8_conv_R2f  $PY 03_Code/phase8_conv.py R2 --variant fold
step p8_conv_R4   $PY 03_Code/phase8_conv.py R4
if [ "$RP" -eq 5 ]; then
  step p8_conv_R3   $PY 03_Code/phase8_conv.py R3
  step p8_conv_R3f  $PY 03_Code/phase8_conv.py R3 --variant fold
fi
step p8_decomp_flow  $PY 03_Code/phase8_decomp.py R1 R2 R3 R4 R5
step p8_decomp_conv  $PY 03_Code/phase8_decomp.py R1 R2 R4 R5 --stage conv
step p8_decomp_convf $PY 03_Code/phase8_decomp.py R2 --stage conv --variant fold
if [ "$RP" -eq 5 ]; then
  step p8_decomp_conv_R3  $PY 03_Code/phase8_decomp.py R3 --stage conv
  step p8_decomp_conv_R3f $PY 03_Code/phase8_decomp.py R3 --stage conv --variant fold
fi
step p8_verdicts      $PY 03_Code/phase8_verdicts.py --variant sched
step p8_verdicts_fold $PY 03_Code/phase8_verdicts.py --variant fold
# Phase 9: one-off counts quoted in the text (UNSW protocol names, truncation window, near
# repeats) and the leave-one-arm-out shares of RQ2
step p9_oneoffs $PY 03_Code/phase9_oneoffs.py
step p9_loao    $PY 03_Code/phase9_loao.py
# Paper: every table and number macro, the figures, the sanity gate, then all unit tests
step numbers    $PY 03_Code/make_numbers.py
step figures    $PY 03_Code/make_figures.py
# Step 9.3: three drawn results re-derived from the raw CSVs
step p9_sanity  $PY 03_Code/phase9_sanity.py
step tests      $PY -m unittest discover -s 03_Code/tests
# every committed result removed at the start was written again
step regen      $PY 03_Code/freeze_clear.py --check
date +%s > $F/end_time
say "ALL DONE"
