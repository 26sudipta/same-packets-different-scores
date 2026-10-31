#!/bin/bash
# Phase 8, step 8.2: every per-flow rebuild, one claim after another.
# Start:  setsid nohup systemd-inhibit --what=sleep:idle bash 03_Code/run_phase8.sh &
# Watch:  watch -n 30 bash 03_Code/phase8_progress.sh
# A restart skips every finished cell. R3 waits for the CRiSIS-2022 RP and RD files.
cd "$(dirname "$0")/.."
L=04_Results/phase8/logs
mkdir -p $L
[ -f $L/start_time ] || date +%s > $L/start_time
PY=.venv/bin/python
DL="$(.venv/bin/python -c 'import sys; sys.path.insert(0, "03_Code"); import releases as R; print(R.DATA)')/_downloads/crisis2022_extra"
for c in R1 R5 R2 R4; do
  echo "$(date '+%F %T') start $c" >> $L/run.log
  $PY 03_Code/phase8_flow.py $c >> $L/flow_$c.log 2>&1 || echo "$(date '+%F %T') FAILED $c" >> $L/run.log
done
# R4 failed on the first pass (28 Oct 03:34, missing UNSW-NB15 categories read as NaN,
# fixed in phase8_common.own_label), so its decomposition was not written then
if grep -q "FAILED decomp_flow" $L/rest.log 2>/dev/null && [ ! -f 04_Results/phase8/decomp/R4_gaps.csv ]; then
  echo "$(date '+%F %T') start decomp R4 R5" >> $L/run.log
  $PY 03_Code/phase8_decomp.py R4 R5 >> $L/decomp_flow.log 2>&1 || echo "$(date '+%F %T') FAILED decomp R4 R5" >> $L/run.log
fi
echo "$(date '+%F %T') waiting for the CRiSIS-2022 downloads" >> $L/run.log
until grep -q "ALL DONE\|FAILED" "$DL/gfetch3.log" 2>/dev/null; do sleep 60; done
for c in R3 R3rp; do
  echo "$(date '+%F %T') start $c" >> $L/run.log
  $PY 03_Code/phase8_flow.py $c >> $L/flow_$c.log 2>&1 || echo "$(date '+%F %T') FAILED $c" >> $L/run.log
done
echo "$(date '+%F %T') ALL DONE" >> $L/run.log
