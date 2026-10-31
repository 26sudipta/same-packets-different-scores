#!/bin/bash
# Phase 8 progress. Run from anywhere:  watch -n 30 bash 03_Code/phase8_progress.sh
cd "$(dirname "$0")/.."
L=04_Results/phase8/logs
PR=04_Results/phase8/preds
CV=04_Results/phase8/conv/preds
DL="/home/sudipta/Research/AI-Driven Cyber Threat Detection and Log Analysis/Datasets"
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%5.1f%%", (b>0?100*a/b:0)}'; }
bar() { awk -v a="$1" -v b="$2" 'BEGIN{n=int(30*a/b); s=""; for(i=0;i<30;i++) s=s (i<n?"#":"."); printf "[%s]", s}'; }
cnt() { find "$1" -name "*.parquet" 2>/dev/null | wc -l; }

echo "Phase 8 (RQ5)   $(date '+%H:%M:%S')"
echo
echo "8.2 per-flow rebuilds (run_phase8.sh)"
# cells: R1 2 arms x (LDA 1 + DT 5 + RF 5); R5 4 days x 6 arms x 5; R2 2 x 5;
# R4 3 arms x 2 tasks x 5; R3 RD, RPD, R and RP x 3 models x 5
# weights: rough minutes per cell on this laptop, for the time estimate only
total_w=0; done_w=0
for spec in "R1 22 1" "R5 120 1" "R2 10 10" "R4 30 5" "R3 60 2"; do
  set -- $spec
  n=$(cnt $PR/$1); [ "$n" -gt "$2" ] && n=$2
  printf "  %-4s %s %s  (%d of %d cells)\n" $1 "$(bar $n $2)" "$(pct $n $2)" $n $2
  total_w=$((total_w + $2 * $3)); done_w=$((done_w + n * $3))
done
echo
echo "8.4 conversation footing (run_phase8_rest.sh)"
# cells: R1 2 x 3 x 5; R2 2 x 5 (+ fold); R3 2 x 3 x 5 (+ fold); R4 2 x 2 x 5; R5 12 pairs x 2 x 5
for spec in "R1 30 1" "R2 20 4" "R3 60 1" "R4 20 4" "R5 120 0"; do
  set -- $spec
  n=$(cnt $CV/$1); [ "$n" -gt "$2" ] && n=$2
  printf "  %-4s %s %s  (%d of %d cells)\n" $1 "$(bar $n $2)" "$(pct $n $2)" $n $2
  total_w=$((total_w + $2 * $3)); done_w=$((done_w + n * $3))
done
nd=$(grep -c " done decomp" $L/rest.log 2>/dev/null); nd=${nd:-0}
nd=$((nd + $(grep -c " done decomp\| done verdicts" $L/rp.log 2>/dev/null || echo 0)))
echo "  decompositions, bootstraps, verdicts: $nd of 10 runs done"
total_w=$((total_w + 100)); done_w=$((done_w + nd * 10))
echo
echo "All  $(bar $done_w $total_w) $(pct $done_w $total_w)  (weighted by expected run time)"
if [ -f $L/start_time ] && [ $done_w -gt 0 ]; then
  el=$(( $(date +%s) - $(cat $L/start_time) ))
  left=$(awk -v e=$el -v d=$done_w -v t=$total_w 'BEGIN{printf "%d", e*(t-d)/d}')
  printf "Elapsed %dh%02dm, about %dh%02dm left (rough; R3 also waits for the downloads)\n" \
    $((el/3600)) $((el%3600/60)) $((left/3600)) $((left%3600/60))
fi
echo
got=$(ls "$DL"/CRiSIS2022_RP/*-WorkingHours.csv "$DL"/CRiSIS2022_R/Thursday-WorkingHours.csv 2>/dev/null | wc -l)
mb=$(du -cm "$DL"/CRiSIS2022_RP/*.part 2>/dev/null | tail -1 | cut -f1)
echo "CRiSIS-2022 Exp_RP: $got of 6 files ready (one at a time, about 10 min each); current file ${mb:-0} MB"
echo "        $(tail -1 "$DL"/_downloads/crisis2022_extra/gfetch6.log 2>/dev/null)"
echo "Latest: $(tail -1 $L/run.log 2>/dev/null)"
echo "        $(tail -1 $L/rest.log 2>/dev/null)"
echo "        $(tail -1 $L/rp.log 2>/dev/null)"
pgrep -f "phase8_(flow|conv|decomp|rp_map).py" > /dev/null && echo "Status: running" \
  || echo "Status: not running"
echo "Memory: $(free -m | awk '/Mem:/{printf "%d of %d MB free", $7, $2}')"
