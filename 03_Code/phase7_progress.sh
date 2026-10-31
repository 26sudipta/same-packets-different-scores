#!/bin/bash
# Phase 7 progress. Run from anywhere:  watch -n 30 bash 03_Code/phase7_progress.sh
cd "$(dirname "$0")/.."
L=04_Results/phase7/logs
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%5.1f%%", (b>0?100*a/b:0)}'; }
bar() { awk -v a="$1" -v b="$2" 'BEGIN{n=int(30*a/b); s=""; for(i=0;i<30;i++) s=s (i<n?"#":"."); printf "[%s]", s}'; }

echo "Phase 7 progress   $(date '+%H:%M:%S')"
echo
# 1. sensitivity re-scoring: 9 variants, 17 benchmark runs (Attempted variant is CICIDS2017 only)
done_s=$(grep -c "bootstrap done" $L/rq4_scoring.log 2>/dev/null)
echo "1. Sensitivity re-scoring   $(bar $done_s 17) $(pct $done_s 17)  ($done_s of 17 runs)"
echo "   latest: $(tail -1 $L/rq4_scoring.log 2>/dev/null)"
echo
# 2. retraining: 2 label variants x 7 arms x 3 models x 5 seeds = 210 cells
nf=$(ls 04_Results/phase7/retrain/fold/cells/CICIDS2017/*/*.csv 2>/dev/null | wc -l)
no=$(ls 04_Results/phase7/retrain/own/cells/CICIDS2017/*/*.csv 2>/dev/null | wc -l)
nt=$((nf + no))
echo "2. Retraining               $(bar $nt 210) $(pct $nt 210)  (fold $nf/105, own labels $no/105)"
P=$(pgrep -f "phase7_retrain.py" | head -1)
if [ -n "$P" ]; then
  st=$(ps -o stat= -p $P | cut -c1)
  [ "$st" = "T" ] && echo "   status: paused until the re-scoring ends (resumes by itself)" \
                  || echo "   status: running"
else
  echo "   status: not running"
fi
echo "   latest: $(tail -1 $L/retrain_cic.log 2>/dev/null | cut -c1-110)"
echo
echo "Memory: $(free -m | awk '/Mem:/{printf "%d of %d MB free", $7, $2}')"
