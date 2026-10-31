#!/bin/bash
# Progress of run_phase11.sh: cells done against cells planned, percent and a rough time left.
#   watch -n 30 bash 03_Code/phase11_progress.sh
cd "$(dirname "$0")/.."
F=04_Results/phase11
count() { find "$1" -name "$2" 2>/dev/null | wc -l; }
tune_rows() { local n=0; [ -f $F/models/tuning.csv ] && n=$(grep -c "$1" $F/models/tuning.csv); echo ${n:-0}; }
# planned units of work, weighted roughly by fit time (one unit = one tree or MLP cell)
p_pay=$(count $F/payload/cells "*.csv");          t_pay=105
p_ttl=$(count $F/ttl/cells "*.csv");              t_ttl=30
p_tc=$(tune_rows CICIDS2017);                     t_tc=56
p_rc=$(count $F/models/cells/CICIDS2017 "*.csv"); t_rc=70
p_tu=$(tune_rows UNSW-NB15);                      t_tu=24
p_ru=$(count $F/models/cells/UNSW-NB15 "*.csv");  t_ru=30
p_full=$(count $F/full/cells "*.csv");            t_full=10
p_loo=$(ls $F/steps 2>/dev/null | grep -c '^loo_');  t_loo=2
printf "%-34s %6s / %-6s\n" "step" "done" "of"
printf "%-34s %6s / %-6s\n" "payload-aware labels (M1)" $p_pay $t_pay
printf "%-34s %6s / %-6s\n" "TTL ablation (m13)" $p_ttl $t_ttl
printf "%-34s %6s / %-6s\n" "LR and kNN tuning, CICIDS2017" $p_tc $t_tc
printf "%-34s %6s / %-6s\n" "LR and kNN cells, CICIDS2017" $p_rc $t_rc
printf "%-34s %6s / %-6s\n" "LR and kNN tuning, UNSW-NB15" $p_tu $t_tu
printf "%-34s %6s / %-6s\n" "LR and kNN cells, UNSW-NB15" $p_ru $t_ru
printf "%-34s %6s / %-6s\n" "full-data check (M5)" $p_full $t_full
printf "%-34s %6s / %-6s\n" "instrument leave-one-out (M4)" $p_loo $t_loo
# rough weights in minutes per unit, from the Phase 5 and 7 timings on this laptop
done_min=$(echo "$p_pay*1.5 + $p_ttl*1.5 + $p_tc*2 + $p_rc*3 + $p_tu*2 + $p_ru*3 + $p_full*6 + $p_loo*45" | bc)
all_min=$(echo "$t_pay*1.5 + $t_ttl*1.5 + $t_tc*2 + $t_rc*3 + $t_tu*2 + $t_ru*3 + $t_full*6 + $t_loo*45" | bc)
pct=$(echo "scale=1; 100*$done_min/$all_min" | bc)
echo
echo "about $pct percent done (weighted by expected fit time)"
if [ -f $F/run.log ]; then
  start=$(stat -c %Y $F/steps 2>/dev/null)
  first=$(head -1 $F/run.log | cut -c1-19)
  el=$(( $(date +%s) - $(date -d "$first" +%s) ))
  if [ "$(echo "$done_min > 0" | bc)" = 1 ]; then
    left=$(echo "$el * ($all_min - $done_min) / $done_min / 60" | bc)
    echo "elapsed $(( el / 60 )) min, about $left min left at the current pace"
  fi
  echo; tail -4 $F/run.log
fi
cur=$(ls -t $F/logs/*.log 2>/dev/null | head -1)
[ -n "$cur" ] && { echo; echo "== $(basename $cur)"; tail -3 "$cur"; }
