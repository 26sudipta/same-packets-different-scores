#!/bin/bash
# Progress of run_phase8_rpfill.sh. Run from anywhere:  watch -n 10 bash 03_Code/phase8_rpfill_progress.sh
cd "$(dirname "$0")/.."
F=04_Results/phase8/rpfill
# step, expected seconds (rough, from the run 2 freeze timings; for the estimate only)
S="p8_flow_R5:300 p8_rp_map:600 p8_conv_R5:200 p8_flow_R3rp:600 p8_conv_R3:600 p8_conv_R3f:600
p8_decomp_flow:700 p8_decomp_flow_R3:300 p8_decomp_conv:180 p8_decomp_convf:40 p8_decomp_conv_R3:200 p8_decomp_conv_R3f:200
p8_verdicts:5 p8_verdicts_fold:5 numbers:10 figures:10 tests:10"
now=$(date +%s); tot=0; don=0; act=0; exp_done=0; n=0; k=0
echo "Phase 8 Exp_RP fill   $(date '+%H:%M:%S')"
echo
for e in $S; do
  s=${e%%:*}; w=${e##*:}; tot=$((tot + w)); n=$((n + 1))
  if [ -f $F/steps/$s.done ]; then
    t=$(cat $F/steps/$s.done); don=$((don + w)); act=$((act + t)); exp_done=$((exp_done + w)); k=$((k + 1))
    printf "  [done]    %-20s %5d s\n" $s $t
  elif grep -q "start $s\$" $F/run.log 2>/dev/null && ! grep -q "FAILED $s" $F/run.log; then
    t0=$(date -d "$(grep "start $s\$" $F/run.log | tail -1 | cut -c1-19)" +%s)
    el=$((now - t0)); part=$(( el < w ? el : w * 9 / 10 )); don=$((don + part))
    printf "  [running] %-20s %5d s so far (expected about %d s)\n" $s $el $w
  else
    printf "  [ ]       %-20s\n" $s
  fi
done
pct=$(awk -v a=$don -v b=$tot 'BEGIN{printf "%.1f", 100*a/b}')
bar=$(awk -v a=$don -v b=$tot 'BEGIN{n=int(40*a/b); s=""; for(i=0;i<40;i++) s=s (i<n?"#":"."); print s}')
echo
echo "Steps $k of $n   [$bar] $pct%"
if [ -f $F/start_time ]; then
  el=$((now - $(cat $F/start_time)))
  r=$(awk -v a=$act -v e=$exp_done 'BEGIN{print (e>0 && a>0) ? a/e : 1}')
  left=$(awk -v t=$tot -v d=$don -v r=$r 'BEGIN{printf "%d", (t-d)*r}')
  printf "Elapsed %dh%02dm, about %dh%02dm left (rough)\n" $((el/3600)) $((el%3600/60)) $((left/3600)) $((left%3600/60))
fi
echo "Latest: $(tail -1 $F/run.log 2>/dev/null)"
grep -q "waiting for Exp_R" $F/run.log 2>/dev/null && ! grep -q "Exp_R Thursday present" $F/run.log && echo "        (waiting for the Exp_R Thursday download)"
pgrep -f "run_phase8_rpfill.sh" > /dev/null && echo "Status: running" || echo "Status: not running"
echo "Memory: $(free -m | awk '/Mem:/{printf "%d of %d MB available", $7, $2}')"
