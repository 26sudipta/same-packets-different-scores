#!/bin/bash
# Progress of the freeze run (03_Code/run_all.sh). From the clone's workspace root:
#   watch -n 30 bash 03_Code/freeze_progress.sh
# Only the run log lines since the latest (re)start of run_all.sh are read, so a
# failure that a restart has since fixed is not reported again.
cd "$(dirname "$0")/.."
F=04_Results/freeze
# step name and expected minutes on the laptop (for the time estimate only): the minutes of
# the first freeze run (28 Oct 2026), and for the tuning, which that run skipped, the tuning
# time recorded in phase5/tuning.csv
W="env 1 hashes 1 p2_inventory 2 p2_counts 1 p2_clock 2 p3 24 p4 3
p5_data_cic 2 p5_data_unsw 1 p5_tune_cic 45 p5_tune_unsw 15 p5_grid_unsw 100 p5_grid_cic 119
p5_analysis 9 p5_sanity 1 p5_tests 1 p5_repro 2 p6 1 p7_extras 11 p7_fold 52 p7_own 51
p7_rq4 42 p7_resid 1 p7_omega 1 p8_flow_R1 7 p8_flow_R5 23 p8_flow_R2 36 p8_flow_R4 53
p8_flow_R3 8 p8_conv_R5 2 p8_conv_R1 6 p8_conv_R2 4 p8_conv_R2f 4 p8_conv_R4 7
p8_decomp_flow 13 p8_decomp_conv 3 p8_decomp_convf 1 p8_verdicts 1 p8_verdicts_fold 1
p9_oneoffs 1 p9_loao 1 numbers 1 figures 1 p9_sanity 105 tests 1 regen 1"
cnt() { find "$1" -type f 2>/dev/null | wc -l; }
now=$(date +%s)
# run log lines since the latest start of run_all.sh
seg=$(awk '/ run_all.sh at commit /{b=""} {b=b $0 "\n"} END{printf "%s", b}' $F/run.log 2>/dev/null)
# the step running now: the last one started and not yet done or failed
read cur cur_d cur_t <<<$(echo "$seg" | awk '$3=="start"{c=$4; d=$1; t=$2}
  ($3=="done" || $3=="FAILED") && $4==c {c=""}
  END{print (c=="" ? "none" : c), d, t}')
step_el=0
[ "$cur" != none ] && step_el=$(( now - $(date -d "$cur_d $cur_t" +%s) ))
# minutes: all steps, steps done, expected minutes of the running step, steps still to come
read total done wcur later <<<$(echo $W | awk -v d="$F/steps" -v c="$cur" '{
  for (i = 1; i < NF; i += 2) { t += $(i+1)
    if (system("test -f " d "/" $i ".done") == 0) s += $(i+1)
    else if ($i == c) w = $(i+1)
    else l += $(i+1) }
  print t, s + 0, w + 0, l + 0 }')
# progress inside the running step: partial minutes and seconds left, from its own output
part=0; left_cur=""; extra=""
case $cur in
  p5_grid_unsw) n=$(cnt 04_Results/phase5/cells/UNSW-NB15); part=$((100*n/150)); extra="$n of 150 cells" ;;
  p5_grid_cic)  n=$(cnt 04_Results/phase5/cells/CICIDS2017); part=$((119*n/210)); extra="$n of 210 cells" ;;
  p7_fold)      n=$(cnt 04_Results/phase7/retrain/fold/cells); part=$((52*n/105)); extra="$n of 105 models" ;;
  p7_own)       n=$(cnt 04_Results/phase7/retrain/own/cells); part=$((51*n/105)); extra="$n of 105 models" ;;
  p9_sanity)
    # log lines "[  3111 s] (a) bootstrap 500 of 1000"; the bootstrap is nearly all of the step
    read k t0 tk <<<$(awk '{ s = $0; sub(/^\[ */, "", s); sub(/ s\].*/, "", s) }
      /bootstrap of 1000 resamples/{t0 = s + 0}
      /\(a\) bootstrap [0-9]+ of 1000/{tk = s + 0; k = $(NF-2) + 0}
      END{print k + 0, t0 + 0, tk + 0}' $F/logs/p9_sanity.log 2>/dev/null)
    if [ "${t0:-0}" -gt 0 ]; then
      extra="bootstrap $k of 1000 resamples"
      part=$((105*k/1000))
      if [ "$k" -gt 0 ]; then
        # rate from the resamples so far; 30 s for the last checks after the bootstrap
        left_cur=$(awk -v k=$k -v t0=$t0 -v tk=$tk -v el=$step_el 'BEGIN{
          r = (tk - t0) / k; x = t0 + 1000 * r + 30 - el; printf "%d", (x < 30 ? 30 : x) }')
      fi
    else
      extra="re-reading the raw CSVs"
    fi ;;
esac
# no step output to go on: expected minutes less the time the step has run, at least 1 min
if [ -z "$left_cur" ]; then
  left_cur=$(( wcur*60 - step_el )); [ $left_cur -lt 60 ] && left_cur=60
  [ "$cur" = none ] && left_cur=0
fi
left=$(( left_cur + later*60 ))
dn=$((done + part)); [ $dn -gt $total ] && dn=$total
pct=$(awk -v a=$dn -v b=$total 'BEGIN{printf "%.1f", 100*a/b}')
bar=$(awk -v a=$dn -v b=$total 'BEGIN{n=int(40*a/b); for(i=0;i<40;i++) printf (i<n?"#":".")}')
echo "Freeze run (step 9.4)   $(date '+%F %H:%M:%S')"
echo
echo "[$bar] $pct %   ($(ls $F/steps 2>/dev/null | wc -l) steps done)"
if [ "$cur" != none ]; then
  printf "Current step: %s, running %dh%02dm%s\n" $cur $((step_el/3600)) $((step_el%3600/60)) "${extra:+ ($extra)}"
fi
if [ -f $F/end_time ]; then
  el=$(( $(cat $F/end_time) - $(cat $F/start_time) ))
  printf "Finished. Took %dh%02dm from the first start\n" $((el/3600)) $((el%3600/60))
elif [ "$cur" != none ]; then
  printf "About %dh%02dm left (running step from its own progress, later steps from expected times)\n" \
    $((left/3600)) $((left%3600/60))
fi
echo
echo "Since the latest start:"
echo "$seg" | tail -4
echo "$seg" | grep -q " FAILED " && echo "A STEP FAILED since the latest start: see the log named above"
pgrep -f "bash 03_Code/run_all.sh" > /dev/null && echo "Status: running" || echo "Status: not running"
echo "Memory: $(free -m | awk '/Mem:/{printf "%d of %d MB available", $7, $2}')"
