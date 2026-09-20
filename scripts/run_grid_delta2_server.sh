#!/bin/bash
# T82 follow-on (2026-09-19): (1) Paper 1 Structure-B dataset labels, 30 stratified samples,
# grid 64 (control) + 256, all 100 wavelengths; (2) the 23 surrogate-claimed committed designs
# 1-2 pp from tau not covered by T78's 31, grid 256 only, 2 workers.  GPU is idle: no wait.
export INVERSETL_PROFILE=pub INVERSETL_UPSTREAM_ROOT=${INVERSETL_UPSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/upstream_inputs_pub}
PY=${PY:-python3}   # interpreter with torch + torcwa (the audit host used $HOME/anaconda3/bin/python)
cd "$(dirname "$0")/.."
L=logs_pub; mkdir -p $L results_pub/grid_delta results_pub/grid_delta_paper1B
log(){ echo "[$(date '+%F_%T')] $*" >> $L/grid_delta2_server.log; }
log "stage 1: Paper 1 B dataset, 30 samples, grids 64,256"
$PY scripts/paper1_b_grid_delta.py --npz $HOME/mim_novel/data/raw/struct_B_500_redesign.npz \
   --grids 64,256 --device cuda >> $L/grid_delta_paper1B.log 2>&1
log "stage 1 done (exit $?)"; echo DONE > $L/grid_delta_paper1B_DONE.txt
log "stage 2: 23 near-tau committed designs, grid 256, 2 workers"
QUEUE=scripts/grid_delta_queue2.txt
run_worker(){ local w=$1 n=0
  while read s seed i; do [ -z "$s" ] && continue
    if [ $((n % 2)) -eq $w ]; then log "w$w start $s s$seed i$i"
      $PY scripts/grid_delta_resolve.py $s $seed $i --n-wl 0 --grids 256 --device cuda \
         --out results_pub/grid_delta >> $L/grid_delta2_w$w.log 2>&1
      log "w$w done  $s s$seed i$i (exit $?)"; fi
    n=$((n+1)); done < $QUEUE; }
run_worker 0 & sleep 120; run_worker 1 & wait
log "ALL DONE"; echo DONE > $L/grid_delta2_DONE.txt
