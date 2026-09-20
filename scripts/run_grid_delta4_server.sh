#!/bin/bash
# T98b (2026-09-20): the T78 pilot design (A s123 i3, previously 10 wavelengths, log only) re-solved on grid 256 at all 100 wavelengths.
# 10-20-wavelength subsample are re-solved on grid 256 at all 100 wavelengths, 2 workers.
export INVERSETL_PROFILE=pub INVERSETL_UPSTREAM_ROOT=${INVERSETL_UPSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/upstream_inputs_pub}
PY=${PY:-python3}
cd "$(dirname "$0")/.."
L=logs_pub; mkdir -p $L results_pub/grid_delta_full
log(){ echo "[$(date '+%F_%T')] $*" >> $L/grid_delta4_server.log; }
log "stage: pilot design A s123 i3, grid 256, all 100 wavelengths, 2 workers"
QUEUE=scripts/grid_delta_queue4.txt
run_worker(){ local w=$1 n=0
  while read s seed i; do [ -z "$s" ] && continue
    if [ $((n % 2)) -eq $w ]; then log "w$w start $s s$seed i$i"
      $PY scripts/grid_delta_resolve.py $s $seed $i --n-wl 0 --grids 256 --device cuda \
         --out results_pub/grid_delta_full >> $L/grid_delta4_w$w.log 2>&1
      log "w$w done  $s s$seed i$i (exit $?)"; fi
    n=$((n+1)); done < $QUEUE; }
run_worker 0 & sleep 120; run_worker 1 & wait
log "ALL DONE"; echo DONE > $L/grid_delta4_DONE.txt
