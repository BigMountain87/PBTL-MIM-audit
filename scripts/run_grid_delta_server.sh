#!/bin/bash
# Grid-fidelity follow-on for docs/notes_rcwa_grid_truncation_v11.md.
# Waits for T76 (logs_pub/gate_seeds_DONE.txt), then re-solves 20 committed designs on a
# 256 x 256 raster at the pinned adaptive order, full 100-wavelength grid, and compares each
# to its archived grid-64 GPU spectrum.  Two workers (A at N = 17 peaks ~5 GB each).
export INVERSETL_PROFILE=pub INVERSETL_UPSTREAM_ROOT=${INVERSETL_UPSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/upstream_inputs_pub}
PY=${PY:-python3}   # interpreter with torch + torcwa (the audit host used $HOME/anaconda3/bin/python)
cd "$(dirname "$0")/.."
L=logs_pub; mkdir -p $L results_pub/grid_delta
log(){ echo "[$(date '+%F_%T')] $*" >> $L/grid_delta_server.log; }
log "waiting for $L/gate_seeds_DONE.txt"
while [ ! -f $L/gate_seeds_DONE.txt ]; do sleep 300; done
log "T76 done; starting 20-design grid-256 queue, 2 workers"
QUEUE=scripts/grid_delta_queue.txt
run_worker(){   # $1 = worker id (0/1): takes every other line of the queue
  local w=$1 n=0
  while read s seed i; do
    [ -z "$s" ] && continue
    if [ $((n % 2)) -eq $w ]; then
      log "w$w start $s s$seed i$i"
      $PY scripts/grid_delta_resolve.py $s $seed $i --n-wl 0 --grids 256 --device cuda \
         --out results_pub/grid_delta >> $L/grid_delta_w$w.log 2>&1
      log "w$w done  $s s$seed i$i (exit $?)"
    fi
    n=$((n+1))
  done < $QUEUE
}
run_worker 0 &
sleep 120     # let worker 0 allocate before worker 1 starts
run_worker 1 &
wait
log "ALL GRID-DELTA SOLVES DONE"; echo DONE > $L/grid_delta_DONE.txt
