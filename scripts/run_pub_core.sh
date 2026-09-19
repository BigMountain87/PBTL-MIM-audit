#!/bin/bash
# T54 — pub-arm core protocol driver (compute-host). Resume-safe: every stage skips
# work whose artifact already exists; the RCWA stage is cached per design (oracle.py).
#   stage 1  fine-tune seeds 123/777 for A/B/C            (seed 42 done in T52)
#   stage 2  inverse design, 9 runs, up to $PAR concurrent
#   stage 3  RCWA reference solver, 180 designs, $NSHARD workers per run (adaptive order)
#   stage 4  random_baseline + restart0 (seed 42), sharded
#   stage 5  reliability JSONs, 9 runs
# Usage: setsid nohup ./run_pub_core.sh > logs_pub/core_driver.log 2>&1 < /dev/null &
set -u
export INVERSETL_PROFILE=pub INVERSETL_UPSTREAM_ROOT=${INVERSETL_UPSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/upstream_inputs_pub}
PY=${PY:-python3}   # interpreter with torch + torcwa (the audit host used $HOME/anaconda3/bin/python)
PAR=${PAR:-3}          # concurrent python processes for inverse / RCWA workers
NSHARD=${NSHARD:-1}    # RCWA shard workers per (structure, seed); VRAM-bound, keep low
cd "$(dirname "$0")/.."
R=results_pub; L=logs_pub; mkdir -p $L
sfx() { [ "$1" = 42 ] && echo "" || echo "_s$1"; }
log() { echo "[$(date +%F_%T)] $*"; }

log "stage 1: fine-tune seeds 123/777"
for SEED in 123 777; do for S in A B C; do
  [ -f $R/finetune_${S}$(sfx $SEED)_v8.json ] && continue
  $PY src_v8/finetune.py --structure $S --seed $SEED > $L/finetune_${S}$(sfx $SEED)_pub.log 2>&1 || echo "FAIL finetune $S $SEED" >> $L/core_FAILED.txt
done; done

log "stage 2: inverse design (9 runs, $PAR concurrent)"
for SEED in 42 123 777; do for S in A B C; do
  [ -f $R/inverse_${S}$(sfx $SEED)_v8.npz ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
  $PY src_v8/inverse.py --structure $S --seed $SEED > $L/inverse_${S}$(sfx $SEED)_pub.log 2>&1 || echo "FAIL inverse $S $SEED" >> $L/core_FAILED.txt &
done; done; wait

# The reference solver at adaptive order is VRAM-bound (~4-6 GB per solve in complex64 on a
# 16 GB card), so this stage runs at RPAR concurrency, not PAR: three concurrent workers each
# holding an allocator cache deadlocked on 2026-09-06. oracle.wait_for_vram now frees its own
# cache before waiting, and RPAR=2 keeps the peak inside the card.
RPAR=${RPAR:-2}
log "stage 3: RCWA oracle, adaptive order, $NSHARD shard(s) per run, $RPAR concurrent"
for SEED in 42 123 777; do for S in A B C; do
  [ -f $R/rcwa_${S}$(sfx $SEED)_v8.npz ] && continue
  for K in $(seq 0 $((NSHARD-1))); do
    while [ "$(jobs -rp | wc -l)" -ge "$RPAR" ]; do sleep 20; done
    $PY src_v8/rcwa_validate.py --structure $S --seed $SEED --shard $K --nshards $NSHARD > $L/rcwa_${S}$(sfx $SEED)_shard${K}_pub.log 2>&1 || echo "FAIL rcwa $S $SEED shard $K" >> $L/core_FAILED.txt &
  done
done; done; wait
# a worker only assembles the npz when all designs are cached; make sure every run is assembled
for SEED in 42 123 777; do for S in A B C; do
  [ -f $R/rcwa_${S}$(sfx $SEED)_v8.npz ] || $PY src_v8/rcwa_validate.py --structure $S --seed $SEED > $L/rcwa_${S}$(sfx $SEED)_assemble_pub.log 2>&1
done; done

log "stage 4: random baseline + restart-0 controls (seed 42)"
for S in A B C; do
  for K in $(seq 0 $((NSHARD-1))); do
    while [ "$(jobs -rp | wc -l)" -ge "$RPAR" ]; do sleep 20; done
    [ -f $R/random_baseline_${S}_v8.json ] || $PY src_v8/random_baseline.py --structure $S --shard $K --nshards $NSHARD > $L/random_baseline_${S}_shard${K}_pub.log 2>&1 &
    [ -f $R/restart0_${S}_v8.json ] || $PY src_v8/restart0_validate.py --structure $S --shard $K --nshards $NSHARD > $L/restart0_${S}_shard${K}_pub.log 2>&1 &
  done
done; wait
for S in A B C; do
  [ -f $R/random_baseline_${S}_v8.json ] || $PY src_v8/random_baseline.py --structure $S > $L/random_baseline_${S}_assemble_pub.log 2>&1
  [ -f $R/restart0_${S}_v8.json ] || $PY src_v8/restart0_validate.py --structure $S > $L/restart0_${S}_assemble_pub.log 2>&1
done

log "stage 5: reliability"
for SEED in 42 123 777; do for S in A B C; do
  [ -f $R/reliability_${S}$(sfx $SEED)_v8.json ] && continue
  $PY src_v8/reliability.py --structure $S --seed $SEED > $L/reliability_${S}$(sfx $SEED)_pub.log 2>&1 || echo "FAIL reliability $S $SEED" >> $L/core_FAILED.txt
done; done
log "ALL STAGES DONE"; echo DONE > $L/core_DONE.txt
