#!/bin/bash
# Follow-on to run_pub_extend.sh (author decision 2026-09-12, after first holding both):
#   stage 0  install the second-pass oracle (c1d9445) and stamp the cache -- explicit migration
#   stage 1  pub-arm reproducibility check: 4 committed geometries per structure (indices
#            0,1,2,6 -- the same choice as the legacy check) re-solved with the cache bypassed
#            (tag 'reprocheck' keys separate entries) and compared bitwise to the archive
#   stage 2  20 -> 50 target extension for the nine base runs under tag _n50: inverse, then
#            oracle, then reliability.  The first 20 designs of a run are the protocol's 20
#            (pick_targets is a prefix, T68); their cache entries are copied under the new
#            tag so they are not re-solved -- the copied entry is accepted only if its
#            fingerprint (geometry, wavelengths, solver identity) matches, else re-solved.
# Waits for logs_pub/extend_DONE.txt.  Aborts on extend_FAILED.txt.
# Usage: setsid nohup ./scripts/run_pub_followon.sh >> logs_pub/followon_driver.log 2>&1 < /dev/null &
set -u
export INVERSETL_PROFILE=pub INVERSETL_UPSTREAM_ROOT=${INVERSETL_UPSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/upstream_inputs_pub}
PY=${PY:-python3}   # interpreter with torch + torcwa (the audit host used $HOME/anaconda3/bin/python)
RPAR=${RPAR:-2}
cd "$(dirname "$0")/.."
R=results_pub; L=logs_pub; C=$R/oracle_cache; mkdir -p $L
log() { echo "[$(date +%F_%T)] $*"; }
die() { echo "ABORT: $*" | tee -a $L/followon_FAILED.txt; exit 1; }

while [ ! -f $L/extend_DONE.txt ]; do
  [ -f $L/extend_FAILED.txt ] && die "run_pub_extend.sh failed; not continuing"
  log "waiting for run_pub_extend.sh (extend_DONE.txt)"; sleep 600
done
log "extension driver done; starting the follow-on"

# ---------------------------------------------------------------- stage 0: oracle + stamp
[ -f src_v8/oracle.py.new ] || die "src_v8/oracle.py.new not staged"
cp src_v8/oracle.py src_v8/oracle.py.pre_c1d9445 && mv src_v8/oracle.py.new src_v8/oracle.py
$PY -c "import sys; sys.path.insert(0,'src_v8'); import oracle; assert hasattr(oracle,'CONTRACT') and 'module_sha256' in open('src_v8/oracle.py').read()" \
    || die "installed oracle.py is not the second-pass module"
$PY scripts/stamp_oracle_cache_v10.py --apply > $L/stamp_apply_$(date +%m%d).log 2>&1 || die "stamp --apply failed"
tail -1 $L/stamp_apply_$(date +%m%d).log
$PY scripts/stamp_oracle_cache_v10.py > $L/stamp_verify_$(date +%m%d).log 2>&1
tail -1 $L/stamp_verify_$(date +%m%d).log
log "stage 0 done"

# ---------------------------------------------------------------- stage 1: repro check
log "stage 1: pub-arm reproducibility check (indices 0,1,2,6 per structure, cache bypassed)"
for S in B C A; do
  while [ "$(jobs -rp | wc -l)" -ge "$RPAR" ]; do sleep 20; done
  [ -f $R/rcwa_${S}_v8_reprocheck.npz ] || $PY src_v8/rcwa_validate.py --structure $S \
      --subset 0,1,2,6 --tag reprocheck > $L/rcwa_${S}_reprocheck_pub.log 2>&1 &
done; wait
$PY scripts/repro_check_v8.py --results $R --tag reprocheck > $L/repro_check_pub.log 2>&1 \
    || die "repro_check_v8.py failed"
tail -3 $L/repro_check_pub.log
log "stage 1 done"

# ---------------------------------------------------------------- stage 2: 20 -> 50 targets
log "stage 2: target extension, tag _n50, nine base runs"
export INVERSETL_TAG=_n50 INVERSETL_N_TARGETS=50
for SEED in 42 123 777; do
  SX=$([ $SEED = 42 ] && echo "" || echo "_s$SEED")
  for S in B A C; do
    [ -f $R/inverse_${S}${SX}_n50_v8.npz ] || $PY src_v8/inverse.py --structure $S --seed $SEED \
        > $L/inverse_${S}${SX}_n50_pub.log 2>&1 || die "inverse $S seed $SEED (n50) failed"
    # reuse the protocol's 20 solves: copy, never move; the fingerprint decides on read
    for i in $(seq -f "%03g" 0 19); do
      [ -f $C/rcwa_${S}${SX}_n50_${i}.npz ] || { [ -f $C/rcwa_${S}${SX}_${i}.npz ] && cp $C/rcwa_${S}${SX}_${i}.npz $C/rcwa_${S}${SX}_n50_${i}.npz; }
    done
  done
done
log "  inverses done; oracle over the 30 new designs per run (RPAR=$RPAR)"
for SEED in 42 123 777; do
  SX=$([ $SEED = 42 ] && echo "" || echo "_s$SEED")
  for S in B A C; do
    while [ "$(jobs -rp | wc -l)" -ge "$RPAR" ]; do sleep 30; done
    [ -f $R/rcwa_${S}${SX}_n50_v8.npz ] || $PY src_v8/rcwa_validate.py --structure $S --seed $SEED \
        > $L/rcwa_${S}${SX}_n50_pub.log 2>&1 &
  done
done; wait
for SEED in 42 123 777; do
  SX=$([ $SEED = 42 ] && echo "" || echo "_s$SEED")
  for S in B A C; do
    [ -f $R/rcwa_${S}${SX}_n50_v8.npz ] || die "oracle $S seed $SEED (n50) produced no artifact"
    [ -f $R/reliability_${S}${SX}_n50_v8.json ] || $PY src_v8/reliability.py --structure $S --seed $SEED \
        > $L/reliability_${S}${SX}_n50_pub.log 2>&1 || die "reliability $S seed $SEED (n50) failed"
  done
done
unset INVERSETL_TAG INVERSETL_N_TARGETS
log "stage 2 done"
echo DONE > $L/followon_DONE.txt
