#!/bin/bash
# Cross-solver probe, corrected.  run_pub_extend.sh stage 2 set INVERSETL_TAG=_o5, which
# also retargets the *inputs* (seed_sfx), so rcwa_validate looked for inverse_A_o5_v8.npz
# and died; the stage used `&` + `wait` without checking exit codes and reported "done"
# (2026-09-12 13:45).  The right switch is rcwa_validate's own --tag, which keeps the pub
# inverse artifacts as input and writes rcwa_<S>_v8_o5.npz under cache kind rcwa_o5.
# Usage: setsid nohup ./scripts/run_pub_crosssolver.sh >> logs_pub/crosssolver_driver.log 2>&1 < /dev/null &
set -u
export INVERSETL_PROFILE=pub INVERSETL_UPSTREAM_ROOT=${INVERSETL_UPSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/upstream_inputs_pub}
PY=${PY:-python3}   # interpreter with torch + torcwa (the audit host used $HOME/anaconda3/bin/python)
cd "$(dirname "$0")/.."; R=results_pub; L=logs_pub
log() { echo "[$(date +%F_%T)] $*"; }
for S in B A C; do
  [ -f $R/rcwa_${S}_v8_o5.npz ] && { log "$S already done"; continue; }
  log "cross-solver $S at fixed order 5"
  $PY src_v8/rcwa_validate.py --structure $S --order 5 --tag o5 > $L/rcwa_${S}_o5_pub.log 2>&1 \
    || { echo "ABORT: cross-solver $S failed" | tee -a $L/crosssolver_FAILED.txt; exit 1; }
  tail -n1 $L/rcwa_${S}_o5_pub.log
done
log "cross-solver done"; echo DONE > $L/crosssolver_DONE.txt
