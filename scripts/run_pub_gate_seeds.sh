#!/bin/bash
# T76 — pub-arm control extension to training seeds 123 and 777 (author decision 2026-09-16,
# on codex advice: the controls that back an abstract-level claim were seed-42 only).
#   stage 1  M0 from-scratch control      INVERSETL_TAG=_m0    finetune->inverse->rcwa->reliability
#   stage 2  feasibility-constrained rerun INVERSETL_TAG=_feas  inverse --feasible->rcwa->reliability
# 2 controls x 2 seeds x 60 designs = 240 adaptive-order solves, ~120 GPU-h, ~60 h wall at RPAR=2.
# Artifact names carry both suffixes, e.g. reliability_A_s123_m0_v8.json (common.seed_sfx).
# Usage: setsid nohup ./scripts/run_pub_gate_seeds.sh > logs_pub/gate_seeds_driver.log 2>&1 < /dev/null &
set -u
export INVERSETL_PROFILE=pub INVERSETL_UPSTREAM_ROOT=${INVERSETL_UPSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/upstream_inputs_pub}
PY=${PY:-python3}   # interpreter with torch + torcwa (the audit host used $HOME/anaconda3/bin/python)
RPAR=${RPAR:-2}
M0_LR=${M0_LR:-1e-3}
SEEDS=${SEEDS:-"123 777"}
cd "$(dirname "$0")/.."
R=results_pub; L=logs_pub; mkdir -p $L
log() { echo "[$(date +%F_%T)] $*"; }
die() { echo "ABORT: $*" | tee -a $L/gate_seeds_FAILED.txt; exit 1; }

# Same provenance guards as run_pub_gate.sh: common.input_sfx() falls back to the untagged
# artifact when a tagged one is missing, so a failed M0 fine-tune would silently turn the
# control into the treatment.
check_m0() {
  $PY - "$1" "$2" <<'PYEOF' || die "M0 identity check failed for $1 seed $2"
import json, sys
from pathlib import Path
s, seed = sys.argv[1], sys.argv[2]
f = Path("results_pub") / f"finetune_{s}_s{seed}_m0_v8.json"
if not f.exists():
    sys.exit(f"{f} is missing")
d = json.loads(f.read_text())
if d.get("init") != "scratch":
    sys.exit(f"{f} records init={d.get('init')!r}, not 'scratch'")
if d.get("ckpt") is not None:
    sys.exit(f"{f} loaded a checkpoint ({d['ckpt']}); this is not a from-scratch control")
if int(d.get("seed", -1)) != int(seed):
    sys.exit(f"{f} records seed={d.get('seed')!r}, expected {seed}")
PYEOF
}

check_feas() {
  $PY - "$1" "$2" <<'PYEOF' || die "feasible-parametrization check failed for $1 seed $2"
import numpy as np, sys
from pathlib import Path
s, seed = sys.argv[1], sys.argv[2]; R = Path("results_pub")
f = R / f"inverse_{s}_s{seed}_feas_v8.npz"
if not f.exists():
    sys.exit(f"{f} is missing")
d = np.load(f, allow_pickle=True)
if not bool(d["feasible"]):
    sys.exit(f"{f} records feasible=False; the constrained rerun did not run constrained")
for later in (f"rcwa_{s}_s{seed}_feas_v8.npz", f"reliability_{s}_s{seed}_feas_v8.json"):
    g = R / later
    if g.exists() and g.stat().st_mtime < f.stat().st_mtime:
        sys.exit(f"{later} is older than {f.name}: stale downstream artifact, delete and rerun")
PYEOF
}

# the seed-123/777 transfer-learned surrogates the feasible rerun reuses must already exist
for SEED in $SEEDS; do for S in A B C; do
  [ -f $R/surrogate_${S}_s${SEED}_v8.pt ] || die "missing $R/surrogate_${S}_s${SEED}_v8.pt (main-arm seed $SEED not run)"
done; done

log "stage 1a: M0 from-scratch fine-tunes (seeds $SEEDS, tag _m0, lr $M0_LR)"
for SEED in $SEEDS; do for S in A B C; do
  [ -f $R/finetune_${S}_s${SEED}_m0_v8.json ] || INVERSETL_TAG=_m0 $PY src_v8/finetune.py \
      --structure $S --seed $SEED --init scratch --lr $M0_LR > $L/finetune_${S}_s${SEED}_m0_pub.log 2>&1 \
      || echo "FAIL finetune m0 $S $SEED" >> $L/gate_seeds_FAILED.txt
  check_m0 $S $SEED
done; done
log "M0 fine-tunes verified as from-scratch"

log "stage 1b: M0 inverse (seeds $SEEDS)"
for SEED in $SEEDS; do for S in A B C; do
  [ -f $R/inverse_${S}_s${SEED}_m0_v8.npz ] || INVERSETL_TAG=_m0 $PY src_v8/inverse.py \
      --structure $S --seed $SEED > $L/inverse_${S}_s${SEED}_m0_pub.log 2>&1 \
      || echo "FAIL inverse m0 $S $SEED" >> $L/gate_seeds_FAILED.txt
done; done

log "stage 2a: feasibility-constrained inverse (seeds $SEEDS, tag _feas)"
for SEED in $SEEDS; do for S in A B C; do
  [ -f $R/inverse_${S}_s${SEED}_feas_v8.npz ] || INVERSETL_TAG=_feas $PY src_v8/inverse.py \
      --structure $S --seed $SEED --feasible > $L/inverse_${S}_s${SEED}_feas_pub.log 2>&1 \
      || echo "FAIL inverse feas $S $SEED" >> $L/gate_seeds_FAILED.txt
  check_feas $S $SEED
done; done
log "feasible reruns verified as constrained"

log "stage 3: oracle on all 12 runs (RPAR=$RPAR)"
for TAG in m0 feas; do for SEED in $SEEDS; do for S in A B C; do
  while [ "$(jobs -rp | wc -l)" -ge "$RPAR" ]; do sleep 20; done
  [ -f $R/rcwa_${S}_s${SEED}_${TAG}_v8.npz ] || INVERSETL_TAG=_${TAG} $PY src_v8/rcwa_validate.py \
      --structure $S --seed $SEED > $L/rcwa_${S}_s${SEED}_${TAG}_pub.log 2>&1 \
      || echo "FAIL rcwa $TAG $S $SEED" >> $L/gate_seeds_FAILED.txt &
done; done; done; wait

log "stage 4: reliability"
for TAG in m0 feas; do for SEED in $SEEDS; do for S in A B C; do
  [ -f $R/reliability_${S}_s${SEED}_${TAG}_v8.json ] || INVERSETL_TAG=_${TAG} $PY src_v8/reliability.py \
      --structure $S --seed $SEED > $L/reliability_${S}_s${SEED}_${TAG}_pub.log 2>&1 \
      || echo "FAIL rel $TAG $S $SEED" >> $L/gate_seeds_FAILED.txt
done; done; done

if [ -f $L/gate_seeds_FAILED.txt ]; then log "FINISHED WITH FAILURES:"; cat $L/gate_seeds_FAILED.txt; exit 1; fi
log "ALL GATE-SEED STAGES DONE"; echo DONE > $L/gate_seeds_DONE.txt
