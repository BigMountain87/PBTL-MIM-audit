#!/bin/bash
# T55 — pub-arm gate experiments, seed 42 only (author decision 2026-09-07: the pub arm's
# pretender counts are ~10 %, so three seeds of the controls buy no power).
#   stage 1  M0 from-scratch control      INVERSETL_TAG=_m0    finetune->inverse->rcwa->reliability
#   stage 2  feasibility-constrained rerun INVERSETL_TAG=_feas  inverse --feasible->rcwa->reliability
#   stage 3  feasible random baseline      INVERSETL_TAG=_feas  random_baseline --feasible
# ~180 adaptive-order designs = ~87 GPU-hours, ~44 h wall at RPAR=2.
# Waits for the T54 driver to finish so the two never compete for the 16 GB card.
# Usage: setsid nohup ./scripts/run_pub_gate.sh > logs_pub/gate_driver.log 2>&1 < /dev/null &
set -u
export INVERSETL_PROFILE=pub INVERSETL_UPSTREAM_ROOT=${INVERSETL_UPSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/upstream_inputs_pub}
PY=${PY:-python3}   # interpreter with torch + torcwa (the audit host used $HOME/anaconda3/bin/python)
RPAR=${RPAR:-2}
M0_LR=${M0_LR:-1e-3}     # upstream_cb486b5/src/training/trainer.py:42 default
cd "$(dirname "$0")/.."
R=results_pub; L=logs_pub; mkdir -p $L
log() { echo "[$(date +%F_%T)] $*"; }

# --- control-identity guards (codex review 2026-09-09, item 5) --------------------
# common.input_sfx() falls back to the untagged artifacts when a tagged one is missing.
# If an M0 fine-tune failed, inverse.py under INVERSETL_TAG=_m0 would silently load the
# transfer-learned surrogate and write it out as inverse_*_m0_v8.npz: a "control" that is
# the treatment. Every stage therefore verifies provenance before the next one starts.
die() { echo "ABORT: $*" | tee -a $L/gate_FAILED.txt; exit 1; }

check_m0() {
  $PY - "$1" <<'PYEOF' || die "M0 identity check failed for $1"
import json, sys
from pathlib import Path
s = sys.argv[1]
f = Path("results_pub") / f"finetune_{s}_m0_v8.json"
if not f.exists():
    sys.exit(f"{f} is missing")
d = json.loads(f.read_text())
if d.get("init") != "scratch":
    sys.exit(f"{f} records init={d.get('init')!r}, not 'scratch'")
if d.get("ckpt") is not None:
    sys.exit(f"{f} loaded a checkpoint ({d['ckpt']}); this is not a from-scratch control")
PYEOF
}

check_feas() {
  $PY - "$1" <<'PYEOF' || die "feasible-parametrization check failed for $1"
import numpy as np, sys
from pathlib import Path
s = sys.argv[1]; R = Path("results_pub")
f = R / f"inverse_{s}_feas_v8.npz"
if not f.exists():
    sys.exit(f"{f} is missing")
d = np.load(f, allow_pickle=True)
if not bool(d["feasible"]):
    sys.exit(f"{f} records feasible=False; the constrained rerun did not run constrained")
# a resumed run must not accept downstream artifacts that predate this inverse file
for later in (f"rcwa_{s}_feas_v8.npz", f"reliability_{s}_feas_v8.json"):
    g = R / later
    if g.exists() and g.stat().st_mtime < f.stat().st_mtime:
        sys.exit(f"{later} is older than {f.name}: stale downstream artifact, delete and rerun")
PYEOF
}

check_feas_random() {
  $PY - "$1" <<'PYEOF' || die "feasible random-baseline check failed for $1"
import numpy as np, sys
from pathlib import Path
s = sys.argv[1]
f = Path("results_pub") / f"random_baseline_{s}_feas_v8.npz"
if not f.exists():
    sys.exit(f"{f} is missing")
d = np.load(f, allow_pickle=True)
if "feasible" not in d.files or not bool(d["feasible"]):
    sys.exit(f"{f} does not record feasible=True; --feasible was not in effect")
PYEOF
}


while [ ! -f $L/core_DONE.txt ]; do log "waiting for T54 (core_DONE.txt)"; sleep 600; done
log "T54 done; starting the gate experiments"

log "stage 1: M0 from-scratch control (seed 42, tag _m0, lr $M0_LR)"
for S in A B C; do
  [ -f $R/finetune_${S}_m0_v8.json ] || INVERSETL_TAG=_m0 $PY src_v8/finetune.py \
      --structure $S --init scratch --lr $M0_LR > $L/finetune_${S}_m0_pub.log 2>&1 \
      || echo "FAIL finetune m0 $S" >> $L/gate_FAILED.txt
done
for S in A B C; do check_m0 $S; done
log "M0 fine-tunes verified as from-scratch"
for S in A B C; do
  [ -f $R/inverse_${S}_m0_v8.npz ] || INVERSETL_TAG=_m0 $PY src_v8/inverse.py \
      --structure $S > $L/inverse_${S}_m0_pub.log 2>&1 || echo "FAIL inverse m0 $S" >> $L/gate_FAILED.txt
done
for S in A B C; do
  while [ "$(jobs -rp | wc -l)" -ge "$RPAR" ]; do sleep 20; done
  [ -f $R/rcwa_${S}_m0_v8.npz ] || INVERSETL_TAG=_m0 $PY src_v8/rcwa_validate.py \
      --structure $S > $L/rcwa_${S}_m0_pub.log 2>&1 || echo "FAIL rcwa m0 $S" >> $L/gate_FAILED.txt &
done; wait
for S in A B C; do
  [ -f $R/reliability_${S}_m0_v8.json ] || INVERSETL_TAG=_m0 $PY src_v8/reliability.py \
      --structure $S > $L/reliability_${S}_m0_pub.log 2>&1 || echo "FAIL rel m0 $S" >> $L/gate_FAILED.txt
done

log "stage 2: feasibility-constrained rerun (seed 42, tag _feas)"
for S in A B C; do
  [ -f $R/inverse_${S}_feas_v8.npz ] || INVERSETL_TAG=_feas $PY src_v8/inverse.py \
      --structure $S --feasible > $L/inverse_${S}_feas_pub.log 2>&1 || echo "FAIL inverse feas $S" >> $L/gate_FAILED.txt
done
for S in A B C; do check_feas $S; done
log "feasible reruns verified as constrained; starting their oracle"
for S in A B C; do
  while [ "$(jobs -rp | wc -l)" -ge "$RPAR" ]; do sleep 20; done
  [ -f $R/rcwa_${S}_feas_v8.npz ] || INVERSETL_TAG=_feas $PY src_v8/rcwa_validate.py \
      --structure $S > $L/rcwa_${S}_feas_pub.log 2>&1 || echo "FAIL rcwa feas $S" >> $L/gate_FAILED.txt &
done; wait
for S in A B C; do
  [ -f $R/reliability_${S}_feas_v8.json ] || INVERSETL_TAG=_feas $PY src_v8/reliability.py \
      --structure $S > $L/reliability_${S}_feas_pub.log 2>&1 || echo "FAIL rel feas $S" >> $L/gate_FAILED.txt
done

log "stage 3: feasible random baseline (seed 42, tag _feas)"
for S in A B C; do
  while [ "$(jobs -rp | wc -l)" -ge "$RPAR" ]; do sleep 20; done
  [ -f $R/random_baseline_${S}_feas_v8.json ] || INVERSETL_TAG=_feas $PY src_v8/random_baseline.py \
      --structure $S --feasible > $L/random_baseline_${S}_feas_pub.log 2>&1 \
      || echo "FAIL random feas $S" >> $L/gate_FAILED.txt &
done; wait

for S in A B C; do check_feas_random $S; done
log "feasible random baselines verified as constrained"

log "M0 forward test MAE against the printed Table 1-3 M0 values (A 2.64+-0.09, B 2.05+-0.07, C 2.12):"
for S in A B C; do
  [ -f $R/finetune_${S}_m0_v8.json ] && $PY -c "
import json;d=json.load(open('$R/finetune_${S}_m0_v8.json'));print('  $S M0 test MAE %.3f %%' % d['final_test_mae_pct'])"
done
log "ALL GATE STAGES DONE"; echo DONE > $L/gate_DONE.txt
