#!/bin/bash
# T70+ — everything that queues behind T55, in value-per-hour order.
#   stage 1  Structure D arm            finetune -> inverse (50 targets) -> oracle -> reliability
#   stage 2  cross-solver probe         the pub committed geometries re-solved at fixed order 5
# The 20 -> 50 target extension is deferred on purpose; see the note after stage 2.
# Waits for logs_pub/gate_DONE.txt so it never competes with T55 for the 16 GB card.
# Usage: setsid nohup ./scripts/run_pub_extend.sh >> logs_pub/extend_driver.log 2>&1 < /dev/null &
set -u
export INVERSETL_PROFILE=pub INVERSETL_UPSTREAM_ROOT=${INVERSETL_UPSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/upstream_inputs_pub}
PY=${PY:-python3}   # interpreter with torch + torcwa (the audit host used $HOME/anaconda3/bin/python)
RPAR=${RPAR:-2}
cd "$(dirname "$0")/.."
R=results_pub; L=logs_pub; mkdir -p $L
log() { echo "[$(date +%F_%T)] $*"; }
die() { echo "ABORT: $*" | tee -a $L/extend_FAILED.txt; exit 1; }

while [ ! -f $L/gate_DONE.txt ]; do log "waiting for T55 (gate_DONE.txt)"; sleep 600; done
log "T55 done; starting the extension"

# ---------------------------------------------------------------- stage 1: Structure D
log "stage 1: Structure D arm (50 targets = the whole held-out split)"
export INVERSETL_N_TARGETS=50
[ -f $R/finetune_D_v8.json ] || $PY src_v8/finetune.py --structure D > $L/finetune_D_pub.log 2>&1 \
    || die "D fine-tune failed"
$PY - <<'PYEOF' || die "D fine-tune did not start from the pretrained checkpoint"
import json, sys
from pathlib import Path
d = json.loads(Path("results_pub/finetune_D_v8.json").read_text())
if d.get("init") != "pretrained" or not d.get("ckpt"):
    sys.exit(f"init={d.get('init')!r} ckpt={d.get('ckpt')!r}")
print(f"  D fine-tune: init={d['init']} test MAE={d['final_test_mae_pct']:.3f} %")
PYEOF
[ -f $R/inverse_D_v8.npz ] || $PY src_v8/inverse.py --structure D > $L/inverse_D_pub.log 2>&1 \
    || die "D inverse failed"
[ -f $R/rcwa_D_v8.npz ] || $PY src_v8/rcwa_validate.py --structure D > $L/rcwa_D_pub.log 2>&1 \
    || die "D oracle failed"
[ -f $R/reliability_D_v8.json ] || $PY src_v8/reliability.py --structure D > $L/reliability_D_pub.log 2>&1 \
    || die "D reliability failed"
log "stage 1 done"; $PY -c "
import json; d=json.load(open('results_pub/reliability_D_v8.json'))
a=d['tier1a']; print('  D: claimed %d  confirmed %d  pretenders %d of %d'
                     % (a['surrogate_pass'], a['rcwa_pass'], a['pretender'], d['meta']['n_valid']))"
unset INVERSETL_N_TARGETS

# ---------------------------------------------------------------- stage 2: cross-solver
# Same module, materials, precision and grid as the pub arm; only the Fourier truncation
# changes (adaptive 9/13/17 -> fixed 5).  That isolates truncation from surrogate quality,
# which the bundled legacy-vs-pub contrast cannot do.  The tag keeps the artifacts and the
# oracle cache separate, and the cache fingerprint would reject a cross-configuration hit
# even if it did not.
log "stage 2: cross-solver probe at fixed order 5 on the pub committed geometries"
for S in B A C; do
  while [ "$(jobs -rp | wc -l)" -ge "$RPAR" ]; do sleep 20; done
  [ -f $R/rcwa_${S}_o5_v8.npz ] || INVERSETL_TAG=_o5 $PY src_v8/rcwa_validate.py \
      --structure $S --order 5 > $L/rcwa_${S}_o5_pub.log 2>&1 &
done; wait
log "stage 2 done"

# Stage 3 -- extending A/B/C from the protocol's 20 targets to the full held-out 50 --
# is deliberately NOT run here. Two reasons. It would overwrite artifacts the frozen
# protocol defines at N = 20, so it needs its own tag, and a tagged run cannot reuse the
# oracle cache (the cache path carries the tag), which puts it at ~91 GPU-hours rather
# than 66. And it is the least valuable of the three: D adds a fourth structure and the
# cross-solver probe adds a decomposition, while more targets only narrows intervals.
# Decide it once stages 1 and 2 have reported.  scripts/run_pub_targets50.sh will run it.
log "stages 1-2 complete; target extension deferred by design (see the comment above)"
echo DONE > $L/extend_DONE.txt
