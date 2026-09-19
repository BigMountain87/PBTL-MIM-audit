#!/bin/bash
# Full pipeline reproduction for one profile (GPU host).  Statistically (not bitwise) reproducible for
# the training stages; the RCWA stage is bitwise reproducible against the vendored upstream commit.
#   INVERSETL_PROFILE=legacy  (default) : as-submitted release, results_v8/, fixed order 5 (+ order-7 revalidation)
#   INVERSETL_PROFILE=pub               : printed pipeline, results_pub/, adaptive Fourier order
# Requires INVERSETL_UPSTREAM_ROOT pointing at the Paper-1 inputs root (data/raw + results/ checkpoints;
# for pub the checkpoints come from archived_inputs/pub/).  Logs go to logs_<profile>/.
set -e
cd "$(dirname "$0")/.."
export INVERSETL_PROFILE="${INVERSETL_PROFILE:-legacy}"
PY="${PYTHON:-python3}"; L="logs_v8"; [ "$INVERSETL_PROFILE" = pub ] && L="logs_pub"; mkdir -p "$L"
run() { echo "[$(date +%T)] $*"; "$@"; }
run $PY src_v8/selfcheck.py 2>&1 | tee "$L/selfcheck.log"
[ "$INVERSETL_PROFILE" = pub ] && for S in A B C; do run $PY src_v8/pretrain_pub.py --structure $S 2>&1 | tee "$L/pretrain_${S}.log"; done
for SEED in 42 123 777; do for S in A B C; do
  run $PY src_v8/finetune.py --structure $S --seed $SEED 2>&1 | tee "$L/finetune_${S}_${SEED}.log"
  run $PY src_v8/inverse.py  --structure $S --seed $SEED 2>&1 | tee "$L/inverse_${S}_${SEED}.log"
  run $PY src_v8/rcwa_validate.py --structure $S --seed $SEED 2>&1 | tee "$L/rcwa_${S}_${SEED}.log"
done; done
if [ "$INVERSETL_PROFILE" = legacy ]; then
  for S in A B C; do run $PY src_v8/rcwa_validate.py --structure $S --order 7 --tag o7full 2>&1 | tee "$L/rcwa_${S}_o7full.log"; done
fi
for SEED in 42 123 777; do for S in A B C; do run $PY src_v8/reliability.py --structure $S --seed $SEED 2>&1 | tee "$L/reliability_${S}_${SEED}.log"; done; done
for S in A B C; do
  run $PY src_v8/random_baseline.py  --structure $S 2>&1 | tee "$L/random_baseline_${S}.log"
  run $PY src_v8/restart0_validate.py --structure $S 2>&1 | tee "$L/restart0_${S}.log"
done
bash scripts/reproduce_analysis_only.sh "$([ "$INVERSETL_PROFILE" = pub ] && echo results_pub || echo results_v8)"
