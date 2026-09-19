#!/bin/bash
# Analysis of the 20 -> 50 target extension, run on the compute host once
# scripts/run_pub_followon.sh has written logs_pub/followon_DONE.txt.
#   1. build the symlink view results_pub/n50 (tagged artifacts under untagged names)
#   2. rerun the unchanged analysis chain on that view: synthesis x3, evidence JSONs,
#      stats supplement, feasibility cross-tab, lookup null, recoverability, detector
#      benchmark (CPU), supplementary tables, figures
# Nothing in results_pub/ proper is touched; every output lands in results_pub/n50/.
# Usage: setsid nohup ./scripts/run_n50_analysis.sh >> logs_pub/n50_analysis.log 2>&1 < /dev/null &
set -u
export PATH=$HOME/anaconda3/bin:$PATH
export INVERSETL_PROFILE=pub INVERSETL_UPSTREAM_ROOT=${INVERSETL_UPSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/upstream_inputs_pub} INVERSETL_DEVICE=cpu
PY=${PY:-python3}   # interpreter with torch + torcwa (the audit host used $HOME/anaconda3/bin/python)
cd "$(dirname "$0")/.."; L=logs_pub
log() { echo "[$(date +%F_%T)] $*"; }
die() { echo "ABORT: $*" | tee -a $L/n50_analysis_FAILED.txt; exit 1; }
while [ ! -f $L/followon_DONE.txt ]; do
  [ -f $L/followon_FAILED.txt ] && die "follow-on driver failed"
  log "waiting for followon_DONE.txt"; sleep 900
done
$PY scripts/make_n50_view.py || die "view (extension incomplete?)"
export INVERSETL_RESULTS_DIR=$(pwd)/results_pub/n50
V=results_pub/n50
$PY src_v8/synthesize.py --tag preliminary --r A=0.72,B=-0.07,C=0.34 > /dev/null || die synthesize-preliminary
$PY src_v8/synthesize.py --tag printed --r A=0.83,B=0.96,C=0.65 --label "published Table 5 median r" > /dev/null || die synthesize-printed
$PY src_v8/synthesize.py --tag tmm_mae --r A=-7.9,B=-8.9,C=-16.9 --label "published operating-band TMM MAE (negated so that larger = better transfer)" > /dev/null || die synthesize-tmm
$PY src_v8/make_evidence_json.py $V > $L/n50_evidence.log 2>&1 || die evidence
$PY src_v8/stats_supplement_v9.py --results $V > $L/n50_stats.log 2>&1 || die stats
$PY scripts/feasibility_crosstab_v8.py --results $V > $L/n50_feasibility.log 2>&1 || die feasibility
$PY scripts/lookup_null_v8.py --profile pub --results $V > $L/n50_lookup.log 2>&1 || log "lookup_null failed (non-fatal): see $L/n50_lookup.log"
$PY src_v8/recoverability.py > $L/n50_recoverability.log 2>&1 || log "recoverability failed (non-fatal)"
$PY src_v8/detector_bench.py > $L/n50_detector.log 2>&1 || log "detector_bench failed (non-fatal)"
$PY scripts/make_supp_table_v9.py --results $V > $L/n50_supp.log 2>&1 || log "supp table failed (non-fatal)"
INVERSETL_FIGURES_DIR=figures_v11_n50 $PY scripts/make_figures_v9.py > $L/n50_figures.log 2>&1 || log "figures failed (non-fatal)"
log "n50 analysis done"; echo DONE > $L/n50_analysis_DONE.txt
