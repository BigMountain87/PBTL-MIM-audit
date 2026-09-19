#!/bin/bash
# Analysis-only reproduction (no torch, no GPU): re-derive the evidence JSONs and figures from the
# archived per-run artifacts.  Usage: bash scripts/reproduce_analysis_only.sh [results_dir]
set -e
cd "$(dirname "$0")/.."
export INVERSETL_RESULTS_DIR="${1:-results_v8}"
python3 src_v8/synthesize.py --tag preliminary --r A=0.72,B=-0.07,C=0.34 > /dev/null
python3 src_v8/synthesize.py --tag printed --r A=0.83,B=0.96,C=0.65 --label "printed Paper 1 Table 5 median r (PNFA 72(B) 101617)" > /dev/null
python3 src_v8/make_evidence_json.py "$INVERSETL_RESULTS_DIR" > /dev/null
python3 scripts/make_figures_v9.py > /dev/null
echo "analysis-only reproduction done ($INVERSETL_RESULTS_DIR)"
