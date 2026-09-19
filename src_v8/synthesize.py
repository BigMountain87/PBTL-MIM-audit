#!/usr/bin/env python3
"""Cross-structure synthesis — 3-structure table + honest inferential stats (T13).

Reads reliability_{A,B,C}_v8.json (seed 42) from RESULTS and emits
  RESULTS/synthesis_v8_<tag>.{json,md}   (+ synthesis_v8.{json,md} when tag == 'preliminary')
The Paper-1 r convention is a parameter:
  --r 'A=0.72,B=-0.07,C=0.34' --tag preliminary --label '...'
ORDER (descending r) and the H1a/H1b/H1c hypothesis statements are derived from --r;
nothing about the counts or the Fisher tests depends on the convention.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from scipy import stats as sps

import os
from pathlib import Path
# torch-free analysis path (T37): no `common` import
RESULTS = Path(os.environ.get("INVERSETL_RESULTS_DIR",
                              Path(os.environ.get("INVERSETL_ROOT", Path(__file__).resolve().parents[1])) / "results_v8"))

DEFAULT_R = "A=0.72,B=-0.07,C=0.34"      # preliminary pilot r at pre-registration (v8 protocol §1)


def parse_r(spec):
    r = {}
    for item in spec.split(","):
        k, v = item.split("=")
        r[k.strip()] = float(v)
    assert set(r) == {"A", "B", "C"}, r
    return r


def load(s):
    return json.loads((RESULTS / f"reliability_{s}_v8.json").read_text())


def fisher(k1, n1, k2, n2):
    odds, p = sps.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return float(p)


def main(r_spec, tag, label):
    R_CONV = parse_r(r_spec)
    ORDER = sorted(R_CONV, key=lambda s: -R_CONV[s])          # descending r
    o = "".join(ORDER)
    hyp = dict(
        H1a=f"oracle-success rate is non-increasing as r falls ({ORDER[0]} >= {ORDER[1]} >= {ORDER[2]})",
        H1b=f"Delta-flag rate is non-decreasing as r falls ({ORDER[0]} <= {ORDER[1]} <= {ORDER[2]})",
        H1c=f"rho(RCWA MAE, Surr MAE) is non-increasing as r falls ({ORDER[0]} >= {ORDER[1]} >= {ORDER[2]})",
        pooled_order=f"structures ranked by {label}: {' > '.join(f'{s} ({R_CONV[s]:+.2f})' for s in ORDER)}",
    )
    R = {s: load(s) for s in ORDER}
    rows = {}
    for s in ORDER:
        d = R[s]
        rows[s] = dict(
            r_convention=R_CONV[s],
            n_train=d["meta"]["n_train"],
            forward_test_mae=d["meta"]["forward_test_mae_pct"],
            n_valid=d["meta"]["n_valid"],
            oracle_success=d["tier1a"]["rcwa_pass"],
            surrogate_pass=d["tier1a"]["surrogate_pass"],
            pretender=d["tier1a"]["pretender"],
            flagged=d["tier1b"]["flagged"],
            flag_thr=d["tier1b"]["flag_threshold_pct"],
            max_delta=d["tier1b"]["max_delta_pct"],
            max_amp=d["tier1b"]["max_amplification"],
            t1=d["taxonomy"]["t1"], t2=d["taxonomy"]["t2"],
            t3=d["taxonomy"]["t3"], t4=d["taxonomy"]["t4"],
            t2_null=d["taxonomy"]["t2_null"]["analytical"],
            t3_null=d["taxonomy"]["t3_null"]["mean"],
            modes=d["taxonomy"]["mode_count"],
            rho_rcwa_surr=d["discrimination"]["rcwa_vs_surr_PRIMARY"]["rho"],
            rho_p=d["discrimination"]["rcwa_vs_surr_PRIMARY"]["p"],
            rho_ci=d["discrimination"]["rcwa_vs_surr_PRIMARY"]["ci95"],
            rho_delta_rcwa=d["discrimination"]["delta_vs_rcwa_secondary"]["rho"],
            rho_delta_surr=d["discrimination"]["delta_vs_surr_secondary_COUPLED"]["rho"],
            endpoint_spread=d["tier1c"]["endpoint_spread_u_mean"],
        )

    # Fisher exact between every structure pair (adjacent in ORDER, plus the extremes)
    pairs = [(ORDER[0], ORDER[1]), (ORDER[1], ORDER[2]), (ORDER[0], ORDER[2])]
    fishers = {metric: {f"{a}_vs_{b}": fisher(rows[a][metric], rows[a]["n_valid"],
                                             rows[b][metric], rows[b]["n_valid"]) for a, b in pairs}
               for metric in ("oracle_success", "flagged", "pretender")}

    def mono(metric, direction):
        v = [rows[s][metric] for s in ORDER]
        ok = (v[0] >= v[1] >= v[2]) if direction == "noninc" else (v[0] <= v[1] <= v[2])
        return dict(order=ORDER, values=v, monotone=bool(ok))

    h1 = dict(H1a_oracle_success_nonincreasing=mono("oracle_success", "noninc"),
              H1b_flag_rate_nondecreasing=mono("flagged", "nondec"),
              H1c_rho_rcwa_surr_nonincreasing=mono("rho_rcwa_surr", "noninc"))

    out = dict(convention=dict(tag=tag, label=label, r=R_CONV, order=ORDER, hypotheses=hyp),
               per_structure=rows, fisher_exact_p=fishers, hypotheses=h1)
    md = [f"# v8 cross-structure synthesis — convention `{tag}` ({label})", "",
          "| Quantity | " + " | ".join(f"{s} (r={R_CONV[s]:+.2f})" for s in ORDER) + " |",
          "|---|---|---|---|"]

    def row(lbl, fmt, key):
        md.append("| " + lbl + " | " + " | ".join(fmt.format(rows[s][key]) for s in ORDER) + " |")

    def row_count(lbl, key):
        md.append("| " + lbl + " | " + " | ".join(f"{rows[s][key]}/{rows[s]['n_valid']}" for s in ORDER) + " |")

    row("n_train", "{}", "n_train"); row("Forward test MAE (%)", "{:.2f}", "forward_test_mae")
    row_count("Oracle success (tau=5%)", "oracle_success"); row_count("Surrogate-claimed pass", "surrogate_pass")
    row_count("Pretenders", "pretender"); row_count("Delta-flagged (k=2)", "flagged")
    row("Flag threshold (%)", "{:.2f}", "flag_thr"); row("Max Delta (%)", "{:.2f}", "max_delta")
    row("Max amplification", "{:.2f}x", "max_amp")
    for t in ("t1", "t2", "t3", "t4"):
        row(f"{t.upper()} / 20", "{}", t)
    row("Distinct modes", "{}", "modes"); row("rho(RCWA,Surr) PRIMARY", "{:+.2f}", "rho_rcwa_surr")
    row("  p-value", "{:.3f}", "rho_p"); row("rho(Delta,RCWA) secondary", "{:+.2f}", "rho_delta_rcwa")
    row("rho(Delta,Surr) [coupled]", "{:+.2f}", "rho_delta_surr"); row("Restart endpoint spread (u)", "{:.2f}", "endpoint_spread")
    md += ["", "## Fisher exact p-values (pairwise)"]
    for metric, d in fishers.items():
        md.append(f"- **{metric}**: " + ", ".join(f"{k}: p={v:.3f}" for k, v in d.items()))
    md += ["", f"## Pre-specified hypotheses under convention `{tag}` (order {o})"]
    for k, v in h1.items():
        md.append(f"- {k}: values {','.join(ORDER)} = {v['values']} -> {'MONOTONE' if v['monotone'] else 'NOT monotone'}")
    md += ["", "## Hypothesis statements"] + [f"- {k}: {v}" for k, v in hyp.items()]

    for stem in ([f"synthesis_v8_{tag}"] + (["synthesis_v8"] if tag == "preliminary" else [])):
        (RESULTS / f"{stem}.json").write_text(json.dumps(out, indent=2))
        (RESULTS / f"{stem}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", default=DEFAULT_R, help="Paper-1 r convention, e.g. 'A=0.83,B=0.96,C=0.65'")
    ap.add_argument("--tag", default="preliminary")
    ap.add_argument("--label", default="preliminary pilot median r at pre-registration (2026-06-10)")
    a = ap.parse_args()
    main(a.r, a.tag, a.label)
