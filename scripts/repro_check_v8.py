#!/usr/bin/env python3
"""T05 — RCWA reproducibility spot check (legacy arm).

Compares the archived seed-42 reference-solver results with a re-simulation of the same
committed geometries against the vendored 920b1bd tree, design by design.
Writes <results>/repro_check_v8.json.  Acceptance: every |dMAE| < 0.05 pp.
Usage: python3 scripts/repro_check_v8.py [--results results_v8] [--tag reprocheck]
"""
import argparse, json, platform
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(results, tag):
    R = ROOT / results
    rows, worst = [], 0.0
    meta = {}
    for s in "ABC":
        new = R / f"rcwa_{s}_v8_{tag}.npz"
        if not new.exists():
            print(f"[{s}] {new.name} missing — run rcwa_validate.py --tag {tag} first")
            continue
        a = np.load(R / f"rcwa_{s}_v8.npz", allow_pickle=True)
        b = np.load(new, allow_pickle=True)
        keys = ("A_TE", "A_TM") if s == "C" else ("A",)
        meta.setdefault("rcwa_settings", str(b["rcwa_settings"]) if "rcwa_settings" in b.files else "")
        meta.setdefault("sim_dtype", str(b["sim_dtype"]) if "sim_dtype" in b.files else "")
        meta.setdefault("torcwa_version", str(b["torcwa_version"]) if "torcwa_version" in b.files else "")
        meta.setdefault("upstream_commit", str(b["upstream_commit"]) if "upstream_commit" in b.files else "")
        done = np.where(~b["failed"].astype(bool))[0]
        for i in done:
            dm = float(b["mae_rcwa"][i] - a["mae_rcwa"][i])
            dA = max(float(np.nanmax(np.abs(b[f"A_rcwa_{k}"][i] - a[f"A_rcwa_{k}"][i]))) for k in keys)
            worst = max(worst, abs(dm))
            rows.append(dict(structure=s, design_index=int(i), mae_archived_pct=round(float(a["mae_rcwa"][i]), 6),
                             mae_new_pct=round(float(b["mae_rcwa"][i]), 6), d_mae_pp=round(dm, 8),
                             max_abs_dA=round(dA, 10)))
    out = dict(results_dir=results, tag=tag, n_designs=len(rows), max_abs_d_mae_pp=round(worst, 8),
               tolerance_pp=0.05, pass_=bool(rows and worst < 0.05), host=platform.node(), **meta, designs=rows)
    out["pass"] = out.pop("pass_")
    (R / "repro_check_v8.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {R/'repro_check_v8.json'}: {len(rows)} designs, max |dMAE| = {worst:.2e} pp, pass = {out['pass']}")
    for r in rows:
        print(f"  {r['structure']} #{r['design_index']:2d}  archived {r['mae_archived_pct']:8.4f}  new {r['mae_new_pct']:8.4f}  "
              f"dMAE {r['d_mae_pp']:+.2e} pp  max|dA| {r['max_abs_dA']:.2e}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results_v8"); ap.add_argument("--tag", default="reprocheck")
    a = ap.parse_args(); raise SystemExit(main(a.results, a.tag))
