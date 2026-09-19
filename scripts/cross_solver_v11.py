#!/usr/bin/env python3
"""§3.11 — cross-solver probe: the pub arm's seed-42 committed geometries re-solved at a
FIXED Fourier order 5 (the as-submitted release's truncation) with everything else held
at the pub settings (same module, complex64, Johnson-Christy, 64x64 grid).  Design by
design this isolates what the truncation alone does to the verdict, which the bundled
legacy-vs-pub contrast of §3.10 cannot.

    python3 scripts/cross_solver_v11.py [--results results_pub]

Reads rcwa_<S>_v8.npz (adaptive) and rcwa_<S>_v8_o5.npz (fixed order 5, written by
rcwa_validate.py --order 5 --tag o5); writes cross_solver_v11.{json,md}.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
TAU = 5.0
ORDER = ["B", "A", "C"]


def main(results):
    R = ROOT / results
    out = dict(tau_pct=TAU, seed=42, per_structure={}, note=(
        "paired on the identical committed geometries and the identical surrogate; only the "
        "Fourier truncation differs (adaptive 9/13/17 -> fixed 5). Single seed, 20 distinct "
        "targets per structure, so no repeated-target dependence; pooled tests are nominal "
        "across structures."))
    rows, dl_all, pa_all, p5_all = [], [], [], []
    for s in ORDER:
        a = np.load(R / f"rcwa_{s}_v8.npz", allow_pickle=True)
        f5 = R / f"rcwa_{s}_v8_o5.npz"
        if not f5.exists():
            out["per_structure"][s] = dict(status="order-5 artifact missing"); continue
        b = np.load(f5, allow_pickle=True)
        assert np.allclose(np.asarray(a["best_params"], float), np.asarray(b["best_params"], float)), f"{s}: geometries differ"
        assert b["order"] == 5 and not bool(b["adaptive"]) and bool(a["adaptive"]), f"{s}: solver settings not as expected"
        assert str(b["sim_dtype"]) == str(a["sim_dtype"]) and str(b["materials"]) == str(a["materials"]), f"{s}: dtype/materials differ"
        ms = np.asarray(a["mae_surrogate"], float); ma = np.asarray(a["mae_rcwa"], float); m5 = np.asarray(b["mae_rcwa"], float)
        v = (~a["failed"].astype(bool)) & (~b["failed"].astype(bool)) & np.isfinite(ma) & np.isfinite(m5)
        pre_a = (ms <= TAU) & (ma > TAU) & v; pre_5 = (ms <= TAU) & (m5 > TAU) & v
        d = (m5 - ma)[v]
        flips = np.where((pre_a != pre_5) & v)[0]
        w = sps.wilcoxon(m5[v], ma[v]) if v.sum() > 5 else None
        disc_5only, disc_aonly = int((pre_5 & ~pre_a).sum()), int((pre_a & ~pre_5).sum())
        mc = sps.binomtest(disc_5only, disc_5only + disc_aonly, 0.5).pvalue if disc_5only + disc_aonly else None
        row = dict(n_valid=int(v.sum()), pretender_adaptive=int(pre_a.sum()), pretender_order5=int(pre_5.sum()),
                   confirmed_adaptive=int(((ms <= TAU) & (ma <= TAU) & v).sum()), confirmed_order5=int(((ms <= TAU) & (m5 <= TAU) & v).sum()),
                   flips_adaptive_confirmed_to_order5_pretender=disc_5only, flips_order5_confirmed_to_adaptive_pretender=disc_aonly,
                   flip_indices=[int(i) for i in flips],
                   flip_mae_adaptive_pct=[round(float(ma[i]), 2) for i in flips], flip_mae_order5_pct=[round(float(m5[i]), 2) for i in flips],
                   median_abs_shift_pp=float(np.median(np.abs(d))), median_shift_pp=float(np.median(d)),
                   max_abs_shift_pp=float(np.abs(d).max()), n_order5_higher=int((d > 0).sum()), n_order5_lower=int((d < 0).sum()),
                   wilcoxon_p=float(w.pvalue) if w else None, mcnemar_p=float(mc) if mc is not None else None,
                   pearson_r_adaptive_vs_order5=float(np.corrcoef(ma[v], m5[v])[0, 1]),
                   median_elapsed_s_order5=float(np.median(np.asarray(b["elapsed"], float)[v])),
                   median_elapsed_s_adaptive=float(np.median(np.asarray(a["elapsed"], float)[v])))
        out["per_structure"][s] = row; rows.append((s, row))
        dl_all.append(d); pa_all.append(pre_a[v]); p5_all.append(pre_5[v])
    if rows:
        d = np.concatenate(dl_all); pa = np.concatenate(pa_all); p5 = np.concatenate(p5_all)
        d5, da = int((p5 & ~pa).sum()), int((pa & ~p5).sum())
        out["pooled"] = dict(n_valid=int(d.size), pretender_adaptive=int(pa.sum()), pretender_order5=int(p5.sum()),
                             flips_to_order5_pretender=d5, flips_to_adaptive_pretender=da,
                             mcnemar_p=float(sps.binomtest(d5, d5 + da, 0.5).pvalue) if d5 + da else None,
                             median_abs_shift_pp=float(np.median(np.abs(d))), median_shift_pp=float(np.median(d)),
                             wilcoxon_p=float(sps.wilcoxon(d).pvalue), n_order5_higher=int((d > 0).sum()), n_order5_lower=int((d < 0).sum()))
    md = ["# Cross-solver probe — pub geometries at fixed order 5 (seed 42)", "", out["note"], "",
          "| S | n | pretenders adaptive → order 5 | flips (conf→pret / pret→conf) | McNemar p | median |ΔMAE| pp | median ΔMAE pp | max |ΔMAE| | Wilcoxon p | r | s/design order 5 vs adaptive |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in rows:
        md.append(f"| {s} | {r['n_valid']} | {r['pretender_adaptive']} → {r['pretender_order5']} | "
                  f"{r['flips_adaptive_confirmed_to_order5_pretender']} / {r['flips_order5_confirmed_to_adaptive_pretender']} | "
                  + (f"{r['mcnemar_p']:.3f}" if r["mcnemar_p"] is not None else "—")
                  + f" | {r['median_abs_shift_pp']:.2f} | {r['median_shift_pp']:+.2f} | {r['max_abs_shift_pp']:.2f} | "
                  + (f"{r['wilcoxon_p']:.4f}" if r["wilcoxon_p"] is not None else "—")
                  + f" | {r['pearson_r_adaptive_vs_order5']:.2f} | {r['median_elapsed_s_order5']:.0f} / {r['median_elapsed_s_adaptive']:.0f} |")
    if "pooled" in out:
        p = out["pooled"]
        md += ["", f"Pooled: pretenders {p['pretender_adaptive']} → {p['pretender_order5']} of {p['n_valid']}; flips {p['flips_to_order5_pretender']} / "
                   f"{p['flips_to_adaptive_pretender']}, McNemar p = {p['mcnemar_p']:.4f}; median |ΔMAE| {p['median_abs_shift_pp']:.2f} pp, "
                   f"median ΔMAE {p['median_shift_pp']:+.2f} pp, order 5 higher on {p['n_order5_higher']} / lower on {p['n_order5_lower']}, Wilcoxon p = {p['wilcoxon_p']:.2e}."]
    (R / "cross_solver_v11.json").write_text(json.dumps(out, indent=2))
    (R / "cross_solver_v11.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results_pub"); main(ap.parse_args().results)
