#!/usr/bin/env python3
"""T21 — per-design pretender geometry table for all nine runs.

Writes <results>/table_S6_pretenders_v8.csv (one row per committed design) with the
committed and true geometry, self-reported and reference-solver MAE (order 5, and order 7
for seed 42), status, box-edge count, smallest feature, the pixel size P/64, which
generator rule the design violates, and the peak-wavelength shift; plus
<results>/table_S6_crosstab_v8.json with per-run feasible x status Fisher tests
(author-facing, not for the manuscript).
Usage: python3 scripts/pretender_table_v8.py [--results results_v8]
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"": 42, "_s123": 123, "_s777": 777}
TAU = 5.0


def violated(s, p):
    """List of generator rules the design violates (same constants as feasibility_v8.json)."""
    v = []
    if s == "B":
        if p["R_out"] > 0.45 * p["P"]:
            v.append("R_out>0.45P")
        if p["R_in"] > p["R_out"] - 10:
            v.append("R_in>R_out-10")
        if p["R_disk"] > p["R_in"] - 10:
            v.append("R_disk>R_in-10")
    else:
        for w in (("Wx", "Wy", "W2") if s == "A" else ("Wx", "Wy")):
            if p[w] > 0.9 * p["P"]:
                v.append(f"{w}>0.9P")
    return v


def smallest_feature(s, p):
    if s == "A":
        return min(p["Wx"], p["Wy"], p["W2"], p["t1"], p["t2"], p["t_mid"], p["d1"], p["d2"])
    if s == "B":
        return min(p["R_out"] - p["R_in"], p["R_in"] - p["R_disk"], p["t_Cr"])
    return min(p["Wx"], p["Wy"], p["t_Cr"])


def main(results):
    R = ROOT / results
    rows, cross = [], {}
    n_valid = n_pret = 0
    infeas_pret = {s: [0, 0] for s in "ABC"}; infeas_conf = {s: [0, 0] for s in "ABC"}
    for s in "ABC":
        for sx, seed in SEEDS.items():
            inv = np.load(R / f"inverse_{s}{sx}_v8.npz", allow_pickle=True)
            rc = np.load(R / f"rcwa_{s}{sx}_v8.npz", allow_pickle=True)
            o7 = np.load(R / f"rcwa_{s}_v8_o7full.npz", allow_pickle=True) if (seed == 42 and (R / f"rcwa_{s}_v8_o7full.npz").exists()) else None
            names = [str(x) for x in inv["param_names"]]
            keys = ("A_TE", "A_TM") if s == "C" else ("A",)
            lo, hi = inv["design_lo"], inv["design_hi"]
            u = (inv["best_params"] - lo) / (hi - lo)
            ms = np.asarray(rc["mae_surrogate"], float); mr = np.asarray(rc["mae_rcwa"], float)
            failed = rc["failed"].astype(bool); wl = rc["wavelengths"]
            for i in range(len(ms)):
                p = {n: float(inv["best_params"][i, j]) for j, n in enumerate(names)}
                t = {n: float(inv["true_params"][i, j]) for j, n in enumerate(names)}
                bad = failed[i] or not np.isfinite(mr[i])
                status = "degenerate" if bad else ("pretender" if (ms[i] <= TAU and mr[i] > TAU) else ("confirmed" if ms[i] <= TAU else "honest_failure"))
                viol = violated(s, p)
                if not bad:
                    n_valid += 1
                    if status == "pretender":
                        n_pret += 1; infeas_pret[s][1] += 1; infeas_pret[s][0] += bool(viol)
                    elif status == "confirmed":
                        infeas_conf[s][1] += 1; infeas_conf[s][0] += bool(viol)
                shift = []
                for k in keys:
                    if bad or not np.all(np.isfinite(rc[f"A_rcwa_{k}"][i])):
                        shift.append("")
                    else:
                        shift.append(f"{wl[int(np.argmax(rc[f'A_rcwa_{k}'][i]))] - wl[int(np.argmax(rc[f'A_target_{k}'][i]))]:+.0f}")
                row = dict(structure=s, seed=seed, target_no=i + 1, orig_index=int(inv["orig_indices"][i]),
                           mae_surr_pct=round(float(ms[i]), 3),
                           mae_rcwa_o5_pct=("" if bad else round(float(mr[i]), 3)),
                           mae_rcwa_o7_pct=(round(float(o7["mae_rcwa"][i]), 3) if o7 is not None and not o7["failed"][i] else ""),
                           status=status,
                           n_box_edge=int(((u[i] < 0.05) | (u[i] > 0.95)).sum()),
                           smallest_feature_nm=round(smallest_feature(s, p), 1),
                           pixel_nm=round(p["P"] / 64, 2),
                           violated_rules=";".join(viol), n_violations=len(viol),
                           peak_shift_nm="/".join(shift))
                if s == "B":
                    row["twice_R_out_over_P"] = round(2 * p["R_out"] / p["P"], 3)
                for n in names:
                    row[f"committed_{n}"] = round(p[n], 2); row[f"true_{n}"] = round(t[n], 2)
                rows.append(row)
            v = ~failed & np.isfinite(mr)
            pret = (ms <= TAU) & (mr > TAU) & v
            fe = np.array([not violated(s, {n: float(inv["best_params"][i, j]) for j, n in enumerate(names)}) for i in range(len(ms))])
            tab = [[int((~fe & pret).sum()), int((~fe & ~pret & v).sum())], [int((fe & pret).sum()), int((fe & ~pret & v).sum())]]
            cross[f"{s}{sx}"] = dict(table_infeasible_feasible_x_pretender_not=tab,
                                     fisher_p=float(fisher_exact(tab)[1]) if min(map(min, tab)) >= 0 and sum(map(sum, tab)) else None)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("committed_"), k.startswith("true_"), k))
    with open(R / "table_S6_pretenders_v8.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
    (R / "table_S6_crosstab_v8.json").write_text(json.dumps(cross, indent=1))
    print(f"wrote {R/'table_S6_pretenders_v8.csv'}: {len(rows)} rows ({n_valid} valid, {n_pret} pretenders)")
    print("infeasible among pretenders:", {s: f"{infeas_pret[s][0]}/{infeas_pret[s][1]}" for s in "ABC"})
    print("infeasible among confirmed:", {s: f"{infeas_conf[s][0]}/{infeas_conf[s][1]}" for s in "ABC"})
    b = [r for r in rows if r["structure"] == "B" and r["seed"] == 777 and r["target_no"] == 15]
    if b:
        print("B s777 target 15: 2R_out/P =", b[0]["twice_R_out_over_P"], "| status", b[0]["status"], "| surr", b[0]["mae_surr_pct"], "| rcwa", b[0]["mae_rcwa_o5_pct"], "| R_in", b[0]["committed_R_in"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results_v8"); main(ap.parse_args().results)
