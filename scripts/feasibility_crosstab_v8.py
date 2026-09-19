#!/usr/bin/env python3
"""T08 — geometric feasibility of committed designs vs pretender status.

Generator constraints (upstream rcwa_struct_{a,b,c}.py sampling clamps):
  A: Wx, Wy, W2 <= 0.9 P    C: Wx, Wy <= 0.9 P
  B: R_out <= 0.45 P,  R_in <= R_out - 10,  R_disk <= R_in - 10
Writes <results>/feasibility_v8.json and <results>/t2_by_param_v8.json.
numpy + scipy only.  Usage: python3 scripts/feasibility_crosstab_v8.py [--results results_v8]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, norm

ROOT = Path(__file__).resolve().parents[1]
TAU, EPS = 5.0, 0.05
SEEDS = {"": 42, "_s123": 123, "_s777": 777}
import os
UPSTREAM = Path(os.environ.get("INVERSETL_UPSTREAM_ROOT", Path.home() / "mim_novel")) / "data/raw"   # Paper-1 inputs root
DATASETS = {"legacy": {"A": "struct_A_vis_500.npz", "B": "struct_B_500.npz", "C": "struct_C_500.npz"},
            "pub": {"A": "struct_A_500_redesign.npz", "B": "struct_B_500_redesign.npz", "C": "struct_C_500_redesign.npz"}}


def feasible(s, P, names):
    c = {n: P[:, i] for i, n in enumerate(names)}
    tol = 1e-9
    if s == "B":
        return (c["R_out"] <= 0.45 * c["P"] + tol) & (c["R_in"] <= c["R_out"] - 10 + tol) & (c["R_disk"] <= c["R_in"] - 10 + tol)
    ok = np.ones(len(P), bool)
    for w in (("Wx", "Wy", "W2") if s == "A" else ("Wx", "Wy")):
        ok &= c[w] <= 0.9 * c["P"] + tol
    return ok


def violations(s, P, names):
    c = {n: P[:, i] for i, n in enumerate(names)}
    if s == "B":
        return {"R_out>0.45P": int((c["R_out"] > 0.45 * c["P"]).sum()), "R_in>R_out-10": int((c["R_in"] > c["R_out"] - 10).sum()),
                "R_disk>R_in-10": int((c["R_disk"] > c["R_in"] - 10).sum()), "wrap_R_in>R_out": int((c["R_in"] > c["R_out"]).sum()),
                "wrap_2R_out>P": int((2 * c["R_out"] > c["P"]).sum())}
    ws = ("Wx", "Wy", "W2") if s == "A" else ("Wx", "Wy")
    out = {f"{w}>0.9P": int((c[w] > 0.9 * c["P"]).sum()) for w in ws}
    out["wrap_any_W>P"] = int(np.any(np.stack([c[w] > c["P"] for w in ws], 1), axis=1).sum())
    return out


def fisher(table):
    (a, b), (c, d) = table
    if min(a + b, c + d, a + c, b + d) == 0:
        return dict(odds_ratio=None, p=1.0)
    o, p = fisher_exact([[a, b], [c, d]])
    return dict(odds_ratio=(None if not np.isfinite(o) else float(o)), p=float(p))


def cmh(tables):
    """Run-stratified Mantel–Haenszel test on 2x2 tables [[a,b],[c,d]] (rows: infeasible / feasible,
    cols: pretender / not).  z = sum(a_i - E_i)/sqrt(sum V_i); MH common odds ratio."""
    num = den = 0.0; n_or = d_or = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        if n == 0 or min(a + b, c + d, a + c, b + d) == 0:
            continue
        r1, c1 = a + b, a + c
        E = r1 * c1 / n
        V = r1 * (n - r1) * c1 * (n - c1) / (n * n * (n - 1)) if n > 1 else 0.0
        num += a - E; den += V
        n_or += a * d / n; d_or += b * c / n
    if den == 0:
        return dict(z=None, p=None, mh_odds_ratio=None)
    z = num / np.sqrt(den)
    return dict(z=float(z), p=float(2 * norm.sf(abs(z))), mh_odds_ratio=(float(n_or / d_or) if d_or > 0 else None), n_strata=len(tables))


def main(results, mc_draws):
    R = ROOT / results
    profile = "pub" if "pub" in results else "legacy"
    out = dict(results_dir=results, profile=profile, tau_pct=TAU, eps=EPS,
               constraints={"A": "Wx,Wy,W2 <= 0.9P", "B": "R_out <= 0.45P, R_in <= R_out-10, R_disk <= R_in-10", "C": "Wx,Wy <= 0.9P"},
               dataset_check={}, runs={}, pooled={}, random_baseline={}, monte_carlo={}, c_w_gt_400={})
    t2out = dict(eps=EPS, runs={}, pooled={})
    for s in "ABC":
        # ---- constraints hold on the generator's own samples
        for prof, files in DATASETS.items():
            f = UPSTREAM / files[s]
            if f.exists():
                z = np.load(f, allow_pickle=True); names = [str(x) for x in z["param_names"]]
                out["dataset_check"][f"{s}_{prof}"] = dict(file=files[s], n=int(len(z["params"])), violations=int((~feasible(s, z["params"], names)).sum()))
        tables, pooled = [], np.zeros((2, 2), int)
        feas_pret = feas_tot = 0
        wflag = {"pret": [0, 0], "succ": [0, 0]}   # C: W>400 flag counts [flagged, n]
        t2_pooled = None
        for sx, seed in SEEDS.items():
            rc = np.load(R / f"rcwa_{s}{sx}_v8.npz", allow_pickle=True)
            inv = np.load(R / f"inverse_{s}{sx}_v8.npz", allow_pickle=True)
            rel = json.load(open(R / f"reliability_{s}{sx}_v8.json"))
            names = [str(x) for x in rc["param_names"]]
            P = rc["best_params"]; ms, mr = rc["mae_surrogate"], rc["mae_rcwa"]
            valid = (~rc["failed"].astype(bool)) & np.isfinite(mr)
            pret = (ms <= TAU) & (mr > TAU) & valid
            succ = (ms <= TAU) & (mr <= TAU) & valid
            fe = feasible(s, P, names)
            tab = [[int((~fe & pret).sum()), int((~fe & ~pret & valid).sum())],
                   [int((fe & pret).sum()), int((fe & ~pret & valid).sum())]]
            tables.append(tab); pooled += np.array(tab)
            feas_pret += int((fe & pret).sum()); feas_tot += int((fe & valid).sum())
            # per-parameter box-edge (T2) counts
            lo, hi = inv["design_lo"], inv["design_hi"]
            u = (inv["best_params"] - lo) / (hi - lo)
            band = (u < EPS) | (u > 1 - EPS)
            t2_any = band.any(axis=1)
            assert int(t2_any.sum()) == rel["taxonomy"]["t2"], f"T2 mismatch {s}{sx}: {t2_any.sum()} vs {rel['taxonomy']['t2']}"
            per_param = {n: dict(pretender=int(band[pret, i].sum()), confirmed=int(band[succ, i].sum()), all=int(band[:, i].sum()),
                                 at_lo=int((u[:, i] < EPS).sum()), at_hi=int((u[:, i] > 1 - EPS).sum())) for i, n in enumerate(names)}
            t2out["runs"][f"{s}{sx}"] = dict(seed=seed, n=int(len(u)), n_valid=int(valid.sum()), t2_total=int(t2_any.sum()),
                                             n_pretender=int(pret.sum()), n_confirmed=int(succ.sum()), per_param=per_param)
            if t2_pooled is None:
                t2_pooled = {n: dict(pretender=0, confirmed=0, all=0) for n in names}
            for n in names:
                for k in ("pretender", "confirmed", "all"):
                    t2_pooled[n][k] += per_param[n][k]
            run = dict(seed=seed, n_valid=int(valid.sum()), n_pretender=int(pret.sum()), n_confirmed=int(succ.sum()),
                       n_feasible=int((fe & valid).sum()), n_infeasible=int((~fe & valid).sum()),
                       table_infeasible_feasible_x_pretender_not=tab, fisher=fisher(tab),
                       pretender_rate_feasible_only=(float((fe & pret).sum() / max((fe & valid).sum(), 1))),
                       pretender_rate_infeasible_only=(float((~fe & pret).sum() / max((~fe & valid).sum(), 1))),
                       violations_all=violations(s, P[valid], names), violations_pretenders=violations(s, P[pret], names),
                       violations_confirmed=violations(s, P[succ], names))
            if s == "C":
                w400 = (P[:, names.index("Wx")] > 400) | (P[:, names.index("Wy")] > 400)
                run["W_gt_400"] = dict(pretender=[int((w400 & pret).sum()), int(pret.sum())], confirmed=[int((w400 & succ).sum()), int(succ.sum())])
                wflag["pret"][0] += int((w400 & pret).sum()); wflag["pret"][1] += int(pret.sum())
                wflag["succ"][0] += int((w400 & succ).sum()); wflag["succ"][1] += int(succ.sum())
            out["runs"][f"{s}{sx}"] = run
        out["pooled"][s] = dict(table=pooled.tolist(), fisher=fisher(pooled.tolist()), cmh_run_stratified=cmh(tables),
                                test_note="nominal: the three runs pooled here share one target set, and stratifying "
                                          "the CMH test by run does not remove the dependence between strata; "
                                          "a design's feasibility is a property of the committed geometry, which "
                                          "differs by seed, so the dependence is partial but not zero",
                                n_pretender=int(pooled[:, 0].sum()), n_valid=int(pooled.sum()),
                                n_infeasible=int(pooled[0].sum()), infeasible_fraction=float(pooled[0].sum() / pooled.sum()),
                                pretender_rate_feasible_only=float(feas_pret / max(feas_tot, 1)),
                                pretender_rate_infeasible_only=float(pooled[0, 0] / max(pooled[0].sum(), 1)))
        t2out["pooled"][s] = t2_pooled
        if s == "C":
            tab = [[wflag["pret"][0], wflag["pret"][1] - wflag["pret"][0]], [wflag["succ"][0], wflag["succ"][1] - wflag["succ"][0]]]
            out["c_w_gt_400"] = dict(table_pretender_confirmed_x_flag_not=tab, fisher=fisher(tab))
        # ---- random baseline (seed 42)
        rb = R / f"random_baseline_{s}_v8.npz"
        if rb.exists():
            z = np.load(rb, allow_pickle=True); names = [str(x) for x in rc["param_names"]]
            ms, mr = z["mae_surrogate"], z["mae_rcwa"]; valid = np.isfinite(mr)
            fe = feasible(s, z["best_params"], names); pret = (ms <= TAU) & (mr > TAU) & valid
            out["random_baseline"][s] = dict(n_valid=int(valid.sum()), n_feasible=int((fe & valid).sum()), n_pretender=int(pret.sum()),
                                             n_pretender_feasible=int((fe & pret).sum()), violations=violations(s, z["best_params"][valid], names))
        # ---- Monte-Carlo: probability that a uniform draw from the design box is infeasible
        st = np.load(R / f"stats_{s}_v8.npz")
        rng = np.random.default_rng(0)
        x = rng.uniform(st["design_lo"], st["design_hi"], (mc_draws, len(st["design_lo"])))
        out["monte_carlo"][s] = dict(draws=mc_draws, p_infeasible=float(1 - feasible(s, x, names).mean()))
    (R / "feasibility_v8.json").write_text(json.dumps(out, indent=1))
    (R / "t2_by_param_v8.json").write_text(json.dumps(t2out, indent=1))
    print(f"wrote {R/'feasibility_v8.json'} and {R/'t2_by_param_v8.json'}")
    for s in "ABC":
        p = out["pooled"][s]
        print(f"{s}: pooled table {p['table']}  infeasible {p['n_infeasible']}/{p['n_valid']} ({p['infeasible_fraction']:.2f})  "
              f"pretender rate feasible-only {p['pretender_rate_feasible_only']:.2f} vs infeasible-only {p['pretender_rate_infeasible_only']:.2f}  "
              f"Fisher p={p['fisher']['p']:.4f}  CMH p={p['cmh_run_stratified']['p']}  MC p_infeasible={out['monte_carlo'][s]['p_infeasible']:.3f}")
    print("dataset violations:", {k: v["violations"] for k, v in out["dataset_check"].items()})
    if out["c_w_gt_400"]:
        print("C W>400 pretender vs confirmed:", out["c_w_gt_400"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v8")
    ap.add_argument("--mc-draws", type=int, default=200000)
    a = ap.parse_args()
    main(a.results, a.mc_draws)
