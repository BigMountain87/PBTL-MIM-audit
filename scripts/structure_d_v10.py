#!/usr/bin/env python3
"""Structure D — the held-out test of the transferability diagnostic.

    python3 scripts/structure_d_v10.py [--results results_pub]

D did not exist when the protocol was frozen, so it takes no part in H1a-c.  It is used
instead for something the pre-specified tests cannot give: a structure whose diagnostic
value was measured before its inverse audit ran, with the prediction recorded in
docs/structure_D_prespecification.md.  This script scores that prediction and places D in
the four-point orderings.  Writes <results>/structure_d_v10.{json,md}.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
TAU = 5.0
# fidelity of all four structures under the printed estimator (structure_D/fidelity_D.py;
# A, B and C reproduce the companion's printed Table 5)
DIAG = {"B": dict(r=0.962, mae=8.93), "D": dict(r=0.905, mae=10.51),
        "A": dict(r=0.830, mae=7.94), "C": dict(r=0.647, mae=16.94)}
# Governing rule = the ORIGINAL band as written (5-15 % = 3 to 7 of 50), arithmetic
# corrected; the wider union band is secondary only.  16.9 % was C's TMM MAE, not its
# pretender rate -- the falsification bounds are A's 3.4 % and B's 15.0 %.  See the
# clarification at the end of the pre-specification, which also states that 42 of 50
# oracle results existed on disk when it was written.
PREDICTION = dict(band_pct=(5.0, 15.0), secondary_union_band_pct=(3.4, 15.0),
                  falsified_below_pct=3.4, falsified_above_pct=15.0,
                  source="docs/structure_D_prespecification.md (2026-09-10), clarification "
                         "of 2026-09-12 08:55 (42/50 oracle results existed, not inspected)")


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def rate(R, s, seeds=("",)):
    """Counts for one structure, pooled over the given seed suffixes.  A, B and C are
    pooled over their three training seeds to match Sections 3.2 and 3.4; D has one."""
    ms, mr, okm = [], [], []
    for sx in seeds:
        f = R / f"rcwa_{s}{sx}_v8.npz"
        if not f.exists():
            return None
        d = np.load(f, allow_pickle=True)
        ms.append(d["mae_surrogate"]); mr.append(d["mae_rcwa"]); okm.append(~d["failed"])
    ms, mr, ok = np.concatenate(ms), np.concatenate(mr), np.concatenate(okm)
    claimed = (ms <= TAU) & ok
    pre = claimed & (mr > TAU)
    return dict(n_valid=int(ok.sum()), claimed=int(claimed.sum()), pretender=int(pre.sum()),
                confirmed=int((claimed & (mr <= TAU)).sum()), seeds=list(seeds),
                rate_pct=100 * int(pre.sum()) / max(int(claimed.sum()), 1),
                ci95=wilson(int(pre.sum()), int(claimed.sum())),
                median_delta_pp=float(np.median((mr - ms)[ok])),
                max_delta_pp=float((mr - ms)[ok].max()))


def trend(order, counts):
    """Cochran-Armitage along `order`, scored by the diagnostic value itself."""
    x = np.array([order[s] for s in counts], float)
    k = np.array([counts[s]["pretender"] for s in counts], float)
    n = np.array([counts[s]["claimed"] for s in counts], float)
    pbar = k.sum() / n.sum()
    xbar = (n * x).sum() / n.sum()
    T = ((k - n * pbar) * x).sum()
    V = pbar * (1 - pbar) * (n * (x - xbar) ** 2).sum()
    if V <= 0:
        return None
    z = T / math.sqrt(V)
    return dict(z=float(z), p=float(2 * (1 - sps.norm.cdf(abs(z)))))


def main(results):
    R = ROOT / results
    d = rate(R, "D")
    abc = {s: rate(R, s, ("", "_s123", "_s777")) for s in "BAC"}
    out = dict(tau_pct=TAU, diagnostic=DIAG, prediction=PREDICTION,
               per_structure_pooled_3seed={s: v for s, v in abc.items() if v},
               structure_D=d)
    if d is None:
        out["status"] = "Structure D oracle not run yet"
        (R / "structure_d_v10.json").write_text(json.dumps(out, indent=2))
        print(f"Structure D has no rcwa artifact in {results} yet; nothing to score.")
        return

    # post hoc, mirroring §3.8: are D's pretenders the generator-infeasible geometries?
    # (L <= 0.9 P and w <= L, common.feasible_mask; the pub profile is needed for STRUCTS["D"])
    try:
        import os, sys
        os.environ.setdefault("INVERSETL_PROFILE", "pub")
        sys.path.insert(0, str(ROOT / "src_v8"))
        from common import feasible_mask
        inv = np.load(R / "inverse_D_v8.npz", allow_pickle=True); rel = json.load(open(R / "reliability_D_v8.json"))
        fm = feasible_mask("D", np.asarray(inv["best_params"], float))
        pret = np.zeros(len(fm), bool); pret[rel["tier1a"]["pretender_indices"]] = True
        tab = [[int((~fm & pret).sum()), int((~fm & ~pret).sum())], [int((fm & pret).sum()), int((fm & ~pret).sum())]]
        out["feasibility_D"] = dict(rule="L <= 0.9 P and w <= L", n_infeasible=int((~fm).sum()), n_valid=int(len(fm)),
                                    table_infeasible_feasible_x_pretender_not=tab,
                                    pretenders_infeasible=int((~fm & pret).sum()), pretenders_feasible=int((fm & pret).sum()),
                                    fisher_p=float(sps.fisher_exact(tab)[1]), note="post hoc, single seed, 50 distinct targets")
    except Exception as e:                      # analysis-only hosts without the pub profile inputs
        out["feasibility_D"] = dict(error=f"{type(e).__name__}: {e}")

    lo, hi = PREDICTION["band_pct"]
    ulo, uhi = PREDICTION["secondary_union_band_pct"]
    out["prediction_outcome"] = dict(
        rate_pct=d["rate_pct"], inside_band=bool(lo <= d["rate_pct"] <= hi),
        inside_secondary_union_band=bool(ulo <= d["rate_pct"] <= uhi),
        falsifying=bool(d["rate_pct"] < PREDICTION["falsified_below_pct"]
                        or d["rate_pct"] > PREDICTION["falsified_above_pct"]),
        claimed_all=bool(d["claimed"] == d["n_valid"]))
    # four-point orderings, using the pooled A/B/C counts and D's single-seed counts
    four = {s: (abc[s] if s in abc else d) for s in ("B", "D", "A", "C")}
    for name, key, rev in (("printed_r", "r", True), ("operating_band_mae", "mae", False)):
        order = {s: DIAG[s][key] for s in four}
        seq = sorted(four, key=lambda s: -order[s] if rev else order[s])
        rates = [four[s]["rate_pct"] for s in seq]
        out.setdefault("orderings", {})[name] = dict(
            sequence=seq, rates_pct=[round(r, 1) for r in rates],
            monotone_worse_as_score_degrades=bool(all(rates[i] <= rates[i + 1] for i in range(3))),
            trend=trend(order, four))

    md = ["# Structure D — held-out test of the diagnostic", "",
          f"D: **{d['pretender']}/{d['claimed']} pretenders = {d['rate_pct']:.1f} %** "
          f"[{d['ci95'][0]:.1f}, {d['ci95'][1]:.1f}], {d['confirmed']} oracle-confirmed of "
          f"{d['n_valid']} valid; median Δ {d['median_delta_pp']:+.2f} pp, max {d['max_delta_pp']:+.2f} pp.", "",
          f"Prediction ({PREDICTION['source']}): {lo:.0f}–{hi:.0f} %. "
          f"Outcome: **{'inside' if out['prediction_outcome']['inside_band'] else 'outside'} the band**"
          f"{'; falsifying' if out['prediction_outcome']['falsifying'] else '; not falsifying'}.", "",
          "| structure | median *r* | op-band MAE | pretenders | rate [95 %] |", "|---|---|---|---|---|"]
    for s in ("B", "D", "A", "C"):
        v = four[s]
        md.append(f"| {s} | {DIAG[s]['r']:+.3f} | {DIAG[s]['mae']:.2f} % | "
                  f"{v['pretender']}/{v['claimed']} | {v['rate_pct']:.1f} % "
                  f"[{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}] |")
    fd = out["feasibility_D"]
    if "error" not in fd:
        md += ["", f"Feasibility (post hoc, rule {fd['rule']}): {fd['n_infeasible']}/{fd['n_valid']} committed D geometries infeasible; "
                   f"pretenders {fd['pretenders_infeasible']} infeasible / {fd['pretenders_feasible']} feasible; Fisher p = {fd['fisher_p']:.3f}."]
    md += ["", "## Four-point orderings", ""]
    for name, o in out["orderings"].items():
        t = o["trend"]
        md.append(f"- **{name}**: {' > '.join(o['sequence'])} → rates {o['rates_pct']}; "
                  f"monotone: {o['monotone_worse_as_score_degrades']}"
                  + (f"; Cochran–Armitage z = {t['z']:+.2f}, p = {t['p']:.3f}" if t else ""))
    md += ["", "A/B/C counts pool three training seeds (20 targets each); D is one seed over "
               "the full 50-design held-out split, and its *r* was computed by us with the "
               "printed estimator rather than taken from the companion's table.",
           "", "The A/B/C intervals and the four-point trend tests are nominal: the three seeds of a "
               "structure share one target set, so the pooled counts are correlated replicates and "
               "the effective n is smaller than 60 (design effects in stats_supplement_v9.json)."]
    out["ci_note"] = ("A/B/C: nominal Wilson over three seeds that share targets; D: single seed, "
                      "50 distinct targets, no repeated-target dependence; four-point trend tests nominal")
    (R / "structure_d_v10.json").write_text(json.dumps(out, indent=2))
    (R / "structure_d_v10.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_pub")
    main(ap.parse_args().results)
