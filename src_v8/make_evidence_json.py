#!/usr/bin/env python3
"""
Build machine-readable evidence artifacts for the v8 manuscript so every number
cited in §3.4 (multi-seed pooled, order-7), §3.5 (mechanism), and the pooled
Wilson CI is verifiable from a JSON file (codex round-1 blockers #2, #3, nitpick).

Outputs into results_v8/:
  - mechanism_v8.json          (seed-42 commitment-rule comparison + Fisher p)
  - order7_revalidation_v8.json(seed-42 committed geometries re-scored at order 7)
  - pooled_v8.json             (3-seed pooled counts + Wilson 95% CIs)

Run from the InverseTL root in the ML conda env:
  python3 src_v8/make_evidence_json.py
All inputs are existing v8 artifacts; this script only re-aggregates them.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
R = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results_v8"   # T15/T37: optional results dir
STRUCTS = ["A", "B", "C"]
SEEDS = ["", "_s123", "_s777"]   # "" = seed 42
TAU = 5.0


def load(name):
    with open(R / name) as f:
        return json.load(f)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(100 * (centre - half), 2), round(100 * (centre + half), 2)]


def fisher_p(a_pre, a_n, b_pre, b_n):
    table = [[a_pre, a_n - a_pre], [b_pre, b_n - b_pre]]
    return round(float(fisher_exact(table)[1]), 4)


# ---------------------------------------------------------------- mechanism ---
mech = {
    "description": (
        "Seed-42 commitment-rule comparison. Gradient best-of-8 = reliability_*_v8 "
        "(committed = lowest-final-surrogate-loss restart). Random search = "
        "random_baseline_*_v8 (budget-matched). Single-start = restart0_*_v8 "
        "(mid-box init, no restarts). Pretender := Surr MAE<=tau AND RCWA MAE>tau."
    ),
    "tau_pct": TAU,
    "per_structure": {},
}
# The commitment-rule controls exist only for the protocol's 20 targets; an analysis view
# built for the 50-target extension (results_pub/n50) deliberately carries none, and a
# 50-design best-of-8 must not be compared with a 20-design random search.
HAVE_MECH = all((R / f"{k}_{s}_v8.json").exists() for s in STRUCTS for k in ("random_baseline", "restart0"))
if not HAVE_MECH:
    mech["status"] = "not computed: random_baseline/restart0 artifacts absent in this results dir (extension view)"
    print("mechanism: skipped (no random_baseline/restart0 in this dir)", flush=True)
for s in (STRUCTS if HAVE_MECH else []):
    rel = load(f"reliability_{s}_v8.json")
    rnd = load(f"random_baseline_{s}_v8.json")
    r0 = load(f"restart0_{s}_v8.json")
    n = rel["meta"]["n_valid"]
    best8_pre = rel["tier1a"]["pretender"]
    mean_rcwa_best8 = round(float(np.mean(rel["per_sample"]["mae_rcwa_pct"])), 4)
    mech["per_structure"][s] = {
        "n": n,
        "gradient_best_of_8": {
            "pretender": best8_pre,
            "oracle_pass": rel["tier1a"]["rcwa_pass"],
            "surrogate_pass": rel["tier1a"]["surrogate_pass"],
            "mean_oracle_mae_pct": mean_rcwa_best8,
        },
        "random_search": {
            "pretender": rnd["pretender"],
            "oracle_pass": rnd["rcwa_pass"],
            "surrogate_pass": rnd["surrogate_pass"],
            "mean_oracle_mae_pct": round(rnd["mean_rcwa"], 4),
            "budget": rnd["budget"],
        },
        "single_start": {
            "pretender": r0["pretender"],
            "oracle_pass": r0["rcwa_pass"],
            "surrogate_pass": r0["surrogate_pass"],
            "mean_oracle_mae_pct": round(r0["mean_rcwa"], 4),
        },
        "fisher_p_random_vs_best8": fisher_p(rnd["pretender"], rnd["n"], best8_pre, n),
        "fisher_p_single_vs_best8": fisher_p(r0["pretender"], r0["n"], best8_pre, n),
    }
json.dump(mech, open(R / "mechanism_v8.json", "w"), indent=2)

# ----------------------------------------------------------- order-7 reval ---
o7 = {
    "description": (
        "Order-7 FULL revalidation of every seed-42 committed geometry. This is a "
        "POST-PROTOCOL expansion: the frozen protocol pre-specified only a 3-highest-"
        "Delta order-7 spot check. Order-5 counts are the data-generating fidelity "
        "from reliability_*_v8 (seed 42). MAEs are in percent."
    ),
    "tau_pct": TAU,
    "fourier_order": 7,
    "scope": "seed 42 committed geometries only",
    "per_structure": {},
}
# The pub arm has no order-7 revalidation: its reference solver picks the order per
# wavelength (9/13/17), so a fixed-order-7 rerun is not a higher-fidelity check of it and
# PLAN_v10_amendment_pub dropped T11. Skip the block rather than abort -- pooled_v8.json
# is written after it and the manuscript audit needs that file.
HAVE_O7 = all((R / f"rcwa_{s}_v8_o7full.npz").exists() for s in STRUCTS)
for s in (STRUCTS if HAVE_O7 else ()):
    rel = load(f"reliability_{s}_v8.json")
    d = np.load(R / f"rcwa_{s}_v8_o7full.npz")
    surr7 = np.asarray(d["mae_surrogate"], float)
    rcwa7 = np.asarray(d["mae_rcwa"], float)
    failed = np.asarray(d["failed"]).astype(bool)
    valid = ~failed
    spass7 = int(np.sum((surr7 <= TAU) & valid))
    opass7 = int(np.sum((rcwa7 <= TAU) & valid))
    pre7 = int(np.sum((surr7 <= TAU) & (rcwa7 > TAU) & valid))
    o7["per_structure"][s] = {
        "n_valid": int(np.sum(valid)),
        "order5": {
            "oracle_pass": rel["tier1a"]["rcwa_pass"],
            "pretender": rel["tier1a"]["pretender"],
        },
        "order7": {
            "surrogate_pass": spass7,
            "oracle_pass": opass7,
            "pretender": pre7,
            "max_rcwa_mae_pct": round(float(np.max(rcwa7[valid])), 4),
        },
        "transition_pretender_o5_to_o7": [rel["tier1a"]["pretender"], pre7],
        "transition_oracle_pass_o5_to_o7": [rel["tier1a"]["rcwa_pass"], opass7],
    }
if o7["per_structure"]:
    json.dump(o7, open(R / "order7_revalidation_v8.json", "w"), indent=2)
else:
    print(f"[make_evidence_json] no rcwa_*_v8_o7full.npz in {R}: "
          "skipping order7_revalidation_v8.json (adaptive-order arm)", flush=True)

# ---------------------------------------------------------------- pooled -----
pooled = {
    "description": "Three-seed (42/123/777) pooled Tier-1A counts with Wilson 95% CIs.",
    "ci_note": "nominal: the three seeds of a structure share one target set, so pooled counts are correlated replicates; the design-effect-adjusted and cluster-bootstrap intervals are in stats_supplement_v9.json (clustering)",
    "tau_pct": TAU,
    "seeds": [42, 123, 777],
    "per_structure": {},
    "totals": {},
}
tot = {"surrogate_pass": 0, "oracle_pass": 0, "pretender": 0, "n_valid": 0}
for s in STRUCTS:
    sp = op = pr = nv = 0
    for sd in SEEDS:
        rel = load(f"reliability_{s}{sd}_v8.json")
        sp += rel["tier1a"]["surrogate_pass"]
        op += rel["tier1a"]["rcwa_pass"]
        pr += rel["tier1a"]["pretender"]
        nv += rel["meta"]["n_valid"]
    pooled["per_structure"][s] = {
        "n_valid": nv, "surrogate_pass": sp, "oracle_pass": op, "pretender": pr,
        "pretender_rate_pct": round(100 * pr / nv, 2),
        "pretender_wilson95_pct": wilson(pr, nv),
    }
    for k, v in (("surrogate_pass", sp), ("oracle_pass", op), ("pretender", pr), ("n_valid", nv)):
        tot[k] += v
pooled["totals"] = {
    **tot,
    "pretender_rate_pct": round(100 * tot["pretender"] / tot["n_valid"], 2),
    "pretender_wilson95_pct": wilson(tot["pretender"], tot["n_valid"]),
}
json.dump(pooled, open(R / "pooled_v8.json", "w"), indent=2)

# ================================================================ T15 extensions
# Every pre-existing key above is left untouched; the blocks below ADD keys.

def wilson_frac(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n; denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def holm(pd):
    keys = list(pd); pv = np.array([pd[k] for k in keys], float); m = len(pv)
    order = np.argsort(pv); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pv[i]); adj[i] = min(run, 1.0)
    return {k: round(float(a), 6) for k, a in zip(keys, adj)}


# --- 1. conditional mechanism rates (pretenders / surrogate-claimed) + conditional Fisher
for s in (STRUCTS if HAVE_MECH else []):
    m = mech["per_structure"][s]
    for rule in ("gradient_best_of_8", "random_search", "single_start"):
        b = m[rule]; sp = b["surrogate_pass"]
        b["pretender_rate_given_claimed"] = round(b["pretender"] / sp, 4) if sp else None
        b["wilson95_given_claimed"] = wilson_frac(b["pretender"], sp)
        b["wilson95_of_n"] = wilson_frac(b["pretender"], m["n"])
    g = m["gradient_best_of_8"]
    for rule, key in (("random_search", "fisher_p_random_vs_best8_conditional"), ("single_start", "fisher_p_single_vs_best8_conditional")):
        b = m[rule]
        m[key] = round(float(fisher_exact([[b["pretender"], b["surrogate_pass"] - b["pretender"]],
                                          [g["pretender"], g["surrogate_pass"] - g["pretender"]]])[1]), 4)

# --- 2. tau sweep with cluster bootstrap over targets; severity bins; mechanism_by_tau
TAUS = [2.5, 3.75, 5.0, 7.5, 10.0, 15.0]
per = {}   # per structure: (3, 20) arrays of surr / rcwa aligned by target (seed order 42/123/777)
for s in STRUCTS:
    S_, Rr = [], []
    for sd in SEEDS:
        rel = load(f"reliability_{s}{sd}_v8.json")
        S_.append(np.array(rel["per_sample"]["mae_surr_pct"], float))
        Rr.append(np.array([np.nan if v is None else v for v in rel["per_sample"]["mae_rcwa_pct"]], float))
    per[s] = (np.stack(S_), np.stack(Rr))


def counts(sel):
    """sel: dict s -> target index array (with replacement allowed). Returns per-tau pooled & per-structure counts."""
    out = {}
    for tau in TAUS:
        tot = dict(claimed=0, pretender=0, valid=0); by = {}
        for s in STRUCTS:
            S_, Rr = per[s]; idx = sel[s]
            sv, rv = S_[:, idx], Rr[:, idx]; v = np.isfinite(rv)
            c = int(((sv <= tau) & v).sum()); pr = int(((sv <= tau) & (rv > tau) & v).sum()); nv = int(v.sum())
            by[s] = dict(claimed=c, pretender=pr, valid=nv, pretender_rate_given_claimed=round(pr / c, 4) if c else None)
            for k, val in (("claimed", c), ("pretender", pr), ("valid", nv)):
                tot[k] += val
        tot["pretender_rate_given_claimed"] = round(tot["pretender"] / tot["claimed"], 4) if tot["claimed"] else None
        out[str(tau)] = dict(pooled=tot, per_structure=by)
    return out


NT = {s: per[s][0].shape[1] for s in STRUCTS}        # targets per structure (20 in the protocol, 50 in the extension view)
full = {s: np.arange(NT[s]) for s in STRUCTS}
sweep = counts(full)
rng = np.random.default_rng(1)
boot = {str(t): {"pooled": [], **{s: [] for s in STRUCTS}} for t in TAUS}
for _ in range(2000):
    sel = {s: rng.integers(0, NT[s], NT[s]) for s in STRUCTS}
    c = counts(sel)
    for t in TAUS:
        e = c[str(t)]
        boot[str(t)]["pooled"].append(e["pooled"]["pretender_rate_given_claimed"] or 0.0)
        for s in STRUCTS:
            boot[str(t)][s].append(e["per_structure"][s]["pretender_rate_given_claimed"] or 0.0)
for t in TAUS:
    e = sweep[str(t)]
    e["pooled"]["bootstrap95_rate_given_claimed"] = [round(float(np.percentile(boot[str(t)]["pooled"], q)), 4) for q in (2.5, 97.5)]
    for s in STRUCTS:
        e["per_structure"][s]["bootstrap95_rate_given_claimed"] = [round(float(np.percentile(boot[str(t)][s], q)), 4) for q in (2.5, 97.5)]
sev = dict(mild=0, moderate=0, severe=0)
for s in STRUCTS:
    S_, Rr = per[s]; v = np.isfinite(Rr); pr = (S_ <= TAU) & (Rr > TAU) & v
    x = Rr[pr]
    sev["mild"] += int(((x >= 5) & (x < 7.5)).sum()); sev["moderate"] += int(((x >= 7.5) & (x < 10)).sum()); sev["severe"] += int((x >= 10).sum())
pooled["tau_sweep"] = dict(taus=TAUS, definition="claimed = Surr MAE <= tau; pretender = claimed & RCWA MAE > tau; valid = finite RCWA MAE",
                           bootstrap=f"cluster bootstrap over the {sorted(set(NT.values()))} targets per structure (each target carries its three seeds), 2000 reps, default_rng(1), percentile 2.5/97.5",
                           by_tau=sweep, severity_bins=dict(edges_pct=[5, 7.5, 10], **sev, note="post-hoc band edges"))
mech["mechanism_by_tau"] = {}
for tau in ((7.5, 10.0) if HAVE_MECH else ()):
    mech["mechanism_by_tau"][str(tau)] = {}
    for s in STRUCTS:
        rel = load(f"reliability_{s}_v8.json"); rnd = load(f"random_baseline_{s}_v8.json"); r0 = load(f"restart0_{s}_v8.json")
        def cnt(sv, rv):
            sv = np.array(sv, float); rv = np.array([np.nan if x is None else x for x in rv], float); v = np.isfinite(rv)
            return dict(claimed=int(((sv <= tau) & v).sum()), pretender=int(((sv <= tau) & (rv > tau) & v).sum()), valid=int(v.sum()))
        mech["mechanism_by_tau"][str(tau)][s] = dict(best8=cnt(rel["per_sample"]["mae_surr_pct"], rel["per_sample"]["mae_rcwa_pct"]),
                                                    random=cnt(rnd["per_sample_surr"], rnd["per_sample_rcwa"]),
                                                    single=cnt(r0["per_sample_surr"], r0["per_sample_rcwa"]))

# --- 3. T4 (surrogate-local smoothness) summary
t4 = dict(per_run={}, total_flags=0, total_designs=0)
gmax = 0.0
for s in STRUCTS:
    for sd in SEEDS:
        rel = load(f"reliability_{s}{sd}_v8.json"); tx = rel["taxonomy"]
        vals = np.array([v for v in tx["t4_mean_pert_mae_pct"] if v is not None], float)
        t4["per_run"][f"{s}{sd or '_s42'}"] = dict(mean=round(float(vals.mean()), 4), max=round(float(vals.max()), 4), flags=int(tx["t4"]))
        t4["total_flags"] += int(tx["t4"]); t4["total_designs"] += int(rel["meta"]["n_valid"]); gmax = max(gmax, float(vals.max()))
t4["global_max"] = round(gmax, 4)
pooled["t4"] = t4

# --- 4. order-7: conv7 spot check, flips, shifts, paired Wilcoxon (legacy arm only)
for s in (STRUCTS if HAVE_O7 else ()):
    d5 = np.load(R / f"rcwa_{s}_v8.npz"); d7 = np.load(R / f"rcwa_{s}_v8_o7full.npz"); c7 = np.load(R / f"rcwa_{s}_v8_conv7.npz")
    m5, m7 = np.asarray(d5["mae_rcwa"], float), np.asarray(d7["mae_rcwa"], float)
    v = (~np.asarray(d5["failed"]).astype(bool)) & (~np.asarray(d7["failed"]).astype(bool)) & np.isfinite(m5) & np.isfinite(m7)
    p5, p7 = (m5 <= TAU), (m7 <= TAU)
    cv = (~np.asarray(c7["failed"]).astype(bool)) & np.isfinite(np.asarray(c7["mae_rcwa"], float))
    e = o7["per_structure"][s]
    e["conv7_spot_check"] = dict(indices=np.where(cv)[0].tolist(), mae_order7_pct=[round(float(x), 4) for x in np.asarray(c7["mae_rcwa"], float)[cv]],
                                 mae_order5_pct=[round(float(x), 4) for x in m5[cv]],
                                 matches_o7full=bool(np.allclose(np.asarray(c7["mae_rcwa"], float)[cv], m7[cv], atol=1e-6)))
    e["flips"] = dict(pass_to_fail=int((p5 & ~p7 & v).sum()), fail_to_pass=int((~p5 & p7 & v).sum()), total=int(((p5 != p7) & v).sum()))
    sh = np.abs(m7 - m5)[v]
    e["shift_pp"] = dict(max=round(float(sh.max()), 4), median=round(float(np.median(sh)), 4), argmax_index=int(np.where(v)[0][int(np.argmax(sh))]))
    e["wilcoxon_paired_p"] = round(float(wilcoxon(m5[v], m7[v])[1]), 4)

# --- 5. timing
tm = dict(order5={}, order7={})
all5, all7 = [], []
for s in STRUCTS:
    e5 = []
    for sd in SEEDS:
        d = np.load(R / f"rcwa_{s}{sd}_v8.npz"); v = ~np.asarray(d["failed"]).astype(bool)
        e5.extend(np.asarray(d["elapsed"], float)[v].tolist())
    tm["order5"][s] = dict(min=round(min(e5), 1), median=round(float(np.median(e5)), 1), max=round(max(e5), 1), n=len(e5))
    if HAVE_O7:
        d = np.load(R / f"rcwa_{s}_v8_o7full.npz"); v = ~np.asarray(d["failed"]).astype(bool)
        e7 = np.asarray(d["elapsed"], float)[v].tolist()
        tm["order7"][s] = dict(min=round(min(e7), 1), median=round(float(np.median(e7)), 1), max=round(max(e7), 1), n=len(e7))
    all5 += e5
    if HAVE_O7:
        all7 += e7
tm["order5"]["overall"] = dict(min=round(min(all5), 1), median=round(float(np.median(all5)), 1), max=round(max(all5), 1), n=len(all5))
if all7:
    tm["order7"]["overall"] = dict(min=round(min(all7), 1), median=round(float(np.median(all7)), 1), max=round(max(all7), 1), n=len(all7))
pooled["timing"] = dict(unit="s per committed geometry (100 wavelengths)", **tm)

# --- 6. multiplicity (Holm) on the nine Spearman p and the nine Fisher p
sp = {}
for s in STRUCTS:
    for sd in SEEDS:
        rel = load(f"reliability_{s}{sd}_v8.json")
        sp[f"{s}_{'42' if sd == '' else sd[2:]}"] = rel["discrimination"]["rcwa_vs_surr_PRIMARY"]["p"]
syn = load("synthesis_v8.json")
fp = {f"{m}:{k}": v for m, d in syn["fisher_exact_p"].items() for k, v in d.items()}
pooled["multiplicity"] = dict(method="holm", alpha=0.05, note="post hoc, not pre-registered",
                              spearman_p_nominal=sp, spearman_p_holm=holm(sp), fisher_p_nominal=fp, fisher_p_holm=holm(fp))

json.dump(mech, open(R / "mechanism_v8.json", "w"), indent=2)
if o7["per_structure"]:
    json.dump(o7, open(R / "order7_revalidation_v8.json", "w"), indent=2)
json.dump(pooled, open(R / "pooled_v8.json", "w"), indent=2)

# ---------------------------------------------------------------- report -----
print("== mechanism (seed 42) pretender counts [random / best8 / single], Fisher p(rnd|sgl vs best8) ==")
for s in (STRUCTS if HAVE_MECH else []):
    m = mech["per_structure"][s]
    print(f"  {s}: rnd {m['random_search']['pretender']:>2}/{m['n']}  "
          f"best8 {m['gradient_best_of_8']['pretender']:>2}/{m['n']}  "
          f"single {m['single_start']['pretender']:>2}/{m['n']}  "
          f"| p_rnd={m['fisher_p_random_vs_best8']}  p_sgl={m['fisher_p_single_vs_best8']}")
if HAVE_O7:
    print("== order-7 pretender (o5 -> o7), oracle_pass (o5 -> o7) ==")
for s in (STRUCTS if HAVE_O7 else ()):
    e = o7["per_structure"][s]
    print(f"  {s}: pretender {e['transition_pretender_o5_to_o7']}  "
          f"oracle_pass {e['transition_oracle_pass_o5_to_o7']}  (n_valid {e['n_valid']})")
print("== pooled ==")
for s in STRUCTS:
    p = pooled["per_structure"][s]
    print(f"  {s}: pretender {p['pretender']}/{p['n_valid']} = {p['pretender_rate_pct']}% CI {p['pretender_wilson95_pct']}")
t = pooled["totals"]
print(f"  TOTAL: pretender {t['pretender']}/{t['n_valid']} = {t['pretender_rate_pct']}% "
      f"CI {t['pretender_wilson95_pct']} | surrogate_pass {t['surrogate_pass']} | oracle_pass {t['oracle_pass']}")
print("written: mechanism_v8.json, pooled_v8.json" + (", order7_revalidation_v8.json" if o7["per_structure"] else "") + "")
