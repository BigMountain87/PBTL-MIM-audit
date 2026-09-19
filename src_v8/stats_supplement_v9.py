#!/usr/bin/env python3
"""T14 — statistics supplement (numpy + scipy only).  Reads the nine reliability JSONs and
synthesis_v8.json from <results>, writes <results>/stats_supplement_v9.json with blocks:
ca_trend, clustering, pairwise_fisher_pooled, t2_pretender, power, holm, pooled_rho,
common_threshold_h1b.  Usage: python3 src_v8/stats_supplement_v9.py [--results results_v8]
"""
from __future__ import annotations
import argparse, json, itertools
from pathlib import Path
import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"": 42, "_s123": 123, "_s777": 777}
TAU = 5.0
CONVENTIONS = {"preliminary": {"A": 0.72, "B": -0.07, "C": 0.34}, "intermediate": {"A": 0.72, "B": 0.64, "C": 0.44},
               "printed": {"A": 0.83, "B": 0.96, "C": 0.65}, "tmm_mae": {"A": -7.9, "B": -8.9, "C": -16.9}}


def wilson(k, n, z=1.96):
    if n <= 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def fisher_p(a, b, c, d):
    return float(sps.fisher_exact([[a, b], [c, d]])[1])


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj.tolist()


def cochran_armitage(k, n, x):
    """k successes of n per group with scores x; T = sum k_i (x_i - xbar) with xbar the n-weighted mean."""
    k, n, x = np.asarray(k, float), np.asarray(n, float), np.asarray(x, float)
    N, K = n.sum(), k.sum(); p = K / N; xbar = (n * x).sum() / N
    T = (k * (x - xbar)).sum(); var = p * (1 - p) * (n * (x - xbar) ** 2).sum()
    z = T / np.sqrt(var) if var > 0 else 0.0
    return dict(z=float(z), p=float(2 * sps.norm.sf(abs(z))), scores=x.tolist(), k=k.tolist(), n=n.tolist())


def icc1(matrix):
    """One-way random-effects ICC(1) for a (targets x seeds) 0/1 matrix (rows with NaN dropped)."""
    M = matrix[np.isfinite(matrix).all(axis=1)]; n, k = M.shape
    grand = M.mean(); row_means = M.mean(axis=1)
    msb = k * ((row_means - grand) ** 2).sum() / (n - 1)
    msw = ((M - row_means[:, None]) ** 2).sum() / (n * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else 0.0
    return float(icc), int(n), int(k)


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool); n = len(a)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def fisher_power_exact(n1, n2, p1, p2, alpha=0.05):
    pw = 0.0
    for a in range(n1 + 1):
        pa = sps.binom.pmf(a, n1, p1)
        for c in range(n2 + 1):
            if fisher_p(a, n1 - a, c, n2 - c) < alpha:
                pw += pa * sps.binom.pmf(c, n2, p2)
    return float(pw)


def spearman_power(n, rho, reps, rng, alpha=0.05):
    cov = [[1, rho], [rho, 1]]; hits = 0
    for _ in range(reps):
        xy = rng.multivariate_normal([0, 0], cov, size=n)
        hits += sps.spearmanr(xy[:, 0], xy[:, 1])[1] < alpha
    return hits / reps


def main(results):
    R = ROOT / results
    syn = json.load(open(R / "synthesis_v8.json"))
    runs = {}
    for s in "ABC":
        for sx, seed in SEEDS.items():
            d = json.load(open(R / f"reliability_{s}{sx}_v8.json")); ps = d["per_sample"]
            surr = np.array(ps["mae_surr_pct"], float); rc = np.array([np.nan if v is None else v for v in ps["mae_rcwa_pct"]], float)
            valid = np.isfinite(rc)
            runs[(s, seed)] = dict(orig=np.array(ps["orig_indices"]), surr=surr, rcwa=rc, valid=valid,
                                   delta=np.array([np.nan if v is None else v for v in ps["delta_pct"]], float),
                                   pret=(surr <= TAU) & (rc > TAU) & valid, opass=(rc <= TAU) & valid,
                                   t2=np.array(d["taxonomy"]["per_sample"]["t2"], bool),
                                   rho_p=d["discrimination"]["rcwa_vs_surr_PRIMARY"]["p"], rho=d["discrimination"]["rcwa_vs_surr_PRIMARY"]["rho"])
        o = [runs[(s, sd)]["orig"] for sd in SEEDS.values()]
        assert all(np.array_equal(o[0], oi) for oi in o), f"target alignment {s}"
    out = {}

    # ---- pooled counts per structure
    pooled = {s: dict(pretender=int(sum(runs[(s, sd)]["pret"].sum() for sd in SEEDS.values())),
                      oracle_pass=int(sum(runs[(s, sd)]["opass"].sum() for sd in SEEDS.values())),
                      n_valid=int(sum(runs[(s, sd)]["valid"].sum() for sd in SEEDS.values()))) for s in "ABC"}

    # ---- 2. clustering (needed by 1 for DEFF)
    clus = {}
    n_eff_sum = 0.0
    for s in "ABC":
        M = np.stack([np.where(runs[(s, sd)]["valid"], runs[(s, sd)]["pret"].astype(float), np.nan) for sd in SEEDS.values()], axis=1)
        icc, n_t, k = icc1(M); deff = 1 + (k - 1) * icc; n_eff = pooled[s]["n_valid"] / deff; n_eff_sum += n_eff
        seeds = list(SEEDS.values())
        kap = {f"{a}_vs_{b}": kappa(runs[(s, a)]["pret"][runs[(s, a)]["valid"] & runs[(s, b)]["valid"]],
                                    runs[(s, b)]["pret"][runs[(s, a)]["valid"] & runs[(s, b)]["valid"]]) for a, b in itertools.combinations(seeds, 2)}
        clus[s] = dict(icc1=icc, n_targets=n_t, n_seeds=k, deff=float(deff), n_eff=float(n_eff), pairwise_kappa=kap,
                       per_target_pretender_fraction=np.nanmean(M, axis=1).tolist(),
                       wilson_pretender_at_n_eff=wilson(pooled[s]["pretender"] / deff, n_eff),
                       wilson_pretender_naive=wilson(pooled[s]["pretender"], pooled[s]["n_valid"]))
    tot_pret = sum(pooled[s]["pretender"] for s in "ABC"); tot_n = sum(pooled[s]["n_valid"] for s in "ABC")
    pooled_deff = tot_n / n_eff_sum
    # cluster bootstrap over targets (each target carries its 3 seeds), 20000 reps
    rng = np.random.default_rng(0); boots = []
    per_target = {s: [(np.nansum(np.stack([np.where(runs[(s, sd)]["valid"], runs[(s, sd)]["pret"], np.nan) for sd in SEEDS.values()], 1)[i]),
                       np.isfinite(np.stack([np.where(runs[(s, sd)]["valid"], 1.0, np.nan) for sd in SEEDS.values()], 1)[i]).sum()) for i in range(len(runs[(s, 42)]["valid"]))] for s in "ABC"}
    NT = {s: len(per_target[s]) for s in "ABC"}     # 20 targets in the protocol, 50 in the extension view
    for _ in range(20000):
        k = n = 0
        for s in "ABC":
            idx = rng.integers(0, NT[s], NT[s])
            for i in idx:
                k += per_target[s][i][0]; n += per_target[s][i][1]
        boots.append(k / n)
    mw = {f"{a}_vs_{b}": float(sps.mannwhitneyu(clus[a]["per_target_pretender_fraction"], clus[b]["per_target_pretender_fraction"], alternative="two-sided")[1])
          for a, b in (("A", "B"), ("B", "C"), ("A", "C"))}
    out["clustering"] = dict(per_structure=clus, pooled=dict(pretender=tot_pret, n_valid=tot_n, deff=float(pooled_deff), n_eff=float(n_eff_sum),
                                                              wilson_naive=wilson(tot_pret, tot_n), wilson_at_n_eff=wilson(tot_pret / pooled_deff, n_eff_sum),
                                                              cluster_bootstrap_95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                                              cluster_bootstrap_reps=20000),
                             mann_whitney_per_target_fraction_exploratory=mw)

    # ---- 1. Cochran–Armitage trend under four conventions
    ca = {}
    for name, rconv in CONVENTIONS.items():
        x = [rconv[s] for s in "ABC"]
        block = {}
        for metric in ("pretender", "oracle_pass"):
            k = [pooled[s][metric] for s in "ABC"]; n = [pooled[s]["n_valid"] for s in "ABC"]
            k42 = [int(runs[(s, 42)][{"pretender": "pret", "oracle_pass": "opass"}[metric]].sum()) for s in "ABC"]; n42 = [int(runs[(s, 42)]["valid"].sum()) for s in "ABC"]
            deff = [clus[s]["deff"] for s in "ABC"]
            block[metric] = dict(nominal=cochran_armitage(k, n, x),
                                 deff_corrected=cochran_armitage([ki / di for ki, di in zip(k, deff)], [ni / di for ni, di in zip(n, deff)], x),
                                 seed42_only=cochran_armitage(k42, n42, x))
        ca[name] = dict(scores=rconv, **block)
    out["ca_trend"] = dict(pooled_counts=pooled, conventions=ca)

    # ---- 3. pairwise Fisher on pooled counts
    out["pairwise_fisher_pooled"] = {m: {f"{a}_vs_{b}": fisher_p(pooled[a][m], pooled[a]["n_valid"] - pooled[a][m], pooled[b][m], pooled[b]["n_valid"] - pooled[b][m])
                                         for a, b in (("A", "B"), ("B", "C"), ("A", "C"))} for m in ("pretender", "oracle_pass")}

    # ---- 4. T2 (box-edge) vs pretender
    tables, per_run = [], {}
    for (s, sd), r in runs.items():
        v = r["valid"]; a = int((r["t2"] & r["pret"] & v).sum()); b = int((r["t2"] & ~r["pret"] & v).sum())
        c = int((~r["t2"] & r["pret"] & v).sum()); d = int((~r["t2"] & ~r["pret"] & v).sum())
        tables.append((a, b, c, d)); per_run[f"{s}_{sd}"] = dict(table=[[a, b], [c, d]], fisher_p=fisher_p(a, b, c, d))
    num = den = 0.0; nor = dor = 0.0; A = B = C = D = 0
    for a, b, c, d in tables:
        n = a + b + c + d; A += a; B += b; C += c; D += d
        if n == 0 or min(a + b, c + d, a + c, b + d) == 0:
            continue
        E = (a + b) * (a + c) / n; V = (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
        num += a - E; den += V; nor += a * d / n; dor += b * c / n
    chi2 = num * num / den; chi2_cc = max(abs(num) - 0.5, 0) ** 2 / den
    sens, spec = A / (A + C), D / (B + D); ppv, npv = A / (A + B), D / (C + D)
    a42 = [t for (s, sd), t in zip(runs.keys(), tables) if sd == 42]
    s42 = np.array(a42).sum(axis=0)
    out["t2_pretender"] = dict(per_run=per_run, pooled_table=[[A, B], [C, D]], pooled_fisher_p=fisher_p(A, B, C, D),
                               cmh=dict(mh_odds_ratio=float(nor / dor), chi2=float(chi2), p=float(sps.chi2.sf(chi2, 1)),
                                        chi2_cc=float(chi2_cc), p_cc=float(sps.chi2.sf(chi2_cc, 1))),
                               sensitivity=float(sens), specificity=float(spec), ppv=float(ppv), npv=float(npv),
                               base_rate=float((A + C) / (A + B + C + D)), auroc_binary=float((sens + spec) / 2),
                               seed42_pooled_table=s42.tolist(), seed42_pooled_fisher_p=fisher_p(*s42.tolist()))

    # ---- 5. power
    rng = np.random.default_rng(0)
    fp = {f"0.55_vs_{p2}": fisher_power_exact(20, 20, 0.55, p2) for p2 in (0.45, 0.35, 0.25, 0.15)}
    mde = None
    for p2 in np.arange(0.54, 0.0, -0.01):
        if fisher_power_exact(20, 20, 0.55, float(p2)) >= 0.80:
            mde = float(round(0.55 - p2, 2)); break
    spw = {str(rho): spearman_power(20, rho, 10000, rng) for rho in (0.3, 0.5, 0.6, 0.7)}
    smde = None
    for rho in np.arange(0.30, 0.95, 0.05):
        if spearman_power(20, float(rho), 4000, rng) >= 0.80:
            smde = float(round(rho, 2)); break
    n_eff_pair = int(round(min(clus[s]["n_eff"] for s in "ABC")))
    out["power"] = dict(fisher_20v20=fp, fisher_20v20_mde_at_0p80=mde, spearman_n20=spw, spearman_n20_mde_at_0p80=smde,
                        fisher_pooled_60v59_0p55_vs_0p36=fisher_power_exact(60, 59, 0.55, 0.36),
                        fisher_pooled_at_n_eff=dict(n_eff=n_eff_pair, power=fisher_power_exact(n_eff_pair, n_eff_pair, 0.55, 0.36)))

    # ---- 6. Holm
    sp_p = {f"{s}_{sd}": runs[(s, sd)]["rho_p"] for s in "ABC" for sd in SEEDS.values()}
    fi_p = {f"{m}:{k}": v for m, d in syn["fisher_exact_p"].items() for k, v in d.items()}
    h1b = {k: v for k, v in fi_p.items() if k.startswith("flagged")}
    out["holm"] = dict(spearman_p_nominal=sp_p, spearman_p_holm=dict(zip(sp_p, holm(list(sp_p.values())))),
                       fisher_p_nominal=fi_p, fisher_p_holm=dict(zip(fi_p, holm(list(fi_p.values())))),
                       h1b_family_3=dict(nominal=h1b, holm=dict(zip(h1b, holm(list(h1b.values()))))), method="holm", alpha=0.05,
                       note="post hoc, not pre-registered")

    # ---- 7. pooled rho with block permutation; per-run permutation p; AUROC of surrogate MAE for pretender status
    pr = {}
    for s in "ABC":
        X = np.stack([runs[(s, sd)]["surr"] for sd in SEEDS.values()]); Y = np.stack([runs[(s, sd)]["rcwa"] for sd in SEEDS.values()])
        V = np.isfinite(Y)
        rho_obs = sps.spearmanr(X[V], Y[V])[0]
        rng = np.random.default_rng(1); cnt = 0
        for _ in range(5000):
            perm = rng.permutation(Y.shape[1]); Yp = Y[:, perm]; Vp = np.isfinite(Yp)
            if abs(sps.spearmanr(X[Vp], Yp[Vp])[0]) >= abs(rho_obs):
                cnt += 1
        per_run_p, auroc = {}, {}
        for sd in SEEDS.values():
            r = runs[(s, sd)]; v = r["valid"]; x, y = r["surr"][v], r["rcwa"][v]
            ro = sps.spearmanr(x, y)[0]; rng2 = np.random.default_rng(1); c2 = 0
            for _ in range(20000):
                if abs(sps.spearmanr(x, rng2.permutation(y))[0]) >= abs(ro):
                    c2 += 1
            per_run_p[str(sd)] = dict(rho=float(ro), perm_p=(c2 + 1) / 20001)
            p_, n_ = r["pret"][v], ~r["pret"][v]
            auroc[str(sd)] = float(sps.mannwhitneyu(x[p_], x[n_])[0] / (p_.sum() * n_.sum())) if p_.any() and n_.any() else None
        xs = X[V]; pp = np.concatenate([runs[(s, sd)]["pret"][runs[(s, sd)]["valid"]] for sd in SEEDS.values()])
        pr[s] = dict(pooled_rho=float(rho_obs), block_perm_p=(cnt + 1) / 5001, per_run=per_run_p, auroc_surr_mae_for_pretender=auroc,
                     auroc_pooled=float(sps.mannwhitneyu(xs[pp], xs[~pp])[0] / (pp.sum() * (~pp).sum())))
    out["pooled_rho"] = pr

    # ---- 8. common-threshold H1b
    thr = {s: syn["per_structure"][s]["flag_thr"] for s in "ABC"}
    ct = {}
    for ts, t in thr.items():
        counts = {s: int(((runs[(s, 42)]["delta"] > t) & runs[(s, 42)]["valid"]).sum()) for s in "ABC"}
        nv = {s: int(runs[(s, 42)]["valid"].sum()) for s in "ABC"}
        ct[f"thr_from_{ts}"] = dict(threshold_pct=t, flagged=counts, n_valid=nv,
                                    fisher={f"{a}_vs_{b}": fisher_p(counts[a], nv[a] - counts[a], counts[b], nv[b] - counts[b]) for a, b in (("A", "B"), ("B", "C"), ("A", "C"))})
    out["common_threshold_h1b"] = ct

    (R / "stats_supplement_v9.json").write_text(json.dumps(out, indent=1))
    t2 = out["t2_pretender"]
    print(f"wrote {R/'stats_supplement_v9.json'}")
    print(f"pooled rates: " + ", ".join(f"{s} {pooled[s]['pretender']}/{pooled[s]['n_valid']}" for s in "ABC"))
    print(f"CMH OR={t2['cmh']['mh_odds_ratio']:.2f} p={t2['cmh']['p']:.4f}; pooled Fisher T2 p={t2['pooled_fisher_p']:.4f}; seed42 Fisher p={t2['seed42_pooled_fisher_p']:.3f}")
    print(f"ICC A={clus['A']['icc1']:.2f} B={clus['B']['icc1']:.2f} C={clus['C']['icc1']:.2f}; pooled DEFF={pooled_deff:.2f}; min Holm Fisher={min(out['holm']['fisher_p_holm'].values()):.3f}")
    for name in CONVENTIONS:
        c = ca[name]["pretender"]; print(f"CA trend ({name}) pretender: nominal p={c['nominal']['p']:.3f}  DEFF-corrected p={c['deff_corrected']['p']:.3f}  seed42 p={c['seed42_only']['p']:.3f}")
    print("pooled rho:", {s: (round(pr[s]['pooled_rho'], 3), round(pr[s]['block_perm_p'], 3)) for s in 'ABC'})
    print("power:", out["power"]["fisher_20v20"], "MDE", out["power"]["fisher_20v20_mde_at_0p80"], "spearman", out["power"]["spearman_n20"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results_v8"); main(ap.parse_args().results)
