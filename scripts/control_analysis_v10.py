#!/usr/bin/env python3
"""T55 control analysis — paired, with the continuous outcome as primary.

    python3 scripts/control_analysis_v10.py [--results results_pub]

The M0 and feasibility controls reuse the same targets as the transfer-learned run:
finetune.py:37 draws the split with `make_split(n_good, subset_seed=seed)`, which does not
depend on --init, and inverse.py draws the targets with rng(TARGET_SEED) from that split.
The comparison is therefore paired design-by-design, and this script asserts that pairing
from the archived orig_indices instead of assuming it.

Primary outcome:   Delta = RCWA MAE - surrogate MAE, paired (Wilcoxon signed-rank).
                   The estimate that goes with that test is the Hodges-Lehmann pseudo-median
                   (median of Walsh averages) with its signed-rank interval; the ordinary
                   median and its bootstrap interval, the mean, the sign test and the
                   negative/positive split are reported beside it so the reader can see the
                   asymmetry (codex full-scope review, slice 3, 2026-09-12).
Secondary:         pretender status, paired (exact McNemar).
Both are reported with the minimum detectable effect at the realized n, because at a ~10 %
pretender rate the binary comparison cannot resolve anything smaller.
Multiplicity:      the 8 per-structure Wilcoxon tests + 2 McNemar tests are one family for
                   the Holm sensitivity column; the two pooled primaries are the confirmatory
                   pair and are Holm-corrected as a family of two.

Writes <results>/control_analysis_v10.{json,md}.  Missing conditions are reported, not fatal.

Three-seed extension (T76, 2026-09-19):  --seeds 42,123,777 runs the identical per-seed
analysis for each seed and adds a pooled block.  The three seeds of a structure share the
same 20 targets (make_split fixes the hold-out before the training subset; T68), so the
pooled 180 pairs are 60 target clusters x 3 correlated replicates, not 180 independent
designs.  The pooled block therefore reports the median with a TARGET-CLUSTER bootstrap
(a target's seeds resampled together, the same device as Sections 3.4 and 3.7) and labels
the pooled signed-rank and McNemar p-values nominal.  The seed-42 analysis and its files are
untouched (regression-checked byte-for-byte); the extension writes
<results>/control_analysis_v10_3seed.{json,md}.  The extension is post hoc (Section 2.7
amendment 2 recorded the seed-42-only decision) and is reported as a replication, not as a
new confirmatory test.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
TAU = 5.0
ORDER = ["B", "A", "C"]
COND = {"base": "", "m0": "_m0", "feas": "_feas"}
RNG = np.random.default_rng(20260908)


def hodges_lehmann(d, alpha=0.05):
    """Pseudo-median of paired differences and its signed-rank (Hollander-Wolfe) interval.
    Walsh averages (d_i + d_j)/2, i <= j; the interval takes the k-th smallest and largest
    with k from the large-sample quantile of the signed-rank statistic."""
    d = np.asarray(d, float); n = d.size
    i, j = np.triu_indices(n)
    w = np.sort((d[i] + d[j]) / 2); M = w.size
    z = sps.norm.ppf(1 - alpha / 2)
    k = int(np.floor(M / 2 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(k, 0)
    return float(np.median(w)), float(w[k]), float(w[M - 1 - k])


def holm(pvals):
    """Holm step-down adjusted p-values, None-safe (None stays None)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    ps = np.array([pvals[i] for i in idx], float); m = ps.size
    order = np.argsort(ps); adj = np.empty(m)
    running = 0.0
    for rank, o in enumerate(order):
        running = max(running, (m - rank) * ps[o]); adj[o] = min(1.0, running)
    out = [None] * len(pvals)
    for a, i in zip(adj, idx):
        out[i] = float(a)
    return out


def seed_sfx(seed):
    return "" if int(seed) == 42 else f"_s{int(seed)}"


def load(R, s, tag, seed=42):
    f = R / f"rcwa_{s}{seed_sfx(seed)}{tag}_v8.npz"
    g = R / f"inverse_{s}{seed_sfx(seed)}{tag}_v8.npz"
    if not (f.exists() and g.exists()):
        return None
    d, inv = np.load(f, allow_pickle=True), np.load(g, allow_pickle=True)
    ok = ~d["failed"]
    return dict(orig=np.asarray(inv["orig_indices"]), ok=ok,
                surr=d["mae_surrogate"], rcwa=d["mae_rcwa"],
                delta=d["mae_rcwa"] - d["mae_surrogate"],
                pretender=(d["mae_surrogate"] <= TAU) & (d["mae_rcwa"] > TAU))


def mde_binary(p0, n, target_power=0.80, alpha=0.05, B=3000):
    """Smallest control rate reaching `target_power` in a PAIRED design, per concordance
    assumption.  Paired power is not fixed by the marginals: with q = P(base=1 and
    control=1) the discordant probabilities are p0-q and p1-q, so the same pair of rates
    can be easy or hard to detect depending on how the two conditions co-occur.  The
    earlier version simulated independent draws only and reported its single answer as
    "the" MDE (codex review 2026-09-09, item 7).
    """
    out = {}
    for label, q_of in (("max_concordance", lambda a, b: min(a, b)),
                        ("independence", lambda a, b: a * b),
                        ("min_concordance", lambda a, b: max(0.0, a + b - 1.0))):
        hit = None
        for p1 in np.arange(p0 + 0.02, min(p0 + 0.70, 0.95), 0.01):
            q = q_of(p0, p1)
            pb, pc = p0 - q, p1 - q                      # base-only, control-only
            draw = RNG.choice(4, size=(B, n), p=[q, pb, pc, 1 - q - pb - pc])
            b_cnt = (draw == 1).sum(1)
            c_cnt = (draw == 2).sum(1)
            pw = np.mean([sps.binomtest(int(bb), int(bb + cc), 0.5).pvalue < alpha
                          if bb + cc else False for bb, cc in zip(b_cnt, c_cnt)])
            if pw >= target_power:
                hit = float(p1)
                break
        out[label] = hit
    return out


def main(results):
    R = ROOT / results
    out = {"tau_pct": TAU, "results_dir": results, "conditions": {}, "missing": [],
           "pairing": "asserted from inverse_*.npz orig_indices"}
    data = {}
    for c, tag in COND.items():
        for s in ORDER:
            d = load(R, s, tag)
            if d is None:
                out["missing"].append(f"{c}:{s}")
            else:
                data[(c, s)] = d
    if not any(k[0] == "base" for k in data):
        print(f"nothing to analyse yet in {results}; missing {out['missing']}")
        (R / "control_analysis_v10.json").write_text(json.dumps(out, indent=2))
        return

    for c in ("m0", "feas"):
        rows, dl_all, pre_a, pre_b = [], [], [], []
        for s in ORDER:
            a, b = data.get(("base", s)), data.get((c, s))
            if a is None or b is None:
                continue
            assert np.array_equal(a["orig"], b["orig"]), \
                f"{c}/{s}: the control does not share the base run's targets; pairing is void"
            m = a["ok"] & b["ok"]
            dl = (b["delta"] - a["delta"])[m]
            w = sps.wilcoxon(a["delta"][m], b["delta"][m]) if m.sum() > 5 else None
            hl, hl_lo, hl_hi = hodges_lehmann(dl) if m.sum() > 5 else (None, None, None)
            rows.append(dict(structure=s, n_paired=int(m.sum()),
                             median_delta_base_pp=float(np.median(a["delta"][m])),
                             median_delta_control_pp=float(np.median(b["delta"][m])),
                             median_paired_shift_pp=float(np.median(dl)),
                             hodges_lehmann_shift_pp=hl, hodges_lehmann_ci95_pp=[hl_lo, hl_hi],
                             mean_paired_shift_pp=float(dl.mean()),
                             n_negative=int((dl < 0).sum()), n_positive=int((dl > 0).sum()),
                             wilcoxon_p=float(w.pvalue) if w else None,
                             pretender_base=int(a["pretender"][m].sum()),
                             pretender_control=int(b["pretender"][m].sum())))
            dl_all.append(dl)
            pre_a.append(a["pretender"][m]); pre_b.append(b["pretender"][m])
        if not rows:
            continue
        dl = np.concatenate(dl_all)
        pa, pb = np.concatenate(pre_a), np.concatenate(pre_b)
        boot = np.array([np.median(RNG.choice(dl, dl.size, replace=True)) for _ in range(4000)])
        hl, hl_lo, hl_hi = hodges_lehmann(dl)
        sign_p = float(sps.binomtest(int((dl < 0).sum()), int((dl != 0).sum()), 0.5).pvalue) if (dl != 0).sum() else None
        disc_b, disc_c = int((pa & ~pb).sum()), int((~pa & pb).sum())
        mc = sps.binomtest(disc_b, disc_b + disc_c, 0.5).pvalue if disc_b + disc_c else None
        p0 = float(pa.mean())
        out["conditions"][c] = dict(
            per_structure=rows, n_paired=int(dl.size),
            primary=dict(outcome="paired shift in Delta (control - base), pp",
                         median=float(np.median(dl)),
                         ci95_bootstrap=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                         ci_note="median: 4000-draw percentile bootstrap over the paired designs (seed 42 only, "
                                 "so no repeated-target dependence); the Wilcoxon test's own estimand is the "
                                 "Hodges-Lehmann pseudo-median below, with its signed-rank interval",
                         hodges_lehmann=hl, hodges_lehmann_ci95=[hl_lo, hl_hi],
                         mean=float(dl.mean()), n_negative=int((dl < 0).sum()), n_positive=int((dl > 0).sum()),
                         range_pp=[float(dl.min()), float(dl.max())],
                         sign_test_p=sign_p,
                         wilcoxon_p=float(sps.wilcoxon(dl).pvalue) if dl.size > 5 else None),
            secondary=dict(outcome="pretender status, exact McNemar",
                           pretender_base=int(pa.sum()), pretender_control=int(pb.sum()),
                           discordant_base_only=disc_b, discordant_control_only=disc_c,
                           mcnemar_p=float(mc) if mc is not None else None),
            power=dict(base_rate=p0, n=int(dl.size),
                       mde_binary_at_80pct=mde_binary(p0, int(dl.size)),
                       note="control rate reaching 80 % power, by concordance assumption; 80 % "
                            "power is a probability criterion, not a detection boundary, so a "
                            "smaller difference is not 'undetectable', only unlikely to be "
                            "flagged. The paired continuous outcome is primary for that reason."))

    # ---- multiplicity: the two pooled primaries are the confirmatory family; all ten
    # control tests (8 per-structure Wilcoxon + 2 pooled McNemar) are the sensitivity family
    conds = list(out["conditions"])
    prim = [out["conditions"][c]["primary"]["wilcoxon_p"] for c in conds]
    for c, a in zip(conds, holm(prim)):
        out["conditions"][c]["primary"]["wilcoxon_p_holm_over_2_primaries"] = a
    fam = [(c, "per_structure", i, r["wilcoxon_p"]) for c in conds for i, r in enumerate(out["conditions"][c]["per_structure"])]
    fam += [(c, "secondary", None, out["conditions"][c]["secondary"]["mcnemar_p"]) for c in conds]
    fam += [(c, "primary", None, out["conditions"][c]["primary"]["wilcoxon_p"]) for c in conds]
    adj = holm([f[3] for f in fam])
    for (c, where, i, _), a in zip(fam, adj):
        tgt = out["conditions"][c][where] if i is None else out["conditions"][c][where][i]
        tgt["p_holm_over_all_control_tests"] = a
    out["multiplicity"] = dict(confirmatory_family="the two pooled primary Wilcoxon tests (m0, feas), Holm",
                               sensitivity_family=f"all {len(fam)} control tests (per-structure Wilcoxon, pooled Wilcoxon, McNemar), Holm",
                               note="the sensitivity family is reported so the reader can see how far the family "
                                    "definition moves the feasibility p-value; it is not the pre-specified family")
    md = ["# T55 control analysis — paired, continuous primary", "",
          f"Results directory `{results}`, tau = {TAU:.0f} %.  "
          + (f"Missing: {', '.join(out['missing'])}." if out["missing"] else "All conditions present.")]
    for c, v in out["conditions"].items():
        p, s2, pw = v["primary"], v["secondary"], v["power"]
        md += ["", f"## {c} vs base ({v['n_paired']} paired designs)", "",
               f"- **primary** Wilcoxon signed-rank p = {p['wilcoxon_p']:.4f} "
               f"(Holm over the two primaries {p['wilcoxon_p_holm_over_2_primaries']:.4f}; over all {len(fam)} control tests "
               f"{p['p_holm_over_all_control_tests']:.4f}); Hodges–Lehmann shift **{p['hodges_lehmann']:+.3f} pp** "
               f"[{p['hodges_lehmann_ci95'][0]:+.3f}, {p['hodges_lehmann_ci95'][1]:+.3f}]; "
               f"median {p['median']:+.3f} pp [{p['ci95_bootstrap'][0]:+.3f}, {p['ci95_bootstrap'][1]:+.3f}] (bootstrap); "
               f"mean {p['mean']:+.3f} pp; negative/positive {p['n_negative']}/{p['n_positive']}, sign test p = {p['sign_test_p']:.4f}; "
               f"range [{p['range_pp'][0]:+.2f}, {p['range_pp'][1]:+.2f}]" if p["wilcoxon_p"] is not None else "",
               f"- secondary pretenders {s2['pretender_base']} → {s2['pretender_control']}; "
               f"discordant {s2['discordant_base_only']}/{s2['discordant_control_only']}, "
               + (f"exact McNemar p = {s2['mcnemar_p']:.4f}" if s2["mcnemar_p"] is not None else "no discordant pairs"),
               f"- at a base rate of {100*pw['base_rate']:.1f} % and n = {pw['n']}, the binary comparison "
               + "reaches 80 % power against a control rate of "
               + ", ".join(f"{k.replace('_', ' ')} {100*v:.0f} %" if v else f"{k.replace('_', ' ')} n/a"
                           for k, v in pw["mde_binary_at_80pct"].items()),
               "", "| structure | n | median Δ base | median Δ control | median shift | HL shift [95 %] | mean shift | neg/pos | Wilcoxon p (Holm, all tests) | pretenders |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in v["per_structure"]:
            md.append(f"| {r['structure']} | {r['n_paired']} | {r['median_delta_base_pp']:+.2f} pp | "
                      f"{r['median_delta_control_pp']:+.2f} pp | {r['median_paired_shift_pp']:+.3f} pp | "
                      + (f"{r['hodges_lehmann_shift_pp']:+.3f} [{r['hodges_lehmann_ci95_pp'][0]:+.3f}, {r['hodges_lehmann_ci95_pp'][1]:+.3f}]" if r["hodges_lehmann_shift_pp"] is not None else "—")
                      + f" | {r['mean_paired_shift_pp']:+.3f} pp | {r['n_negative']}/{r['n_positive']} | "
                      + (f"{r['wilcoxon_p']:.4f} ({r['p_holm_over_all_control_tests']:.4f})" if r["wilcoxon_p"] is not None else "—")
                      + f" | {r['pretender_base']} → {r['pretender_control']} |")
    (R / "control_analysis_v10.json").write_text(json.dumps(out, indent=2))
    (R / "control_analysis_v10.md").write_text("\n".join(x for x in md if x != "") + "\n")
    print("\n".join(x for x in md if x != ""))


def cluster_bootstrap_median(values, clusters, rng, B=4000):
    """Percentile interval for the median of `values`, resampling whole clusters (all
    replicates of a target together) so that seeds of one target are never split."""
    values = np.asarray(values, float); clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    groups = [values[clusters == u] for u in uniq]
    med = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        med[b] = np.median(np.concatenate([groups[i] for i in pick]))
    return [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]


def paired_block(a, b, rng):
    """One paired comparison (base vs control) on the designs valid in both; the same
    statistics main() reports, so per-seed blocks are like-for-like with seed 42."""
    m = a["ok"] & b["ok"]
    dl = (b["delta"] - a["delta"])[m]
    w = sps.wilcoxon(a["delta"][m], b["delta"][m]) if m.sum() > 5 else None
    hl, hl_lo, hl_hi = hodges_lehmann(dl) if m.sum() > 5 else (None, None, None)
    pa, pb = a["pretender"][m], b["pretender"][m]
    disc_b, disc_c = int((pa & ~pb).sum()), int((~pa & pb).sum())
    mc = sps.binomtest(disc_b, disc_b + disc_c, 0.5).pvalue if disc_b + disc_c else None
    return dict(n_paired=int(m.sum()), mask=m, shift=dl, cluster=a["orig"][m],
                median_shift_pp=float(np.median(dl)), hodges_lehmann_pp=hl,
                hodges_lehmann_ci95_pp=[hl_lo, hl_hi], mean_shift_pp=float(dl.mean()),
                n_negative=int((dl < 0).sum()), n_positive=int((dl > 0).sum()),
                wilcoxon_p=float(w.pvalue) if w else None,
                pretender_base=int(pa.sum()), pretender_control=int(pb.sum()),
                discordant_base_only=disc_b, discordant_control_only=disc_c,
                mcnemar_p=float(mc) if mc is not None else None,
                forward_mae_base_pct=a.get("forward_mae"), forward_mae_control_pct=b.get("forward_mae"))


def strip(d):
    return {k: v for k, v in d.items() if k not in ("mask", "shift", "cluster")}


def multi_seed(results, seeds):
    R = ROOT / results
    rng = np.random.default_rng(20260919)          # separate stream: seed-42 files must not move
    out = {"tau_pct": TAU, "results_dir": results, "seeds": seeds, "missing": [], "conditions": {},
           "pairing": "asserted per seed from inverse_*.npz orig_indices; targets asserted identical across seeds",
           "design": "each seed is the seed-42 analysis repeated; the pooled block clusters by target "
                     "(the three seeds of a target are correlated replicates, Section 3.4)"}
    data = {}
    for seed in seeds:
        for c, tag in COND.items():
            for s in ORDER:
                d = load(R, s, tag, seed)
                if d is None:
                    out["missing"].append(f"seed{seed}:{c}:{s}")
                    continue
                fj = R / f"finetune_{s}{seed_sfx(seed)}{tag if c == 'm0' else ''}_v8.json"
                if fj.exists():
                    d["forward_mae"] = float(json.loads(fj.read_text()).get("final_test_mae_pct", float("nan")))
                data[(seed, c, s)] = d
    for s in ORDER:                                  # the cluster variable must mean the same target in every seed
        sets = {seed: set(map(int, data[(seed, "base", s)]["orig"])) for seed in seeds if (seed, "base", s) in data}
        assert len({frozenset(v) for v in sets.values()}) == 1, f"{s}: target sets differ across seeds {sets.keys()}"

    for c in ("m0", "feas"):
        per_seed, pooled_shift, pooled_cluster, pooled_pa, pooled_pb, per_struct = {}, [], [], [], [], {}
        for seed in seeds:
            rows, dl_all, cl_all, pa_all, pb_all = [], [], [], [], []
            for s in ORDER:
                a, b = data.get((seed, "base", s)), data.get((seed, c, s))
                if a is None or b is None:
                    continue
                assert np.array_equal(a["orig"], b["orig"]), f"seed {seed} {c}/{s}: pairing is void"
                blk = paired_block(a, b, rng)
                rows.append(dict(structure=s, **strip(blk)))
                dl_all.append(blk["shift"]); cl_all.append([f"{s}:{t}" for t in blk["cluster"]])
                pa_all.append(a["pretender"][blk["mask"]]); pb_all.append(b["pretender"][blk["mask"]])
                per_struct.setdefault(s, dict(shift=[], cluster=[], pa=[], pb=[]))
                per_struct[s]["shift"].append(blk["shift"]); per_struct[s]["cluster"].append(blk["cluster"])
                per_struct[s]["pa"].append(a["pretender"][blk["mask"]]); per_struct[s]["pb"].append(b["pretender"][blk["mask"]])
            if not rows:
                continue
            dl = np.concatenate(dl_all); pa, pb = np.concatenate(pa_all), np.concatenate(pb_all)
            hl, lo, hi = hodges_lehmann(dl)
            db, dc = int((pa & ~pb).sum()), int((~pa & pb).sum())
            per_seed[str(seed)] = dict(
                per_structure=rows, n_paired=int(dl.size),
                hodges_lehmann=hl, hodges_lehmann_ci95=[lo, hi], median=float(np.median(dl)),
                mean=float(dl.mean()), n_negative=int((dl < 0).sum()), n_positive=int((dl > 0).sum()),
                wilcoxon_p=float(sps.wilcoxon(dl).pvalue), pretender_base=int(pa.sum()),
                pretender_control=int(pb.sum()), discordant_base_only=db, discordant_control_only=dc,
                mcnemar_p=float(sps.binomtest(db, db + dc, 0.5).pvalue) if db + dc else None)
            pooled_shift.append(dl); pooled_cluster += sum(cl_all, []); pooled_pa.append(pa); pooled_pb.append(pb)
        if not per_seed:
            continue
        dl = np.concatenate(pooled_shift); pa, pb = np.concatenate(pooled_pa), np.concatenate(pooled_pb)
        hl, lo, hi = hodges_lehmann(dl)
        db, dc = int((pa & ~pb).sum()), int((~pa & pb).sum())
        struct_rows = []
        for s in ORDER:
            if s not in per_struct:
                continue
            sh = np.concatenate(per_struct[s]["shift"]); cl = np.concatenate(per_struct[s]["cluster"])
            spa, spb = np.concatenate(per_struct[s]["pa"]), np.concatenate(per_struct[s]["pb"])
            shl, slo, shi = hodges_lehmann(sh)
            struct_rows.append(dict(structure=s, n_paired=int(sh.size), n_targets=int(len(np.unique(cl))),
                                    median_shift_pp=float(np.median(sh)),
                                    median_ci95_target_cluster_bootstrap=cluster_bootstrap_median(sh, cl, rng),
                                    hodges_lehmann_pp=shl, hodges_lehmann_ci95_pp_nominal=[slo, shi],
                                    mean_shift_pp=float(sh.mean()), n_negative=int((sh < 0).sum()),
                                    n_positive=int((sh > 0).sum()),
                                    wilcoxon_p_nominal=float(sps.wilcoxon(sh).pvalue),
                                    pretender_base=int(spa.sum()), pretender_control=int(spb.sum())))
        out["conditions"][c] = dict(
            per_seed=per_seed,
            pooled=dict(n_paired=int(dl.size), n_targets=int(len(set(pooled_cluster))), n_seeds=len(per_seed),
                        median=float(np.median(dl)),
                        median_ci95_target_cluster_bootstrap=cluster_bootstrap_median(dl, pooled_cluster, rng),
                        hodges_lehmann=hl, hodges_lehmann_ci95_nominal=[lo, hi],
                        mean=float(dl.mean()), n_negative=int((dl < 0).sum()), n_positive=int((dl > 0).sum()),
                        range_pp=[float(dl.min()), float(dl.max())],
                        wilcoxon_p_nominal=float(sps.wilcoxon(dl).pvalue),
                        pretender_base=int(pa.sum()), pretender_control=int(pb.sum()),
                        discordant_base_only=db, discordant_control_only=dc,
                        mcnemar_p_nominal=float(sps.binomtest(db, db + dc, 0.5).pvalue) if db + dc else None,
                        note="the 180 pairs are 60 targets x 3 correlated seeds; the cluster-bootstrap "
                             "interval is the honest one, the signed-rank/McNemar p-values treat pairs as "
                             "independent and are nominal"),
            per_structure_pooled=struct_rows)

    md = [f"# Control analysis, seeds {', '.join(map(str, seeds))} — replication of the seed-42 analysis and a target-clustered pool", "",
          f"Results directory `{results}`, tau = {TAU:.0f} %.  " + (f"Missing: {', '.join(out['missing'])}." if out["missing"] else "All conditions present."),
          "", "Post hoc extension of the seed-42 controls (Section 2.7, amendment 2); each seed is analysed exactly as seed 42 was; "
          "the pooled block resamples targets (a target's seeds together) for its interval and marks pair-independent p-values nominal."]
    for c, v in out["conditions"].items():
        md += ["", f"## {c} vs base", "", "| seed | n | HL shift [95 %] | median | mean | neg/pos | Wilcoxon p | pretenders | discordant | McNemar p | forward MAE base → control (B/A/C) |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for seed, p in v["per_seed"].items():
            fm = " / ".join(f"{r['forward_mae_base_pct']:.2f}→{r['forward_mae_control_pct']:.2f}" if r.get("forward_mae_base_pct") is not None else "—" for r in p["per_structure"])
            md.append(f"| {seed} | {p['n_paired']} | {p['hodges_lehmann']:+.3f} [{p['hodges_lehmann_ci95'][0]:+.3f}, {p['hodges_lehmann_ci95'][1]:+.3f}] | "
                      f"{p['median']:+.3f} | {p['mean']:+.3f} | {p['n_negative']}/{p['n_positive']} | {p['wilcoxon_p']:.4f} | "
                      f"{p['pretender_base']} → {p['pretender_control']} | {p['discordant_base_only']}/{p['discordant_control_only']} | "
                      + (f"{p['mcnemar_p']:.3f}" if p["mcnemar_p"] is not None else "—") + f" | {fm} |")
        q = v["pooled"]
        md += [f"| **pooled** | {q['n_paired']} ({q['n_targets']} targets) | {q['hodges_lehmann']:+.3f} [{q['hodges_lehmann_ci95_nominal'][0]:+.3f}, {q['hodges_lehmann_ci95_nominal'][1]:+.3f}] (nominal) | "
               f"{q['median']:+.3f} **[{q['median_ci95_target_cluster_bootstrap'][0]:+.3f}, {q['median_ci95_target_cluster_bootstrap'][1]:+.3f}] (target-cluster bootstrap)** | "
               f"{q['mean']:+.3f} | {q['n_negative']}/{q['n_positive']} | {q['wilcoxon_p_nominal']:.4f} (nominal) | "
               f"{q['pretender_base']} → {q['pretender_control']} | {q['discordant_base_only']}/{q['discordant_control_only']} | "
               + (f"{q['mcnemar_p_nominal']:.3f} (nominal)" if q["mcnemar_p_nominal"] is not None else "—") + " | |",
               "", "| structure (pooled over seeds) | n (targets) | median shift [target-cluster 95 %] | HL shift [nominal 95 %] | mean | neg/pos | Wilcoxon p (nominal) | pretenders |",
               "|---|---|---|---|---|---|---|---|"]
        for r in v["per_structure_pooled"]:
            md.append(f"| {r['structure']} | {r['n_paired']} ({r['n_targets']}) | {r['median_shift_pp']:+.3f} [{r['median_ci95_target_cluster_bootstrap'][0]:+.3f}, {r['median_ci95_target_cluster_bootstrap'][1]:+.3f}] | "
                      f"{r['hodges_lehmann_pp']:+.3f} [{r['hodges_lehmann_ci95_pp_nominal'][0]:+.3f}, {r['hodges_lehmann_ci95_pp_nominal'][1]:+.3f}] | "
                      f"{r['mean_shift_pp']:+.3f} | {r['n_negative']}/{r['n_positive']} | {r['wilcoxon_p_nominal']:.4f} | {r['pretender_base']} → {r['pretender_control']} |")
    tag = "_3seed" if len(seeds) > 1 else f"_seed{seeds[0]}"
    (R / f"control_analysis_v10{tag}.json").write_text(json.dumps(out, indent=2))
    (R / f"control_analysis_v10{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_pub")
    ap.add_argument("--seeds", default="42", help="comma-separated; '42' alone runs the original analysis")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if seeds == [42]:
        main(args.results)
    else:
        multi_seed(args.results, seeds)
