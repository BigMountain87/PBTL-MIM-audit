#!/usr/bin/env python3
"""T41 — consistency audit of the manuscript against the evidence JSONs.

What this does and does not establish (codex full-scope review, 2026-09-12, slice 5):

Part A  for each of a fixed list of headline quantities, formats the value from a named
        JSON/npz field and requires that string to occur SOMEWHERE in the body.  This
        catches a stale or missing number.  It does NOT catch an additional, contradictory
        claim elsewhere -- "178 pretenders among 179" passes as long as "16/179" also
        appears -- and it does not recompute anything from spectra.  Part D below is the
        targeted defence against contradictory counts; the substring test remains the
        core, and the text outside the checked list is unchecked.
Part B  forbidden patterns (stale numbers, internal paths, retracted wording, version tags)
Part C  end matter, abstract length, keywords, and the TODO-AUTHOR/[CHECK] inventory
Part D  every "N/179", "N of 179" and "N pretenders" in the body must equal a count that
        the artifacts actually contain, so a second, wrong headline cannot coexist with the
        right one

Usage: python3 scripts/check_manuscript_v10.py paper/manuscript_v10.md [--results results_v8]
       [--skip-endmatter]   (before T40 has added the end matter)
Exit code 1 on any FAIL.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(mpath, results, skip_end):
    M = Path(mpath) if Path(mpath).is_absolute() else ROOT / mpath
    R = ROOT / results
    txt = M.read_text()
    body = txt.split("\n## References")[0]
    J = {n: json.load(open(R / f"{n}.json")) for n in
         ("pooled_v8", "synthesis_v8", "mechanism_v8", "order7_revalidation_v8", "stats_supplement_v9",
          "feasibility_v8", "lookup_null_v8", "recoverability_v8", "detector_bench_v8",
          "control_analysis_v10", "structure_d_v10", "cross_solver_v11", "pub_vs_legacy_v10",
          "oracle_admissibility_v11", "control_analysis_v10_3seed", "paper1_b_feature_census_v11", "grid_delta_summary_v11")
         if (R / f"{n}.json").exists()}
    rel = {f"{s}{sx}": json.load(open(R / f"reliability_{s}{sx}_v8.json"))
           for s in "ABC" for sx in ("", "_s123", "_s777") if (R / f"reliability_{s}{sx}_v8.json").exists()}
    checks, fails, skipped = [], [], []

    def want(label, value, source):
        """Assert that `value` (string) occurs in the body.  The manuscript sets a negative
        number with a typographic minus, so compare with that normalised away."""
        # line wraps inside a phrase are not a mismatch
        norm = lambda x: re.sub(r"\s+", " ", x.replace("\u2212", "-"))
        ok = norm(value) in norm(body)
        checks.append((label, value, source, ok))
        if not ok:
            fails.append(f"[A] {label}: {value!r} (from {source}) not found in the manuscript")

    # ---------------- Part A: numbers
    p = J["pooled_v8"]; tot = p["totals"]; per = p["per_structure"]
    want("pooled pretenders", f"{tot['pretender']}/{tot['n_valid']}", "pooled_v8.totals")
    want("pooled rate", f"{tot['pretender_rate_pct']:.1f} %", "pooled_v8.totals.pretender_rate_pct")
    want("oracle confirmed", str(tot["oracle_pass"]), "pooled_v8.totals.oracle_pass")
    want("surrogate claimed", f"{tot['surrogate_pass']}/{tot['n_valid']}", "pooled_v8.totals.surrogate_pass")
    for s in "ABC":
        want(f"pooled pretenders {s}", f"{per[s]['pretender']}/{per[s]['n_valid']}", f"pooled_v8.per_structure.{s}")
        want(f"pooled oracle pass {s}", f"{per[s]['oracle_pass']}/{per[s]['n_valid']}", f"pooled_v8.per_structure.{s}")
    lo, hi = tot["pretender_wilson95_pct"]
    want("nominal Wilson", f"[{lo:.1f}, {hi:.1f}] %", "pooled_v8.totals.pretender_wilson95_pct")
    # per-seed pretender splits, in the manuscript's B, A, C order
    for s in "ABC":
        seq = "/".join(str(rel[f"{s}{sx}"]["tier1a"]["pretender"]) for sx in ("", "_s123", "_s777"))
        want(f"per-seed pretenders {s}", f"{s} {seq}", "reliability_*.tier1a.pretender")
    # T4
    t4 = p.get("t4")
    if t4:
        want("T4 flags", f"{t4.get('total_flags', 0)}/{t4['total_designs']}", "pooled_v8.t4.total_flags")
        # two decimals: at one decimal '3.0 %' matched an unrelated number elsewhere (codex slice 4)
        want("T4 global max", f"{t4['global_max']:.2f} %", "pooled_v8.t4.global_max")
    # timing
    tm = p.get("timing")
    if tm:
        want("order-5 timing", f"{tm['order5']['overall']['min']:.0f}–{tm['order5']['overall']['max']:.0f} s",
             "pooled_v8.timing.order5.overall")
        # the adaptive-order arm has no fixed-order-7 rerun (PLAN_v10_amendment_pub drops T11)
        if "overall" in tm.get("order7", {}):
            want("order-7 timing", f"{tm['order7']['overall']['min']:.0f}–{tm['order7']['overall']['max']:.0f} s",
                 "pooled_v8.timing.order7.overall")
        else:
            skipped.append("order-7 timing (no fixed-order-7 rerun in this arm)")
    # tau sweep
    ts = p.get("tau_sweep")
    if ts:
        for t in ("7.5", "10.0"):
            e = ts["by_tau"][t]["pooled"]
            want(f"tau {t} pretenders", f"{e['pretender']}/{e['claimed']}", f"pooled_v8.tau_sweep.by_tau.{t}")
        e = ts["by_tau"]["2.5"]["pooled"]      # the abstract's central sensitivity claim (codex second pass, must-fix 8)
        want("tau 2.5 pretenders", f"{e['pretender']} of the {e['claimed']}", "pooled_v8.tau_sweep.by_tau.2.5")
        want("tau 2.5 rate", f"({100 * e['pretender'] / e['claimed']:.0f} %)", "pooled_v8.tau_sweep.by_tau.2.5")
        sev = ts["severity_bins"]
        for k in ("mild", "moderate", "severe"):
            want(f"severity {k}", str(sev[k]), "pooled_v8.tau_sweep.severity_bins")
    # order 7
    o7 = J.get("order7_revalidation_v8", {}).get("per_structure", {})
    for s in "ABC":
        if s in o7 and "flips" in o7[s]:
            want(f"order-7 flips {s}", f"{s} {o7[s]['flips']['total']}/20", "order7_revalidation_v8.flips")
    # statistics
    st = J.get("stats_supplement_v9")
    if st:
        t2 = st["t2_pretender"]
        want("CMH OR", f"{t2['cmh']['mh_odds_ratio']:.1f}", "stats_supplement_v9.t2_pretender.cmh")
        want("CMH p", f"{t2['cmh']['p']:.3f}", "stats_supplement_v9.t2_pretender.cmh.p")
        for k, lab in (("sensitivity", "sensitivity"), ("specificity", "specificity"), ("ppv", "PPV")):
            want(f"T2 {lab}", f"{t2[k]:.2f}", f"stats_supplement_v9.t2_pretender.{k}")
        cl = st["clustering"]["per_structure"]
        for s in "ABC":
            want(f"ICC {s}", f"{cl[s]['icc1']:.2f}", f"stats_supplement_v9.clustering.{s}.icc1")
        want("pooled DEFF", f"{st['clustering']['pooled']['deff']:.2f}", "stats_supplement_v9.clustering.pooled.deff")
        ca = st["ca_trend"]["conventions"]
        want("trend printed nominal", f"{ca['printed']['pretender']['nominal']['p']:.3f}", "stats_supplement_v9.ca_trend.printed")
        want("trend printed DEFF", f"{ca['printed']['pretender']['deff_corrected']['p']:.3f}", "stats_supplement_v9.ca_trend.printed")
        want("trend tmm_mae nominal", f"{ca['tmm_mae']['pretender']['nominal']['p']:.3f}", "stats_supplement_v9.ca_trend.tmm_mae")
    # T4 per-run mean range and Tier-1C endpoint spread range (§3.7 bullets)
    if p.get("t4"):
        means = [v["mean"] for v in p["t4"]["per_run"].values()]
        want("T4 mean range", f"{min(means):.2f}–{max(means):.2f} %", "pooled_v8.t4.per_run.*.mean")
    if rel:
        sp = [r["tier1c"]["endpoint_spread_u_mean"] for r in rel.values()]
        want("endpoint spread range", f"{min(sp):.2f}–{max(sp):.2f}", "reliability_*.tier1c.endpoint_spread_u_mean")
    # feasibility, lookup, recoverability, detector benchmark: the checker used to load these
    # and check nothing from them (codex slice 4, must-fix 10)
    fe = J.get("feasibility_v8", {}).get("pooled", {})
    for s in "ABC":
        if s in fe:
            want(f"infeasible {s}", f"{fe[s]['n_infeasible']}/{fe[s]['n_valid']}", f"feasibility_v8.pooled.{s}")
    if "B" in fe:
        want("infeasibility Fisher B", f"p = {fe['B']['fisher']['p']:.4f}", "feasibility_v8.pooled.B.fisher.p")
    lk = J.get("lookup_null_v8", {}).get("runs", {})
    if all(s in lk for s in "ABC"):
        want("lookup successes", f"{lk['B']['nn_le_tau']}/20 targets on B, {lk['A']['nn_le_tau']}/20 on A and {lk['C']['nn_le_tau']}/20 on C",
             "lookup_null_v8.runs.*.nn_le_tau")
        want("lookup medians", f"{lk['B']['nn_mae_median_pct']:.1f}, {lk['A']['nn_mae_median_pct']:.1f} and {lk['C']['nn_mae_median_pct']:.1f} %",
             "lookup_null_v8.runs.*.nn_mae_median_pct")
    rc = J.get("recoverability_v8", {}).get("runs", {})
    if all(s in rc for s in "ABC"):
        want("surrogate MAE at truth", f"{rc['B']['surr_mae_at_truth_median_pct']:.1f} % on B, {rc['A']['surr_mae_at_truth_median_pct']:.1f} % on A and {rc['C']['surr_mae_at_truth_median_pct']:.1f} % on C",
             "recoverability_v8.runs.*.surr_mae_at_truth_median_pct")
        want("u-distance medians", f"{rc['B']['u_dist_median']:.2f}, {rc['A']['u_dist_median']:.2f} and {rc['C']['u_dist_median']:.2f}",
             "recoverability_v8.runs.*.u_dist_median")
    db = J.get("detector_bench_v8", {}).get("pooled", {})
    if all(s in db for s in "ABC"):
        def cell(s, k):
            v = db[s][k]; return f"{v['auroc']:.2f} [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]"
        for k, lab in (("mae_surr", "the surrogate's own claimed MAE"), ("ens_std", "3-seed ensemble disagreement"),
                       ("infeasible_flag", "generator-infeasibility flag"), ("t4_mean_pert", "T4 local-smoothness value")):
            want(f"Table 10 row {k}", f"| {lab} | {cell('B', k)} | {cell('A', k)} | {cell('C', k)} |", f"detector_bench_v8.pooled.*.{k}")
    # paired controls (§3.6, §3.8): the Wilcoxon-consistent estimate and the bootstrap median
    ctl = J.get("control_analysis_v10", {}).get("conditions", {})
    for c_, lab in (("m0", "M0"), ("feas", "feasible")):
        if c_ in ctl:
            pr = ctl[c_]["primary"]
            want(f"{lab} HL shift", f"{pr['hodges_lehmann']:+.2f} pp", f"control_analysis_v10.conditions.{c_}.primary.hodges_lehmann")
            want(f"{lab} HL CI", f"[{pr['hodges_lehmann_ci95'][0]:+.2f}, {pr['hodges_lehmann_ci95'][1]:+.2f}]",
                 f"control_analysis_v10.conditions.{c_}.primary.hodges_lehmann_ci95")
            want(f"{lab} median shift", f"{pr['median']:+.2f} pp", f"control_analysis_v10.conditions.{c_}.primary.median")
            want(f"{lab} Wilcoxon p", f"{pr['wilcoxon_p']:.3f}" if pr["wilcoxon_p"] >= 0.01 else f"{pr['wilcoxon_p']:.4f}",
                 f"control_analysis_v10.conditions.{c_}.primary.wilcoxon_p")
            want(f"{lab} McNemar", f"{ctl[c_]['secondary']['pretender_base']} → {ctl[c_]['secondary']['pretender_control']}",
                 f"control_analysis_v10.conditions.{c_}.secondary")
    for c_ in ("m0", "feas"):
        if c_ in ctl:
            for r in ctl[c_]["per_structure"]:
                want(f"{c_} {r['structure']} median shift", f"{r['structure']} {r['median_paired_shift_pp']:+.2f} pp (*p* = "
                     + (f"{r['wilcoxon_p']:.3f}" if r["wilcoxon_p"] >= 0.01 else f"{r['wilcoxon_p']:.4f}"),
                     f"control_analysis_v10.conditions.{c_}.per_structure.{r['structure']}")
    if "feas" in ctl:
        want("feasible Holm over all control tests", f"{ctl['feas']['primary']['p_holm_over_all_control_tests']:.3f}",
             "control_analysis_v10.conditions.feas.primary.p_holm_over_all_control_tests")
    # three-seed replication of the controls (§3.6, §3.8, W7, abstract; T80)
    c3 = J.get("control_analysis_v10_3seed", {}).get("conditions", {})
    for c_, lab in (("m0", "M0"), ("feas", "feasible")):
        if c_ not in c3:
            continue
        for seed in ("123", "777"):
            ps = c3[c_]["per_seed"][seed]
            want(f"{lab} seed {seed} HL", f"{ps['hodges_lehmann']:+.2f} pp [{ps['hodges_lehmann_ci95'][0]:+.2f}, {ps['hodges_lehmann_ci95'][1]:+.2f}]",
                 f"control_analysis_v10_3seed.{c_}.per_seed.{seed}")
            want(f"{lab} seed {seed} pretenders", f"{ps['pretender_base']} → {ps['pretender_control']}", f"control_analysis_v10_3seed.{c_}.per_seed.{seed}")
        q = c3[c_]["pooled"]
        want(f"{lab} pooled cluster median", f"{q['median']:+.2f} pp [{q['median_ci95_target_cluster_bootstrap'][0]:+.2f}, {q['median_ci95_target_cluster_bootstrap'][1]:+.2f}]",
             f"control_analysis_v10_3seed.{c_}.pooled.median_ci95_target_cluster_bootstrap")
        want(f"{lab} pooled pretenders", f"{q['pretender_base']} → {q['pretender_control']}", f"control_analysis_v10_3seed.{c_}.pooled")
        want(f"{lab} pooled discordant", f"{q['discordant_base_only']}/{q['discordant_control_only']}", f"control_analysis_v10_3seed.{c_}.pooled")
        for r in c3[c_]["per_structure_pooled"]:
            want(f"{lab} pooled {r['structure']} pretenders", f"{r['pretender_base']} → {r['pretender_control']}", f"control_analysis_v10_3seed.{c_}.per_structure_pooled.{r['structure']}")
        if c_ == "m0":
            r = next(x for x in c3[c_]["per_structure_pooled"] if x["structure"] == "A")
            want("M0 pooled A cluster median", f"{r['median_shift_pp']:+.2f} pp [{r['median_ci95_target_cluster_bootstrap'][0]:+.2f}, {r['median_ci95_target_cluster_bootstrap'][1]:+.2f}]",
                 "control_analysis_v10_3seed.m0.per_structure_pooled.A")
    # W8: oracle spectra outside the companion's admissibility band (T79)
    adm = J.get("oracle_admissibility_v11")
    if adm:
        want("W8 admissibility count", f"{adm['n_outside']} of {adm['n_spectra_checked']} archived committed-design spectra",
             "oracle_admissibility_v11.n_outside / n_spectra_checked")
        assert adm["n_verdict_changes_clipped"] == 0 and adm["n_verdict_changes_dropped"] == 0, \
            "W8 says neither verdict changes under clipping/dropping; the census disagrees"
    # W8: the companion's ">= 50 nm" raster premise measured on its own Structure-B dataset (T81)
    bc = J.get("paper1_b_feature_census_v11")
    if bc:
        mf = bc["features"]["min_feature"]
        want("W8 B census below 50 nm", f"{mf['n_below_50nm']} of the {bc['n_samples']} samples",
             "paper1_b_feature_census_v11.features.min_feature.n_below_50nm / n_samples")
        want("W8 B census pct below 50 nm", f"({mf['pct_below_50nm']:.1f} %)", "paper1_b_feature_census_v11.features.min_feature.pct_below_50nm")
        want("W8 B census below two pixels", f"{bc['min_feature_below_2px']['n']} ({bc['min_feature_below_2px']['pct']:.1f} %) below two pixels",
             "paper1_b_feature_census_v11.min_feature_below_2px")
        want("W8 B design-space floor", f"extends to {bc['design_space_min_nm']['R_disk']} nm", "paper1_b_feature_census_v11.design_space_min_nm.R_disk")
    # W8: raster re-solves of committed designs and of the companion's B dataset (T78/T82/T98)
    gd = J.get("grid_delta_summary_v11")
    if gd:
        c_ = gd["committed"]; b_ = gd["companion_B_dataset"]
        want("W8 re-solved designs", f"Re-solving {c_['n_designs']} committed designs", "grid_delta_summary_v11.committed.n_designs")
        want("W8 within 2 pp", f"every design within 2 pp of τ ({c_['within_2pp_of_tau']['n']})", "grid_delta_summary_v11.committed.within_2pp_of_tau.n")
        want("W8 wavelength coverage", f"{c_['n_full_spectrum']} at all 100 wavelengths and {c_['n_subsampled']} on", "grid_delta_summary_v11.committed.n_full_spectrum/n_subsampled")
        want("W8 mean dA range", f"mean |ΔA| {c_['mean_abs_dA_pp']['min']:.2f}–{c_['mean_abs_dA_pp']['max']:.2f} pp per design", "grid_delta_summary_v11.committed.mean_abs_dA_pp")
        want("W8 worst wavelength", f"single wavelength {c_['worst_single_wavelength_pp']:.1f} pp, against", "grid_delta_summary_v11.committed.worst_single_wavelength_pp")
        assert c_["n_flips"] == 0, "W8 says no verdict changed; the summary disagrees"
        want("W8 companion B n", f"Re-solving {b_['n']} of the companion's Structure-B dataset samples", "grid_delta_summary_v11.companion_B_dataset.n")
        want("W8 companion B shift", f"moves their labels by {b_['mean_abs_dA_pp']['min']:.2f}–{b_['mean_abs_dA_pp']['max']:.2f} pp", "grid_delta_summary_v11.companion_B_dataset.mean_abs_dA_pp")
        want("W8 companion B worst", f"worst single wavelength {b_['max_abs_dA_pp']['max']:.1f} pp), with the grid-64 control", "grid_delta_summary_v11.companion_B_dataset.max_abs_dA_pp")
        want("W8 companion B control", f"archived labels to {b_['control_grid64_vs_archived_mean_pp']['median']:.3f} pp", "grid_delta_summary_v11.companion_B_dataset.control")
    # Structure D (§3.12): the recorded-prediction outcome and the reliability accounting
    sd = J.get("structure_d_v10", {})
    fD = R / "reliability_D_v8.json"
    if sd.get("structure_D") and fD.exists():
        D = sd["structure_D"]; rD = json.load(open(fD))
        want("D pretenders", f"{D['pretender']} of {D['n_valid']} ({D['rate_pct']:.1f} %, Wilson [{D['ci95'][0]:.1f}, {D['ci95'][1]:.1f}])",
             "structure_d_v10.structure_D")
        want("D band", f"{sd['prediction']['band_pct'][0]:.0f}–{sd['prediction']['band_pct'][1]:.0f} %", "structure_d_v10.prediction.band_pct")
        want("D diagnostic r", f"*r* = {sd['diagnostic']['D']['r']:+.3f}", "structure_d_v10.diagnostic.D.r")
        want("D op-band MAE", f"{sd['diagnostic']['D']['mae']:.2f} %", "structure_d_v10.diagnostic.D.mae")
        want("D flags", f"{rD['tier1b']['flagged']} of {rD['meta']['n_valid']} at a threshold of {rD['tier1b']['flag_threshold_pct']:.2f} %", "reliability_D.tier1b")
        want("D rho", f"ρ = {rD['discrimination']['rcwa_vs_surr_PRIMARY']['rho']:+.2f}", "reliability_D.discrimination")
        want("D forward MAE", f"{rD['meta']['forward_test_mae_pct']:.2f} %", "reliability_D.meta.forward_test_mae_pct")
        fe_ = sd.get("feasibility_D", {})
        if "fisher_p" in fe_:
            want("D infeasible", f"{fe_['n_infeasible']} of the {fe_['n_valid']} committed D geometries", "structure_d_v10.feasibility_D")
            want("D feasibility Fisher", f"*p* = {fe_['fisher_p']:.3f}", "structure_d_v10.feasibility_D.fisher_p")
        for name, o in sd.get("orderings", {}).items():
            want(f"D ordering {name}", ", ".join(f"{x:.1f}" for x in o["rates_pct"][:-1]) + f" and {o['rates_pct'][-1]:.1f} %", f"structure_d_v10.orderings.{name}")
            want(f"D trend {name}", f"*p* = {o['trend']['p']:.2f}", f"structure_d_v10.orderings.{name}.trend")
    # two-arm clustered interval (§3.10)
    pl = J.get("pub_vs_legacy_v10", {}).get("pooled_difference_pp", {})
    if "cluster_bootstrap_ci95" in pl:
        want("two-arm clustered CI", f"[{pl['cluster_bootstrap_ci95'][0]:.1f}, {pl['cluster_bootstrap_ci95'][1]:.1f}] pp", "pub_vs_legacy_v10.pooled_difference_pp.cluster_bootstrap_ci95")
    # reproducibility recheck (§3 intro): must exist in this arm and pass
    rp = R / "repro_check_v8.json"
    if rp.exists():
        rj = json.load(open(rp))
        if not rj.get("pass") or rj.get("n_designs", 0) < 12:
            fails.append(f"[A] repro_check_v8.json in {results}: pass={rj.get('pass')} n={rj.get('n_designs')}")
        want("repro designs", f"{rj['n_designs']} committed geometries", "repro_check_v8.n_designs")
    elif "on both arms" in body:
        fails.append(f"[A] manuscript claims the recheck on both arms but {results} has no repro_check_v8.json")
    # cross-solver probe (§3.11)
    cs = J.get("cross_solver_v11", {})
    if cs.get("pooled"):
        pp = cs["pooled"]
        want("cross-solver pooled pretenders", f"{pp['pretender_adaptive']} → {pp['pretender_order5']} of {pp['n_valid']}", "cross_solver_v11.pooled")
        want("cross-solver per structure", ", ".join(f"{s_} {cs['per_structure'][s_]['pretender_adaptive']} → {cs['per_structure'][s_]['pretender_order5']}" for s_ in "BAC"),
             "cross_solver_v11.per_structure")
        want("cross-solver higher/lower", f"higher on {pp['n_order5_higher']} of the {pp['n_valid']} designs and lower on {pp['n_order5_lower']}", "cross_solver_v11.pooled")
        want("cross-solver median shift", f"median shift {pp['median_shift_pp']:+.2f} pp, median |shift| {pp['median_abs_shift_pp']:.2f} pp", "cross_solver_v11.pooled")
        want("cross-solver McNemar", f"*p* = {pp['mcnemar_p']:.3f}", "cross_solver_v11.pooled.mcnemar_p")
        want("cross-solver max shifts", ", ".join(f"{cs['per_structure'][s_]['max_abs_shift_pp']:.1f}" for s_ in "BA") + f" and {cs['per_structure']['C']['max_abs_shift_pp']:.1f} pp", "cross_solver_v11.per_structure.*.max_abs_shift_pp")
        want("cross-solver r", "Pearson *r* = " + ", ".join(f"{cs['per_structure'][s_]['pearson_r_adaptive_vs_order5']:.2f}" for s_ in "BA") + f", {cs['per_structure']['C']['pearson_r_adaptive_vs_order5']:.2f}", "cross_solver_v11.per_structure.*.pearson_r")
    # 20->50 target extension, robustness paragraph in §3.4
    n50dir = R / "n50"
    if (n50dir / "pooled_v8.json").exists() and (n50dir / "stats_supplement_v9.json").exists():
        p50 = json.load(open(n50dir / "pooled_v8.json"))
        s50 = json.load(open(n50dir / "stats_supplement_v9.json"))
        tot50, per50, cl50 = p50["totals"], p50["per_structure"], s50["clustering"]["pooled"]
        want("n50 pooled pretenders", f"{tot50['pretender']}/{tot50['n_valid']} = {tot50['pretender_rate_pct']:.1f} %", "n50/pooled_v8.totals")
        want("n50 nominal Wilson", f"[{tot50['pretender_wilson95_pct'][0]:.1f}, {tot50['pretender_wilson95_pct'][1]:.1f}]", "n50/pooled_v8.totals.pretender_wilson95_pct")
        want("n50 deff Wilson", f"[{cl50['wilson_at_n_eff'][0]*100:.1f}, {cl50['wilson_at_n_eff'][1]*100:.1f}] at n_eff = {cl50['n_eff']:.0f}", "n50/stats_supplement_v9.clustering.pooled.wilson_at_n_eff")
        want("n50 cluster bootstrap", f"[{cl50['cluster_bootstrap_95'][0]*100:.1f}, {cl50['cluster_bootstrap_95'][1]*100:.1f}]", "n50/stats_supplement_v9.clustering.pooled.cluster_bootstrap_95")
        want("n50 per-structure", f"A {per50['A']['pretender']}/{per50['A']['n_valid']} = {per50['A']['pretender_rate_pct']:.1f} %, C {per50['C']['pretender']}/{per50['C']['n_valid']} = {per50['C']['pretender_rate_pct']:.1f} %, B {per50['B']['pretender']}/{per50['B']['n_valid']} = {per50['B']['pretender_rate_pct']:.1f} %", "n50/pooled_v8.per_structure")
        d50 = {sn: s50["clustering"]["per_structure"][sn]["deff"] for sn in "ABC"}
        want("n50 design effects", f"A {d50['A']:.2f}, B {d50['B']:.2f}, C {d50['C']:.2f} at 50 targets", "n50/stats_supplement_v9.clustering.per_structure.*.deff")
        tsw50 = p50.get("tau_sweep", {}).get("by_tau", {}).get("2.5", {}).get("pooled", {})
        if tsw50:
            want("n50 tau=2.5 sweep", f"{tsw50['pretender']} of the {tsw50['claimed']}", "n50/pooled_v8.tau_sweep.by_tau.2.5.pooled")
            want("n50 tau=2.5 rate", f"{tsw50['pretender_rate_given_claimed']*100:.1f} %", "n50/pooled_v8.tau_sweep.by_tau.2.5.pooled.pretender_rate_given_claimed")
            bs50 = tsw50.get("bootstrap95_rate_given_claimed")
            if bs50:
                want("n50 tau=2.5 bootstrap", f"[{bs50[0]*100:.1f}, {bs50[1]*100:.1f}]", "n50/pooled_v8.tau_sweep.by_tau.2.5.pooled.bootstrap95_rate_given_claimed")
    elif "50-target" in body or "n_eff = 348" in body:
        fails.append(f"[A] manuscript references the 50-target extension but {results}/n50 has no pooled_v8.json/stats_supplement_v9.json")
    # mechanism
    me = J.get("mechanism_v8", {}).get("per_structure", {})
    for s in "ABC":
        if s in me:
            b = me[s]
            want(f"mechanism random {s}", f"{b['random_search']['pretender']}/{b['random_search']['surrogate_pass']}",
                 f"mechanism_v8.per_structure.{s}.random_search")
    # elapsed arrays sanity (percent units)
    for s in "ABC":
        f = R / f"rcwa_{s}_v8.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            m = np.asarray(z["mae_rcwa"], float)
            if not (0 < np.nanmax(m) < 100):
                fails.append(f"[A] rcwa_{s}_v8.npz mae_rcwa is not in percent")

    # ---------------- Part B: forbidden patterns
    FORBIDDEN = [
        (r"(?<![0-9/])0/60\b", "stale T4 denominator"), (r"1\.6[–-]2\.9", "stale T4 range"),
        (r"9/9/10", "stale A per-seed counts"), (r"p = 0\.022", "non-reproducible T2 p"),
        (r"uncorrelated", "superseded T2 wording"), (r"\*r\*-independent", "superseded wording"),
        (r"\banchors?\b", "internal term"), (r"oracle pass\b", "internal term"),
        (r"committed design\b", "internal term"), (r"RCWA-oracle success", "internal term"),
        (r"retract", "pilot wording"), (r"withdrawn", "pilot wording"), (r"7\.76", "retracted number"),
        (r"B-11", "retracted pretender"), (r"B-16", "retracted pretender"),
        (r"compute-host", "hostname"), (r"on acceptance", "conditional release"),
        (r"18[–-]36 s", "stale timing"), (r"73[–-]147 s", "stale timing"),
        (r"review_iter", "internal file"), (r"Declaration of Generative", "moved to Acknowledgements"),
        (r"(?<![A-Za-z0-9_])v[4-9]a?(?![A-Za-z0-9_])", "version tag"),
    ]
    PATHS = [r"docs/", r"results_v8/", r"results_pub/", r"src_v8/", r"figures_v9/"]
    da = body.find("## Data availability")
    scope = body if da < 0 else body[:da]
    # HTML comments are editorial markers (PENDING/AUTHOR notes), stripped before submission,
    # so they are exempt from the forbidden-pattern and path scans.
    scope = re.sub(r"<!--.*?-->", "", scope, flags=re.S)
    for pat, why in FORBIDDEN:
        for m in re.finditer(pat, scope, re.I if pat.islower() else 0):
            line = scope[:m.start()].count("\n") + 1
            fails.append(f"[B] forbidden {pat!r} ({why}) at line {line}: …{scope[max(0,m.start()-40):m.end()+40]!r}…")
    for pat in PATHS:
        for m in re.finditer(pat, scope):
            line = scope[:m.start()].count("\n") + 1
            fails.append(f"[B] internal path {pat!r} at line {line}")
    honest = len(re.findall(r"\bhonest", body, re.I))
    if results.endswith("results_pub"):
        # The superseded arm's headline numbers may appear only in a sentence that is
        # explicitly a two-arm contrast.  A sentence-level marker is the right test: the
        # abstract states 16 of 179 at its top and 53.1 % at its end, six hundred characters
        # apart, and the conclusion pairs them through "a sixfold difference".
        CONTRAST = re.compile(r"superseded|release|sixfold|two arms|both arms|as-submitted|legacy", re.I)
        s310 = (body.find("### 3.10"), body.find("### 3.11") if "### 3.11" in body else body.find("## 4.")) if "### 3.10" in body else (-1, -1)
        def in_two_arm(pos):
            return s310[0] >= 0 and s310[0] <= pos < s310[1]
        def sentence_of(pos):
            a = max(body.rfind(". ", 0, pos), body.rfind("\n\n", 0, pos), body.rfind("| ", 0, pos))
            b = min(x for x in (body.find(". ", pos), body.find("\n\n", pos), body.find(" |", pos), len(body)) if x >= 0)
            return body[a + 1:b]
        for pat, why in ((r"95/179", "legacy headline"), (r"95 of 179", "legacy headline"),
                         (r"53\.1 %", "legacy headline"), (r"\b53 %", "legacy headline (rounded)"), (r"confirms 84", "legacy headline"),
                         (r"47 of 179", "legacy tau sweep"), (r"25 of 179", "legacy tau sweep")):
            for m in re.finditer(pat, body):
                if in_two_arm(m.start()) or CONTRAST.search(sentence_of(m.start())):
                    continue
                fails.append(f"[B] {why} {m.group(0)!r} in a sentence that is not a two-arm contrast")

    # ---------------- Part C: end matter, abstract, tags
    todos = [(i + 1, l.strip()) for i, l in enumerate(txt.splitlines())
             if "TODO-AUTHOR" in l or "[CHECK]" in l or "AUTHOR-VERIFY" in l or "AUTHOR-SIGNOFF" in l]
    # the abstract proper: up to the Keywords line (or the section rule), comments excluded
    abstract = re.search(r"## Abstract\n(.*?)\n(?:\*\*Keywords:|---)", txt, re.S)
    aw = len(re.sub(r"<!--.*?-->", "", abstract.group(1), flags=re.S).split()) if abstract else 0
    if not skip_end:
        order = ["Acknowledgements", "Data availability", "Conflict of interest", "Author contributions", "ORCID"]
        pos = [body.find(f"## {h}") for h in order]
        if any(x < 0 for x in pos):
            fails.append(f"[C] missing end matter: {[h for h, x in zip(order, pos) if x < 0]}")
        elif pos != sorted(pos):
            fails.append("[C] end matter is out of order (Acknowledgements, Data availability, Conflict of interest, Author contributions, ORCID)")
        if not re.search(r"^\*\*Keywords:\*\*", body, re.M):
            fails.append("[C] no Keywords line")
        if aw > 300:
            fails.append(f"[C] abstract is {aw} words (> 300)")

    # ---------------- report
    # ---------------- Part D: counts are checked against the quantity they name
    # One whitelist of integers let "163 pretenders" through because 163 is the confirmed
    # count, and "see Table 9" exempted whole sentences.  Each quantity now has its own set
    # of admissible values, drawn from every level the artifacts report (pooled, per
    # structure, per seed, per threshold), and there are no text exemptions.
    if "pooled_v8" in J:
        tot = J["pooled_v8"]["totals"]
        per = J["pooled_v8"]["per_structure"]
        tsw = J["pooled_v8"].get("tau_sweep", {}).get("by_tau", {})
        pret = {int(tot["pretender"])} | {int(v["pretender"]) for v in per.values()}
        pret |= {int(v["pooled"]["pretender"]) for v in tsw.values()}
        pret |= {int(v["per_structure"][s]["pretender"]) for v in tsw.values() for s in v["per_structure"]}
        conf = {int(tot["oracle_pass"])} | {int(v.get("oracle_pass", -1)) for v in per.values()}
        n_pool = {int(tot["n_valid"]), int(tot["surrogate_pass"])} | {int(v["pooled"]["claimed"]) for v in tsw.values()}
        flags = set()
        for r in rel.values():
            pret.add(int(r["tier1a"]["pretender"])); conf.add(int(r["tier1a"]["rcwa_pass"]))
            flags.add(int(r["tier1b"]["flagged"]))
        flags.add(sum(int(r["tier1b"]["flagged"]) for r in rel.values()))
        mech = J.get("mechanism_v8", {}).get("per_structure", {})
        for sd in mech.values():
            for rule in ("random_search", "gradient_best_of_8", "single_start"):
                if rule in sd:
                    pret.add(int(sd[rule]["pretender"])); conf.add(int(sd[rule]["oracle_pass"]))
        ctl = J.get("control_analysis_v10", {}).get("conditions", {})
        for c in ctl.values():
            pret |= {int(c["secondary"]["pretender_base"]), int(c["secondary"]["pretender_control"])}
            pret |= {int(r["pretender_base"]) for r in c["per_structure"]} | {int(r["pretender_control"]) for r in c["per_structure"]}
        legacy = set()
        lp = ROOT / "results_v8/pooled_v8.json"
        if lp.exists():
            lj = json.load(open(lp)); legacy = {int(lj["totals"]["pretender"]), int(lj["totals"]["oracle_pass"])}
            legacy |= {int(v["pretender"]) for v in lj["per_structure"].values()}
        n = int(tot["n_valid"])
        CONTRAST = re.compile(r"superseded|release|sixfold|two arms|both arms|as-submitted|legacy", re.I)
        s310 = (body.find("### 3.10"), body.find("### 3.11") if "### 3.11" in body else body.find("## 4.")) if "### 3.10" in body else (-1, -1)
        def sent(pos):
            a = max(body.rfind(". ", 0, pos), body.rfind("\n", 0, pos))
            b = min(x for x in (body.find(". ", pos), body.find("\n", pos), len(body)) if x >= 0)
            return body[a + 1:b]
        t4 = J["pooled_v8"].get("t4")
        if t4:
            flags.add(int(t4.get("total_flags", 0)))
        rules = [(rf"(?<![\d.])(\d{{1,3}})(?:/| of )({n})\b", pret | conf | n_pool | flags, "of-179 count"),
                 (r"(?<!of )\b(\d{1,3}) pretenders?\b", pret, "pretender count"),
                 (r"(?<!of )\b(\d{1,3}) (?:oracle-confirmed|confirmed)\b", conf, "confirmed count"),
                 (r"(?<![/\d])(?<!of )\b(\d{1,3}) (?:Δ-)?flags\b", flags, "flag count")]
        for pat, allowed, label in rules:
            for m in re.finditer(pat, body):
                k = int(m.group(1))
                if k in allowed:
                    continue
                if s310[0] >= 0 and s310[0] <= m.start() < s310[1]:
                    continue
                if k in legacy and CONTRAST.search(sent(m.start())):
                    continue
                fails.append(f"[D] {label} {m.group(0)!r} matches no artifact value "
                             f"(allowed: {sorted(allowed)})")
    print(f"=== Part A: {sum(1 for c in checks if c[3])}/{len(checks)} numbers matched"
          + (f"; {len(skipped)} skipped ===" if skipped else " ==="))
    for s_ in skipped:
        print(f"  [A] SKIPPED: {s_}")
    for label, val, src, ok in checks:
        print(f"  {'ok ' if ok else 'FAIL'} {label:<28} {val:<18} <- {src}")
    print(f"\n=== Part B: {len([f for f in fails if f.startswith('[B]')])} forbidden-pattern hits ===")
    print(f"    'honest' occurrences (report only): {honest}")
    print(f"\n=== Part C: abstract {aw} words; {len(todos)} author tags ===")
    for ln, l in todos:
        print(f"  line {ln}: {l[:110]}")
    if fails:
        print("\n--- FAILURES ---")
        for f in fails:
            print(" ", f)
    print("\ncheck_manuscript: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manuscript", nargs="?", default="paper/manuscript_v10.md")
    ap.add_argument("--results", default="results_v8")
    ap.add_argument("--skip-endmatter", action="store_true")
    a = ap.parse_args()
    raise SystemExit(main(a.manuscript, a.results, a.skip_endmatter))
