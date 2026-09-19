#!/usr/bin/env python3
"""T22 — supplementary tables S1–S4 (+ S5 placeholder) generated from the JSON/npz artifacts.
S1 per-run Tier-1A/1B/1C/taxonomy/discrimination table (+ results/table_s1_v8.csv)
S2 mechanism table with both denominators and both Fisher families
S3 per-design order-5 vs order-7 table (legacy arm only; skipped if o7full is absent)
S4 verbatim protocol (docs/v8_protocol.md) with the honest modification-time header
S5 placeholder pointing at table_S6 (T21) and Fig S1/S2
Usage: python3 src_v8/make_table_s1.py [--results results_v8] [--out paper/supplementary_v10.md]
"""
from __future__ import annotations
import argparse, csv, datetime, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"": 42, "_s123": 123, "_s777": 777}
TAU = 5.0


def ci(x):
    return f"[{100*x[0]:.1f}, {100*x[1]:.1f}]"


def table_s1(R, L, heading="## S1. Per-run reliability accounting", write_csv=True):
    """Per-run Tier-1A/1B/1C/taxonomy/discrimination rows from results dir R, appended to L."""
    L += [heading, "",
          "| S | seed | n_valid | surrogate-claimed [Wilson 95 %] | oracle-confirmed [Wilson] | pretenders [Wilson] | Δ-flagged [Wilson] | flag thr (%) | modes | T1 | T2 | T3 | T4 | tail RSD | endpoint spread (u) | ρ(RCWA, Surr) [CI95] | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []; tot = dict(pret=0, n=0); per_s = {}
    for s in "ABC":
        ps = pn = 0
        for sx, seed in SEEDS.items():
            d = json.load(open(R / f"reliability_{s}{sx}_v8.json"))
            m, a, b, c, tx, di = d["meta"], d["tier1a"], d["tier1b"], d["tier1c"], d["taxonomy"], d["discrimination"]["rcwa_vs_surr_PRIMARY"]
            row = dict(structure=s, seed=seed, n_valid=m["n_valid"], surrogate_pass=a["surrogate_pass"], surrogate_pass_ci=ci(a["surrogate_pass_ci"]),
                       rcwa_pass=a["rcwa_pass"], rcwa_pass_ci=ci(a["rcwa_pass_ci"]), pretender=a["pretender"], pretender_ci=ci(a["pretender_ci"]),
                       flagged=b["flagged"], flagged_ci=ci(b["flagged_ci"]), flag_threshold_pct=round(b["flag_threshold_pct"], 2), mode_count=tx["mode_count"],
                       t1=tx["t1"], t2=tx["t2"], t3=tx["t3"], t4=tx["t4"], tail_rsd_mean=round(c["tail_rsd_mean"], 4), endpoint_spread_u_mean=round(c["endpoint_spread_u_mean"], 3),
                       rho=round(di["rho"], 3), rho_ci95=f"[{di['ci95'][0]:.2f}, {di['ci95'][1]:.2f}]", p=round(di["p"], 3))
            rows.append(row); ps += a["pretender"]; pn += m["n_valid"]
            pstr = f"{di['p']:.3f}" if di["p"] >= 0.001 else f"{di['p']:.1e}"     # never print an exact 0.000
            L.append(f"| {s} | {seed} | {row['n_valid']} | {row['surrogate_pass']} {row['surrogate_pass_ci']} | {row['rcwa_pass']} {row['rcwa_pass_ci']} | {row['pretender']} {row['pretender_ci']} | "
                     f"{row['flagged']} {row['flagged_ci']} | {row['flag_threshold_pct']} | {row['mode_count']} | {row['t1']} | {row['t2']} | {row['t3']} | {row['t4']} | {row['tail_rsd_mean']} | "
                     f"{row['endpoint_spread_u_mean']} | {row['rho']:+.2f} {row['rho_ci95']} | {pstr} |")
        per_s[s] = (ps, pn); tot["pret"] += ps; tot["n"] += pn
    L.append(f"| **all** | | {tot['n']} | | | **{tot['pret']}/{tot['n']}** ({', '.join(f'{s} {per_s[s][0]}/{per_s[s][1]}' for s in 'ABC')}) | | | | | | | | | | | |")
    if write_csv:               # never rewrite the other arm's archive from here
        with open(R / "table_s1_v8.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("S1 pretender intervals:", [(r["structure"], r["seed"], r["pretender_ci"]) for r in rows])
    print(f"S1 totals: {tot['pret']}/{tot['n']} " + str({s: per_s[s] for s in "ABC"}))


def table_s2(R, L, heading="## S2. Commitment-rule comparison (seed 42)"):
    mech = json.load(open(R / "mechanism_v8.json")).get("per_structure", {})
    if not mech:
        L += ["", heading, "", "*(not computed in this results view: the commitment-rule controls exist only for the protocol's 20 targets)*"]
        return
    L += ["", heading, "",
          "| S | rule | surrogate-claimed | oracle-confirmed | pretenders (of n) | pretenders / claimed [Wilson] | mean oracle MAE (%) | Fisher p vs best-of-8 (design-level) | Fisher p (conditional on claimed) |",
          "|---|---|---|---|---|---|---|---|---|"]
    exp = {}
    for s in "ABC":
        m = mech[s]
        for rule, lab in (("gradient_best_of_8", "gradient best-of-8"), ("random_search", "random search (budget-matched)"), ("single_start", "single start (mid-box)")):
            b = m[rule]; n = m["n"]
            pd_ = {"random_search": ("fisher_p_random_vs_best8", "fisher_p_random_vs_best8_conditional"), "single_start": ("fisher_p_single_vs_best8", "fisher_p_single_vs_best8_conditional")}.get(rule)
            f1 = f"{m[pd_[0]]:.3f}" if pd_ else "—"; f2 = f"{m[pd_[1]]:.3f}" if pd_ else "—"
            L.append(f"| {s} | {lab} | {b['surrogate_pass']} | {b['oracle_pass']} | {b['pretender']}/{n} | {b['pretender']}/{b['surrogate_pass']} {ci(b['wilson95_given_claimed'])} | {b['mean_oracle_mae_pct']:.2f} | {f1} | {f2} |")
            exp[(s, rule)] = (b["pretender"], b["surrogate_pass"])
    print("S2 pretenders random/single/best8:", [exp[(s, 'random_search')][0] for s in 'ABC'], [exp[(s, 'single_start')][0] for s in 'ABC'], [exp[(s, 'gradient_best_of_8')][0] for s in 'ABC'], "C random", exp[('C', 'random_search')])


def table_s3(R, L, heading="## S3. Order-5 vs order-7 reference-solver MAE per committed design (seed 42)"):
    """Legacy (fixed-order) arm only: the pub arm has no order-7 artifacts and no reason for them (§3.5)."""
    if (R / "rcwa_A_v8_o7full.npz").exists():
        o7j = json.load(open(R / "order7_revalidation_v8.json"))["per_structure"]
        L += ["", heading, "",
              "| S | # | Surr MAE (%) | RCWA MAE order 5 (%) | RCWA MAE order 7 (%) | shift (pp) | pass@5 | pass@7 | status |", "|---|---|---|---|---|---|---|---|---|"]
        tots = {}
        for s in "ABC":
            d5 = np.load(R / f"rcwa_{s}_v8.npz"); d7 = np.load(R / f"rcwa_{s}_v8_o7full.npz")
            m5, m7, ms = np.asarray(d5["mae_rcwa"], float), np.asarray(d7["mae_rcwa"], float), np.asarray(d5["mae_surrogate"], float)
            assert 0 < np.nanmax(m5) < 100 and 0 < np.nanmax(m7) < 100, "MAEs must be in percent"
            v = (~d5["failed"].astype(bool)) & (~d7["failed"].astype(bool)) & np.isfinite(m5) & np.isfinite(m7)
            p5, p7 = (m5 <= TAU) & v, (m7 <= TAU) & v
            fl = int(((p5 != p7) & v).sum()); tots[s] = (int(p5.sum()), int(p7.sum()), fl)
            for i in range(len(m5)):
                if not v[i]:
                    L.append(f"| {s} | {i+1} | {ms[i]:.2f} | — | — | — | — | — | degenerate |"); continue
                st = "stable" if p5[i] == p7[i] else ("pass→fail" if p5[i] else "fail→pass")
                L.append(f"| {s} | {i+1} | {ms[i]:.2f} | {m5[i]:.2f} | {m7[i]:.2f} | {m7[i]-m5[i]:+.2f} | {int(p5[i])} | {int(p7[i])} | {st} |")
            assert tots[s] == (o7j[s]["order5"]["oracle_pass"], o7j[s]["order7"]["oracle_pass"], o7j[s]["flips"]["total"]), (s, tots[s])
        L.append(f"| **totals** | | | | | | {' / '.join(f'{s} {tots[s][0]}→{tots[s][1]} (flips {tots[s][2]})' for s in 'ABC')} | | |")
        print("S3 totals (pass@5→pass@7, flips):", tots)


def table_s6(R, L, results, heading="## S6. Per-design table"):
    """Per-design table produced by scripts/make_supp_table_v9.py --results <results> (paper/supplementary_v9<tag>.md)."""
    L += ["", heading]
    tag = "" if results == "results_v8" else f"_{results}"
    s6 = ROOT / "paper" / f"supplementary_v9{tag}.md"
    if s6.exists():
        body = s6.read_text().splitlines()
        L += [""] + [l for l in body if not l.startswith("# ")]
    else:
        L += ["", f"*(run `python3 scripts/make_supp_table_v9.py --results {results}` first)*"]


def main(results, out, legacy_results):
    R = ROOT / results
    L = ["# Supplementary Information", "", f"> Generated by `src_v8/make_table_s1.py` from `{results}/`"
         + (f" (Section S8 from `{legacy_results}/`)" if legacy_results else "") + ". τ = 5 %.", ""]
    table_s1(R, L)
    table_s2(R, L)
    # S3 exists only for the fixed-order arm; when the main arm is pub, it is drawn from the legacy dir and labelled so
    if (R / "rcwa_A_v8_o7full.npz").exists():
        table_s3(R, L)
    elif legacy_results and (ROOT / legacy_results / "rcwa_A_v8_o7full.npz").exists():
        table_s3(ROOT / legacy_results, L, heading="## S3. Order-5 vs order-7 reference-solver MAE per committed design (seed 42; **as-submitted arm**, `"
                 + legacy_results + "/`)")
    # ---------------- S4
    proto = ROOT / "docs" / "v8_protocol.md"
    mtime = datetime.datetime.fromtimestamp(proto.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L += ["", "## S4. Pre-specified protocol (verbatim)", "",
          f"> Frozen 2026-06-10 (stated in the file) for the audit of the as-submitted release, and re-run unchanged on the published pipeline in September 2026 (manuscript Section 2.7); file last modified {mtime}. Two items post-date the freeze (manuscript Section 2.7): the restart-seeding sentence of §3.4, and the multi-seed parenthesis of §5 item 5, which was appended after the seed-42 runs; the seed-123/777 artifacts pre-date the file's last modification, so the record does not establish that the extension was written before those runs.", "",
          "```markdown", proto.read_text().rstrip(), "```"]
    # ---------------- S5 (pilot-stage defects, moved from the main text by T23)
    s5 = ROOT / "docs" / "pilot_defects_s5.md"
    if s5.exists():
        L += ["", s5.read_text().rstrip()]
    # ---------------- S6: per-design table (folded in from scripts/make_supp_table_v9.py)
    table_s6(R, L, results)
    L += ["", "The machine-readable version, with the committed and true geometry of every design, the",
          "smallest feature, the pixel size, the generator rules each design violates and the",
          "peak-wavelength shift, is `table_S6_pretenders_v8.csv` in the released archive.", ""]
    # ---------------- S7: supplementary figures
    L += ["## S7. Supplementary figures", "",
          "> **Figure S1.** Per-parameter position of every committed geometry inside the design box",
          "> (seed 42; one column per design parameter, y = the normalized coordinate u*). Grey bands",
          "> mark the T2 box-edge region, within 0.05 of either bound. Filled markers are pretenders and",
          "> open markers oracle-confirmed designs; triangles are geometries that violate a generator",
          "> constraint. The in-band counts equal the T2 totals of Supplementary Table S1.", "",
          "> **Figure S2.** Held-out ensemble error against reference-solver error for every valid",
          "> committed geometry, per structure and training seed (log-log). The held-out ensemble error",
          "> is the mean absolute error of the two surrogates that did *not* commit the design, so it is",
          "> available without any solver call. Dashed lines mark tau = 5 %; filled markers are",
          "> pretenders. Panel titles give the pooled AUROC for separating pretenders from",
          "> oracle-confirmed designs, with its bootstrap 95 % interval.", ""]
    # ---------------- S8: the as-submitted arm (Section 3.10), same tables, legacy artifacts
    if legacy_results:
        RL = ROOT / legacy_results
        L += ["## S8. The as-submitted arm of Section 3.10", "",
              f"The identical protocol run end to end on the release of [1] that the published pipeline superseded "
              f"(upstream commit `920b1bd`; fixed Fourier order 5, complex128; datasets struct_A_vis_500 (479 good, 380–780 nm), "
              f"struct_B_500 (461, 400–1800 nm), struct_C_500 (400, 400–1800 nm); C trained on 300 samples with a [50, 400] nm normalization box "
              f"and 13 physics features). Artifacts: `{legacy_results}/`. The tables below are the S1, S2 and S6 equivalents for that arm.", ""]
        table_s1(RL, L, heading="### S8.1. Per-run reliability accounting (as-submitted arm)", write_csv=False)
        table_s2(RL, L, heading="### S8.2. Commitment-rule comparison (seed 42; as-submitted arm)")
        table_s6(RL, L, legacy_results, heading="### S8.3. Per-design table (as-submitted arm)")
        L += [""]
    Path(out).write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results_v8"); ap.add_argument("--out", default=str(ROOT / "paper" / "supplementary_v10.md"))
    ap.add_argument("--legacy-results", default=None, help="if given, append Section S8 (S1/S2/S6 of the as-submitted arm) from this dir")
    a = ap.parse_args(); main(a.results, a.out, a.legacy_results)
