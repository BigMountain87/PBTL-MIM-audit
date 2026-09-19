#!/usr/bin/env python3
"""Two-arm contrast — the audited release against the published pipeline.

    python3 scripts/pub_vs_legacy_v10.py

Reads rcwa_{S}{sx}_v8.npz and finetune_{S}{sx}_v8.json from results_v8 (legacy: fixed
Fourier order 5, complex128, as-submitted checkpoints) and results_pub (pub: per-wavelength
adaptive order, complex64, Johnson-Christy, printed-pipeline checkpoints), for every run
present in BOTH arms, and writes

    results_pub/pub_vs_legacy_v10.{json,md}

Counts only; nothing here re-simulates anything.  The manuscript reads its two-arm numbers
from the JSON, never from this script's stdout.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"legacy": ROOT / "results_v8", "pub": ROOT / "results_pub"}
TAU = 5.0
ORDER = ["B", "A", "C"]                      # published r, descending
SEEDS = [("", 42), ("_s123", 123), ("_s777", 777)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def newcombe(k1, n1, k2, n2):
    """Hybrid-score interval for p1 - p2 (percentage points)."""
    p1, p2 = 100 * k1 / n1, 100 * k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def run(arm_dir, s, sx):
    f = arm_dir / f"rcwa_{s}{sx}_v8.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    ms, mr, failed = d["mae_surrogate"], d["mae_rcwa"], d["failed"]
    ok = ~failed
    claimed = (ms <= TAU) & ok
    delta = (mr - ms)[ok]
    ft = arm_dir / f"finetune_{s}{sx}_v8.json"
    fwd = json.loads(ft.read_text()).get("final_test_mae_pct") if ft.exists() else None
    return dict(_claimed=claimed, _pret=claimed & (mr > TAU),      # per-design arrays for the cluster bootstrap
                n_valid=int(ok.sum()), claimed=int(claimed.sum()),
                confirmed=int((claimed & (mr <= TAU)).sum()),
                pretender=int((claimed & (mr > TAU)).sum()),
                median_delta_pp=float(np.median(delta)), max_delta_pp=float(delta.max()),
                mean_surr_mae_pct=float(ms[ok].mean()), mean_rcwa_mae_pct=float(mr[ok].mean()),
                forward_test_mae_pct=fwd)


def main():
    runs, pooled = {}, {a: dict(claimed=0, pretender=0, confirmed=0, n_valid=0) for a in ARMS}
    for s in ORDER:
        for sx, seed in SEEDS:
            both = {a: run(d, s, sx) for a, d in ARMS.items()}
            if any(v is None for v in both.values()):
                continue                      # only runs finished in both arms are compared
            key = f"{s}_{seed}"
            runs[key] = both
            for a in ARMS:
                for k in pooled[a]:
                    pooled[a][k] += both[a][k]

    # cluster bootstrap over targets: within each arm x structure the three seeds share one
    # target set, so a target's three appearances are resampled together (4000 draws)
    rng = np.random.default_rng(20260912)
    per = {a: {s: [runs[k][a] for k in runs if k.startswith(s + "_")] for s in ORDER} for a in ARMS}
    boots = []
    for _ in range(4000):
        rate = {}
        for a in ARMS:
            kc = kp = 0
            for s in ORDER:
                rs = per[a][s]
                if not rs:
                    continue
                n = len(rs[0]["_claimed"]); idx = rng.integers(0, n, n)
                for r in rs:
                    kc += int(r["_claimed"][idx].sum()); kp += int(r["_pret"][idx].sum())
            rate[a] = kp / kc if kc else np.nan
        boots.append(100 * (rate["legacy"] - rate["pub"]))
    boots = np.array(boots)
    cluster_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    for k in runs:                              # arrays are not JSON
        for a in ARMS:
            runs[k][a] = {kk: v for kk, v in runs[k][a].items() if not kk.startswith("_")}

    def fisher(a, b):
        return float(sps.fisher_exact([[a["pretender"], a["claimed"] - a["pretender"]],
                                       [b["pretender"], b["claimed"] - b["pretender"]]])[1])

    out = dict(
        tau_pct=TAU,
        definition="pretender = surrogate MAE <= tau AND reference-solver MAE > tau",
        arms={"legacy": "as-submitted release: fixed Fourier order 5, complex128, "
                        "2026-03 checkpoints and data pools",
              "pub": "printed pipeline: per-wavelength adaptive order 9/13/17, complex64, "
                     "Johnson-Christy, redesign pools"},
        runs_compared=sorted(runs),
        caveat="the two arms differ in checkpoints, data pools and solver settings at once; "
               "this contrast is release-vs-printed-pipeline, not an ablation of any one factor. "
               "Targets differ between arms, so the runs are independent samples, not paired.",
        per_run={k: v for k, v in runs.items()},
        pooled={a: dict(p, pretender_rate_pct=100 * p["pretender"] / p["claimed"],
                        pretender_ci95=wilson(p["pretender"], p["claimed"]),
                        ci_note="nominal Wilson: the three seeds share one target set per structure")
                for a, p in pooled.items()},
        per_structure_fisher_note="nominal: seeds within an arm share targets, so the per-structure "
                                  "counts are correlated replicates rather than independent designs",
        pooled_difference_pp=dict(
            estimate=100 * pooled["legacy"]["pretender"] / pooled["legacy"]["claimed"]
            - 100 * pooled["pub"]["pretender"] / pooled["pub"]["claimed"],
            newcombe_ci95=newcombe(pooled["legacy"]["pretender"], pooled["legacy"]["claimed"],
                                   pooled["pub"]["pretender"], pooled["pub"]["claimed"]),
            fisher_p=fisher(pooled["legacy"], pooled["pub"]),
            cluster_bootstrap_ci95=cluster_ci,
            cluster_bootstrap_note="4000-draw percentile bootstrap resampling targets within each arm x structure, "
                                   "a target's three seeds together (seed 20260912)",
            note="nominal: the shared targets within a run are not accounted for here; "
                 "the design effect of section 3.4 applies to the pooled counts."),
        per_structure={
            s: dict(legacy=dict(pretender=sum(runs[k]["legacy"]["pretender"] for k in runs if k.startswith(s + "_")),
                                claimed=sum(runs[k]["legacy"]["claimed"] for k in runs if k.startswith(s + "_"))),
                    pub=dict(pretender=sum(runs[k]["pub"]["pretender"] for k in runs if k.startswith(s + "_")),
                             claimed=sum(runs[k]["pub"]["claimed"] for k in runs if k.startswith(s + "_"))))
            for s in ORDER if any(k.startswith(s + "_") for k in runs)},
    )
    for s, d in out["per_structure"].items():
        d["fisher_p"] = fisher(d["legacy"], d["pub"])

    md = ["# Two-arm contrast — audited release vs published pipeline", "",
          f"Runs compared (finished in both arms): {', '.join(out['runs_compared'])}.  "
          f"tau = {TAU:.0f} %.", "",
          "| run | forward test MAE | surrogate-claimed | oracle-confirmed | pretenders | median Δ | max Δ |",
          "|---|---|---|---|---|---|---|"]
    for k in out["runs_compared"]:
        for a in ("legacy", "pub"):
            r = runs[k][a]
            md.append(f"| {k} ({a}) | {r['forward_test_mae_pct']:.2f} % | {r['claimed']}/{r['n_valid']} | "
                      f"{r['confirmed']}/{r['n_valid']} | **{r['pretender']}** | "
                      f"{r['median_delta_pp']:+.2f} pp | {r['max_delta_pp']:+.2f} pp |")
    p = out["pooled"]
    md += ["", "## Pooled", "",
           "| arm | pretenders / claimed | rate [Wilson 95 %] |", "|---|---|---|"]
    for a in ("legacy", "pub"):
        md.append(f"| {a} | {p[a]['pretender']}/{p[a]['claimed']} | {p[a]['pretender_rate_pct']:.1f} % "
                  f"[{p[a]['pretender_ci95'][0]:.1f}, {p[a]['pretender_ci95'][1]:.1f}] |")
    d = out["pooled_difference_pp"]
    md += ["", f"Difference {d['estimate']:.1f} pp, Newcombe 95 % CI "
              f"[{d['newcombe_ci95'][0]:.1f}, {d['newcombe_ci95'][1]:.1f}], Fisher exact p = {d['fisher_p']:.2e}; "
              f"target-cluster bootstrap 95 % CI [{d['cluster_bootstrap_ci95'][0]:.1f}, {d['cluster_bootstrap_ci95'][1]:.1f}].",
           "", f"All intervals and tests on this page are nominal — {d['note']} The per-structure Fisher tests "
               "below carry the same caveat.",
           "", "## Per structure (all compared seeds pooled)", "",
           "| structure | legacy | pub | Fisher p |", "|---|---|---|---|"]
    for s, v in out["per_structure"].items():
        pv = f"{v['fisher_p']:.3f}" if v["fisher_p"] >= 1e-3 else f"{v['fisher_p']:.1e}"
        md.append(f"| {s} | {v['legacy']['pretender']}/{v['legacy']['claimed']} | "
                  f"{v['pub']['pretender']}/{v['pub']['claimed']} | {pv} |")
    md += ["", "## Caveat", "", out["caveat"]]

    (ARMS["pub"] / "pub_vs_legacy_v10.json").write_text(json.dumps(out, indent=2))
    (ARMS["pub"] / "pub_vs_legacy_v10.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
