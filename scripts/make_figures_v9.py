"""Generate manuscript figures from the archived evidence (no recomputation).  T17 styling:
Okabe-Ito palette, hatches, 8-pt fonts, Type-42 PDF fonts, panel labels, IOP column widths
(single 8.5 cm, double 15 cm), 600 dpi, deterministic PDF metadata.
Reads INVERSETL_RESULTS_DIR (default results_v8) and writes figures_v9/ (or INVERSETL_FIGURES_DIR).
"""
import json
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = pathlib.Path(os.environ.get("INVERSETL_RESULTS_DIR", ROOT / "results_v8"))   # legacy arm by default; results_pub for the printed-pipeline arm
OUT = ROOT / os.environ.get("INVERSETL_FIGURES_DIR", "figures_v9")
os.makedirs(OUT, exist_ok=True)
TAU = 5.0
STRUCTS = ["B", "A", "C"]          # author decision (T01 ③): printed-r order B 0.96 > A 0.83 > C 0.65
R_PUB = {"A": 0.83, "B": 0.96, "C": 0.65}      # printed Paper 1 Table 5 median r (docs/paper1_published_values.md)
R_PRE = {"A": 0.72, "B": -0.07, "C": 0.34}     # preliminary pilot r at pre-registration
SEEDS = {"42": "", "123": "_s123", "777": "_s777"}
COL = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}                       # Okabe-Ito
HATCH = ["", "//", "xx"]
DOUBLE, SINGLE = 5.9, 3.35            # inches: IOP double column 15 cm, single column 8.5 cm (docs/mlst_requirements_v10.md)
DPI = 600
PDFMETA = {"CreationDate": None}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "axes.spines.top": False, "axes.spines.right": False})
numbers = {}


def rel(s, tag):
    return json.load(open(R / f"reliability_{s}{tag}_v8.json"))


def tick(s):
    return f"{s}\nr = {R_PUB[s]:+.2f} (prelim. {R_PRE[s]:+.2f})"


def label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9)


def save(fig, stem, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT / f"{stem}.pdf", metadata=PDFMETA)
    fig.savefig(OUT / f"{stem}.png", dpi=DPI)
    plt.close(fig)


def pct(arr, what):
    arr = np.asarray(arr, float)
    assert 0 < np.nanmax(arr) < 100, f"{what}: npz MAEs are stored in percent (src_v8/rcwa_validate.py)"
    return arr


# ---------- Fig 2: self-report vs oracle, all 9 runs ----------
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), sharex=True, sharey=True)
numbers["fig2_pretenders_per_panel"] = {}
for k, (ax, s) in enumerate(zip(axes, STRUCTS)):
    n_pre = n_val = 0
    for seed, tag in SEEDS.items():
        ps = rel(s, tag)["per_sample"]
        x = np.array(ps["mae_surr_pct"], float); y = np.array([np.nan if v is None else v for v in ps["mae_rcwa_pct"]], float)
        ok = np.isfinite(y)
        m = {"42": "o", "123": "s", "777": "^"}[seed]
        ax.scatter(x[ok], y[ok], marker=m, s=18, alpha=.8, color=COL[s], label=f"seed {seed}", zorder=2)
        n_pre += int(((x <= TAU) & (y > TAU) & ok).sum()); n_val += int(ok.sum())
    ax.fill_between([0, TAU], TAU, 40, color="0.85", zorder=0)
    ax.axvline(TAU, ls="--", lw=.8, color="k"); ax.axhline(TAU, ls="--", lw=.8, color="k")
    ax.set_title(f"Structure {s} · pretenders {n_pre}/{n_val}", fontsize=7, loc="center", pad=4)
    ax.set_xlabel("surrogate self-reported MAE (%)")
    ax.set_xlim(0, 6); ax.set_ylim(0, 36)
    label(ax, f"({'abc'[k]})")
    numbers["fig2_pretenders_per_panel"][s] = [n_pre, n_val]
axes[0].set_ylabel("reference-solver MAE (%)")
# one legend under the three panels: the shaded pretender region and the seed markers
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(facecolor="0.85", edgecolor="none", label="shaded: pretender region (Surr MAE ≤ τ, RCWA MAE > τ; τ = 5 %)")]
handles += [Line2D([], [], marker=m, linestyle="none", color="0.35", markersize=5, label=f"seed {seed}")
            for seed, m in (("42", "o"), ("123", "s"), ("777", "^"))]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0),
           handletextpad=0.5, columnspacing=1.4)
save(fig, "fig2_selfreport_vs_oracle", rect=[0, 0.08, 1, 1])

pool = json.load(open(R / "pooled_v8.json"))["per_structure"]   # pooled counts (used by fig3_r_vs_reliability)

# ---------- Fig 7: mechanism (pretenders / surrogate-claimed, Wilson bars) ----------
mech = json.load(open(R / "mechanism_v8.json"))["per_structure"]
rules = [("random_search", "random search\n(budget-matched)"), ("gradient_best_of_8", "gradient\nbest-of-8"), ("single_start", "gradient\nsingle start")]
fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
w = .25
numbers["fig7_mechanism"] = {}
for j, s in enumerate(STRUCTS):
    ks = [mech[s][k]["pretender"] for k, _ in rules]; ns = [mech[s][k]["surrogate_pass"] for k, _ in rules]
    rates = [100 * k / n if n else 0 for k, n in zip(ks, ns)]
    cis = [mech[s][k].get("wilson95_given_claimed", [k_ / n_, k_ / n_]) for (k, _), k_, n_ in zip(rules, ks, ns)]
    err = [[r - 100 * c[0] for r, c in zip(rates, cis)], [100 * c[1] - r for r, c in zip(rates, cis)]]
    ax.bar(np.arange(3) + (j - 1) * w, rates, width=w, color=COL[s], hatch=HATCH[j], edgecolor="k", linewidth=0.5, label=f"Structure {s}")
    ax.errorbar(np.arange(3) + (j - 1) * w, rates, yerr=err, fmt="none", ecolor="k", capsize=2, lw=0.8)
    for i, (k, n) in enumerate(zip(ks, ns)):
        ax.text(i + (j - 1) * w, 100 * cis[i][1] + 2, f"{k}/{n}", ha="center", fontsize=6)
    numbers["fig7_mechanism"][s] = {k: [k_, n_] for (k, _), k_, n_ in zip(rules, ks, ns)}
ax.set_xticks(np.arange(3)); ax.set_xticklabels([lab for _, lab in rules])
ax.set_ylabel("pretenders / surrogate-claimed (%)"); ax.set_ylim(0, 118)
ax.legend(frameon=False, loc="upper left", ncol=3)
save(fig, "fig7_mechanism")

# ---------- Fig 5: order-5 -> order-7, seed 42 (legacy arm only) ----------
if (R / "rcwa_A_v8_o7full.npz").exists():
    flips = {}
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), sharey=True)
    for k, (ax, s) in enumerate(zip(axes, STRUCTS)):
        o5 = np.load(R / f"rcwa_{s}_v8.npz"); o7 = np.load(R / f"rcwa_{s}_v8_o7full.npz")
        y5 = pct(o5["mae_rcwa"], f"rcwa_{s}"); y7 = pct(o7["mae_rcwa"], f"rcwa_{s}_o7full")
        ok = np.isfinite(y5) & np.isfinite(y7) & ~o5["failed"].astype(bool) & ~o7["failed"].astype(bool)
        cross = ((y5 <= TAU) != (y7 <= TAU)) & ok
        flip = int(cross.sum()); shifts = np.abs(y7 - y5)[ok]
        flips[s] = flip
        numbers.setdefault("fig5_order7", {})[s] = dict(flips=flip, max_shift_pp=float(shifts.max()), median_shift_pp=float(np.median(shifts)), n=int(ok.sum()))
        for i in np.where(ok)[0]:
            both_pass = (y5[i] <= TAU) and (y7[i] <= TAU)
            if cross[i]:
                ax.plot([0, 1], [y5[i], y7[i]], color="k", lw=1.8, marker="s", ms=3, zorder=3)
            else:
                ax.plot([0, 1], [y5[i], y7[i]], color=("0.6" if both_pass else COL[s]), alpha=(.5 if both_pass else .55), lw=1, marker="o", ms=2.5)
        ax.axhline(TAU, ls="--", lw=.8, color="k")
        ax.set_yscale("log"); ax.set_ylim(0.3, 40); ax.set_yticks([0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["order 5", "order 7"]); ax.set_xlim(-.3, 1.3)
        ax.text(0.03, 0.97, f"Structure {s}\n{flip}/{int(ok.sum())} cross τ, max shift {shifts.max():.1f} pp", transform=ax.transAxes, va="top")
        label(ax, f"({'abc'[k]})")
    axes[0].set_ylabel("reference-solver MAE at committed geometry (%)")
    save(fig, "fig5_order7_revalidation")
    o7j = json.load(open(R / "order7_revalidation_v8.json"))["per_structure"]
    assert flips == {"A": 1, "B": 6, "C": 3}, flips
    assert all(o7j[s]["flips"]["total"] == flips[s] for s in "ABC"), "flips disagree with order7_revalidation_v8.json"
    print("order-7 flips (seed 42):", flips)

# ---------- Fig 6: worst-case pretender spectrum + best confirmed design (T21) ----------
worst = None
for s in STRUCTS:
    for seed, tag in SEEDS.items():
        rc = np.load(R / f"rcwa_{s}{tag}_v8.npz", allow_pickle=True)
        mr = np.asarray(rc["mae_rcwa"], float); ms = np.asarray(rc["mae_surrogate"], float)
        cand = np.where((ms <= TAU) & np.isfinite(mr))[0]
        if len(cand):
            i = int(cand[np.argmax(mr[cand])])
            if worst is None or mr[i] > worst[0]:
                worst = (float(mr[i]), s, seed, tag, i)
mr_w, s, seed, tag, i = worst
rc = np.load(R / f"rcwa_{s}{tag}_v8.npz", allow_pickle=True); inv = np.load(R / f"inverse_{s}{tag}_v8.npz", allow_pickle=True)
mr = pct(rc["mae_rcwa"], "rcwa worst"); ms = pct(rc["mae_surrogate"], "surr worst")
i_ok = int(np.nanargmin(np.where(np.isfinite(mr), mr, np.inf)))          # best oracle-confirmed design of the same run
wl = rc["wavelengths"]; ch = "A_TE" if s == "C" else "A"
pnames = [str(x) for x in inv["param_names"]]


def unit_cell(ax, params, title):
    """Structure-B pattern drawn over a 3x3 tiling with the primitive cell outlined, so that a
    committed ring wider than the period visibly overlaps its periodic images."""
    import matplotlib.patches as mp
    q = {n: float(params[k]) for k, n in enumerate(pnames)}
    P = q["P"]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = P / 2 + dx * P, P / 2 + dy * P
            a = 0.55 if (dx, dy) == (0, 0) else 0.30
            ax.add_patch(mp.Circle((cx, cy), q["R_out"], fc=COL[s], ec="none", alpha=a))
            ax.add_patch(mp.Circle((cx, cy), q["R_in"], fc="w", ec="none"))
            ax.add_patch(mp.Circle((cx, cy), q["R_disk"], fc=COL[s], ec="none", alpha=a))
    ax.add_patch(mp.Rectangle((0, 0), P, P, fc="none", ec="k", lw=0.8, zorder=5))
    ax.set_xlim(-0.55 * P, 1.55 * P); ax.set_ylim(-0.55 * P, 1.55 * P); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{title}\n2R$_{{out}}$/P = {2*q['R_out']/P:.2f}", fontsize=5.5, pad=1)


fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), sharey=True)
for k, (ax, idx, ttl) in enumerate(((axes[0], i, "worst-case pretender"), (axes[1], i_ok, "best oracle-confirmed design"))):
    ax.plot(wl, rc[f"A_target_{ch}"][idx], "k-", lw=1.4, label="target")
    ax.plot(wl, inv[f"A_surr_{ch}"][idx], "-", color="#0072B2", lw=1.2, label="surrogate $f_{NN}$ at g(u*)")
    ax.plot(wl, rc[f"A_rcwa_{ch}"][idx], "--", color="#D55E00", lw=1.2, label="RCWA at g(u*)")
    ax.set_xlabel("wavelength (nm)")
    rc_txt = f"{mr[idx]:.2f}" if mr[idx] < 10 else f"{mr[idx]:.1f}"
    # panel title above the plot area, clear of the spectra
    ax.set_title(f"{ttl}\ntarget {idx+1}: Surr MAE {ms[idx]:.2f} % / RCWA MAE {rc_txt} %", fontsize=7, loc="center", pad=4)
    label(ax, f"({'ab'[k]})")
axes[0].set_ylabel("absorptance"); axes[0].set_ylim(0, 1); axes[1].legend(frameon=False, loc="lower right", fontsize=6.5)
if s == "B":
    # insets in the empty lower band of the panel, below the spectra
    unit_cell(axes[0].inset_axes([0.46, 0.02, 0.30, 0.30]), inv["best_params"][i], "committed g(u*)")
    unit_cell(axes[0].inset_axes([0.16, 0.02, 0.24, 0.24]), inv["true_params"][i], "true geometry")
save(fig, "fig6_worst_pretender_spectrum")
numbers["fig6_worst_pretender"] = dict(structure=s, seed=int(seed), index=i, mae_surr_pct=round(float(ms[i]), 3), mae_rcwa_pct=round(float(mr[i]), 2),
                                       best_confirmed_index=i_ok, best_confirmed_surr_pct=round(float(ms[i_ok]), 3), best_confirmed_rcwa_pct=round(float(mr[i_ok]), 3))
print("worst pretender:", numbers["fig6_worst_pretender"])

# ---------- Fig S1: per-parameter box-edge geometry, seed 42 (T08) ----------
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 3.0), gridspec_kw={"width_ratios": [8, 10, 7]})
numbers["figS1_t2"] = {}
for k, (ax, s) in enumerate(zip(axes, STRUCTS)):
    inv = np.load(R / f"inverse_{s}_v8.npz", allow_pickle=True); rc = np.load(R / f"rcwa_{s}_v8.npz", allow_pickle=True); d = rel(s, "")
    names = [str(x) for x in inv["param_names"]]
    u = (inv["best_params"] - inv["design_lo"]) / (inv["design_hi"] - inv["design_lo"])
    ms = pct(rc["mae_surrogate"], "S1 surr"); mr = np.asarray(rc["mae_rcwa"], float)
    valid = (~rc["failed"].astype(bool)) & np.isfinite(mr); pret = (ms <= TAU) & (mr > TAU) & valid; conf = (ms <= TAU) & (mr <= TAU) & valid
    P = inv["best_params"]; c = {n: P[:, i] for i, n in enumerate(names)}
    if s == "B":
        infeas = (c["R_out"] > 0.45 * c["P"]) | (c["R_in"] > c["R_out"] - 10) | (c["R_disk"] > c["R_in"] - 10)
    else:
        infeas = np.any(np.stack([c[w] > 0.9 * c["P"] for w in (("Wx", "Wy", "W2") if s == "A" else ("Wx", "Wy"))], 1), axis=1)
    ax.axhspan(0, 0.05, color="0.85", zorder=0); ax.axhspan(0.95, 1.0, color="0.85", zorder=0)
    rng = np.random.default_rng(0)
    for j, n in enumerate(names):
        x = j + rng.uniform(-0.18, 0.18, len(u))
        for m, mk, fc in ((pret & ~infeas, "o", COL[s]), (pret & infeas, "^", COL[s]), (conf & ~infeas, "o", "none"), (conf & infeas, "^", "none")):
            ax.scatter(x[m], u[m, j], marker=mk, s=16, facecolors=fc, edgecolors=COL[s], linewidths=0.7, zorder=2)
    in_band = int(((u < 0.05) | (u > 0.95)).any(axis=1).sum())
    assert in_band == d["taxonomy"]["t2"], (s, in_band, d["taxonomy"]["t2"])
    numbers["figS1_t2"][s] = dict(in_band=in_band, n=int(len(u)), pretenders=int(pret.sum()), infeasible=int(infeas[valid].sum()))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, ha="center", fontsize=6)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Structure {s} · T2 {in_band}/{len(u)}", fontsize=7, pad=10)
    label(ax, f"({'abc'[k]})")
axes[0].set_ylabel("u* (normalized design coordinate)")
axes[-1].scatter([], [], marker="o", facecolors="k", edgecolors="k", label="pretender")
axes[-1].scatter([], [], marker="o", facecolors="none", edgecolors="k", label="oracle-confirmed")
axes[-1].scatter([], [], marker="^", facecolors="none", edgecolors="k", label="infeasible")
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6,
           bbox_to_anchor=(0.53, 0.0))
save(fig, "figS1_box_edge_geometry", rect=(0, 0.1, 1, 1))

# ---------- Fig S2: held-out ensemble MAE vs reference-solver MAE (T07) ----------
if (R / "detector_bench_v8.json").exists():
    _db = json.load(open(R / "detector_bench_v8.json"))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5))
    for k, (ax, s) in enumerate(zip(axes, STRUCTS)):
        for seed, mk in ((42, "o"), (123, "s"), (777, "^")):
            r = _db["runs"][f"{s}_{seed}"]; x = np.array(r["_values"]["heldout_mae"], float); y = np.array(r["_labels"]["mae_rcwa_pct"], float)
            pret = np.array(r["_labels"]["pretender"], bool)
            ax.scatter(x[pret], y[pret], marker=mk, s=16, facecolors=COL[s], edgecolors=COL[s], linewidths=0.7, zorder=2)
            ax.scatter(x[~pret], y[~pret], marker=mk, s=16, facecolors="none", edgecolors=COL[s], linewidths=0.7, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axhline(TAU, ls="--", color="0.5", lw=0.8); ax.axvline(TAU, ls="--", color="0.5", lw=0.8)
        au = _db["pooled"][s]["heldout_mae"]
        ax.set_title(f"Structure {s}\nAUROC {au['auroc']:.2f} [{au['ci_lo']:.2f}, {au['ci_hi']:.2f}]", fontsize=7, loc="center", pad=4)
        ax.set_xlabel("held-out ensemble MAE (%)")
        label(ax, f"({'abc'[k]})")
    axes[0].set_ylabel("reference-solver MAE (%)")
    save(fig, "figS2_heldout_vs_rcwa")
    numbers["figS2_detector_alias"] = "figS2_heldout_vs_rcwa"

# ---------- Fig 3: r vs reliability — per-seed rates, DEFF-adjusted pooled bar, rho forest, pooled rho (T19) ----------
if (R / "stats_supplement_v9.json").exists():
    st = json.load(open(R / "stats_supplement_v9.json")); clus = st["clustering"]["per_structure"]; prho = st["pooled_rho"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.6), gridspec_kw={"width_ratios": [1.2, 1.3, 0.8]})
    ax = axes[0]; check = []
    for i, s in enumerate(STRUCTS):
        w = clus[s]["wilson_pretender_at_n_eff"]; rate = 100 * pool[s]["pretender"] / pool[s]["n_valid"]
        ax.bar(i, rate, width=.55, color="0.85", edgecolor="k", linewidth=0.5, hatch=HATCH[i], zorder=1)
        ax.errorbar(i, rate, yerr=[[rate - 100 * w[0]], [100 * w[1] - rate]], color="k", capsize=3, lw=1, zorder=3)
        for j, (seed, tag) in enumerate(SEEDS.items()):
            d = rel(s, tag); k, n = d["tier1a"]["pretender"], d["meta"]["n_valid"]; lo, hi = d["tier1a"]["pretender_ci"]
            x = i + (j - 1) * 0.15; y = 100 * k / n
            ax.errorbar(x, y, yerr=[[y - 100 * lo], [100 * hi - y]], fmt=["o", "s", "^"][j], ms=3.5, color=COL[s], ecolor=COL[s], elinewidth=0.8, capsize=0, zorder=4)
            check.append((s, seed, f"{k}/{n}", [round(100 * lo, 1), round(100 * hi, 1)], round(d["discrimination"]["rcwa_vs_surr_PRIMARY"]["rho"], 3),
                          [round(c, 3) for c in d["discrimination"]["rcwa_vs_surr_PRIMARY"]["ci95"]], round(d["discrimination"]["rcwa_vs_surr_PRIMARY"]["p"], 3)))
    ax.axhline(50, ls=":", lw=.8, color="gray"); ax.set_xticks(range(3))
    ax.set_xticklabels([f"{s}\nr {R_PUB[s]:+.2f}\n({R_PRE[s]:+.2f})" for s in STRUCTS], fontsize=6.5)
    ax.set_ylabel("pretenders / claimed (%)"); ax.set_ylim(0, 100); label(ax, "(a)")
    _fig3_note = True
    ax = axes[1]; y = 0
    for s in STRUCTS:
        for seed, tag in SEEDS.items():
            d = rel(s, tag)["discrimination"]["rcwa_vs_surr_PRIMARY"]; hollow = d["p"] < 0.05
            ax.errorbar(d["rho"], y, xerr=[[d["rho"] - d["ci95"][0]], [d["ci95"][1] - d["rho"]]], fmt="o", ms=3.5, color=COL[s],
                        mfc=("none" if hollow else COL[s]), elinewidth=0.8, capsize=2)
            ax.text(1.02, y, f"{s} s{seed}", va="center", fontsize=6); y += 1
        y += 0.5
    ax.axvline(0, color="k", lw=0.8); ax.set_xlim(-1, 1); ax.set_yticks([]); ax.set_xlabel("ρ(RCWA MAE, Surr MAE), Bonett–Wright 95 % CI"); label(ax, "(b)")
    ax = axes[2]
    for i, s in enumerate(STRUCTS):
        ax.bar(i, prho[s]["pooled_rho"], width=.55, color=COL[s], edgecolor="k", linewidth=0.5, hatch=HATCH[i])
        p = prho[s]["block_perm_p"]
        ax.text(i, max(prho[s]["pooled_rho"], 0) + 0.04, "p < 0.001" if p < 0.001 else f"p = {p:.3f}", ha="center", fontsize=6)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(STRUCTS); ax.set_ylim(-0.2, 0.9)
    ax.set_ylabel("seed-pooled ρ (block permutation)"); label(ax, "(c)")
    from matplotlib.patches import Patch as _Patch
    from matplotlib.lines import Line2D as _Line2D
    _h = [_Patch(facecolor="0.85", edgecolor="k", label="bar: 3-seed pooled rate, design-effect-adjusted Wilson 95 % CI"),
          _Line2D([], [], marker="o", linestyle="none", color="0.35", markersize=4, label="seed 42 (per-run rate, Wilson 95 % CI)"),
          _Line2D([], [], marker="s", linestyle="none", color="0.35", markersize=4, label="seed 123"),
          _Line2D([], [], marker="^", linestyle="none", color="0.35", markersize=4, label="seed 777")]
    fig.legend(handles=_h, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0), handletextpad=0.5, columnspacing=1.5)
    save(fig, "fig3_r_vs_reliability", rect=[0, 0.12, 1, 1])
    numbers["fig3_r_vs_reliability_check"] = check
    for c in check:
        print("fig2 check:", c)

# ---------- Fig 4: tau sensitivity (T19) ----------
pj = json.load(open(R / "pooled_v8.json"))
if "tau_sweep" in pj:
    ts = pj["tau_sweep"]; taus = ts["taus"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    for s in STRUCTS:
        r = [100 * (ts["by_tau"][str(t)]["per_structure"][s]["pretender_rate_given_claimed"] or 0) for t in taus]
        lo = [100 * ts["by_tau"][str(t)]["per_structure"][s]["bootstrap95_rate_given_claimed"][0] for t in taus]
        hi = [100 * ts["by_tau"][str(t)]["per_structure"][s]["bootstrap95_rate_given_claimed"][1] for t in taus]
        ax.fill_between(taus, lo, hi, color=COL[s], alpha=0.15, lw=0); ax.plot(taus, r, "-o", ms=3, color=COL[s], label=f"Structure {s}")
    rp = [100 * ts["by_tau"][str(t)]["pooled"]["pretender_rate_given_claimed"] for t in taus]
    ax.plot(taus, rp, "k--", lw=1.2, label="pooled")
    for t in taus:
        e = ts["by_tau"][str(t)]["pooled"]
        top = max(100 * ts["by_tau"][str(t)]["per_structure"][s]["bootstrap95_rate_given_claimed"][1] for s in STRUCTS)
        ax.text(t, top + 5, f"{e['pretender']}/{e['claimed']}", ha="center", va="bottom", fontsize=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))
    ax.axvline(TAU, ls=":", color="gray", lw=0.8); ax.set_xlabel("τ (%)"); ax.set_ylabel("pretenders / claimed (%)"); ax.set_ylim(0, 100); ax.legend(frameon=False); label(ax, "(a)")
    ax = axes[1]; sev = ts["severity_bins"]
    allr = []
    for s in STRUCTS:
        for seed, tag in SEEDS.items():
            ps = rel(s, tag)["per_sample"]; x = np.array(ps["mae_surr_pct"], float); y = np.array([np.nan if v is None else v for v in ps["mae_rcwa_pct"]], float)
            allr += y[(x <= TAU) & (y > TAU) & np.isfinite(y)].tolist()
    ax.hist(allr, bins=np.arange(5, 36, 1), color="0.7", edgecolor="k", linewidth=0.4)
    for e in (7.5, 10):
        ax.axvline(e, ls="--", color="k", lw=0.8)
    ax.text(0.98, 0.95, f"mild {sev['mild']} / moderate {sev['moderate']} / severe {sev['severe']}", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlabel("reference-solver MAE (%)\namong τ = 5 % pretenders"); ax.set_ylabel("designs"); label(ax, "(b)")
    save(fig, "fig4_tau_sensitivity")
    numbers["fig4_denominators"] = {str(t): ts["by_tau"][str(t)]["pooled"]["claimed"] for t in taus}

# ---------- Fig 8 (optional): oracle-select (only if T10/T56 results exist) ----------
if (R / "oracle_select_v8.json").exists():
    print("oracle_select_v8.json present — fig8 not yet implemented (T56 deferred)")

# ---------- numbers ----------

# ---------- Fig 8 (paper Figure 7): pretender rate under every condition of the audit ----------
def _wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c0 = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c0 - h), 100 * (c0 + h)
_pub = json.load(open(R / "pooled_v8.json")); _c3 = json.load(open(R / "control_analysis_v10_3seed.json")); _st = json.load(open(R / "stats_supplement_v9.json"))
_mech = json.load(open(R / "mechanism_v8.json"))["per_structure"]; _leg = json.load(open(ROOT / "results_v8/pooled_v8.json"))
_t = _pub["totals"]; _t25 = _pub["tau_sweep"]["by_tau"]["2.5"]["pooled"]
rows8 = [  # (label, k, n, group)
    ("main audit, gradient best-of-8", _t["pretender"], _t["n_valid"], "published pipeline, three seeds"),
    ("from-scratch surrogate (no transfer)", _c3["conditions"]["m0"]["pooled"]["pretender_control"], _t["n_valid"], "published pipeline, three seeds"),
    ("feasibility-constrained search", _c3["conditions"]["feas"]["pooled"]["pretender_control"], _t["n_valid"], "published pipeline, three seeds"),
    ("tolerance tightened to τ = 2.5 %", _t25["pretender"], _t25["claimed"], "published pipeline, three seeds"),
    ("gradient best-of-8", sum(_mech[s]["gradient_best_of_8"]["pretender"] for s in "ABC"), sum(_mech[s]["gradient_best_of_8"]["surrogate_pass"] for s in "ABC"), "seed 42, commitment rules"),
    ("budget-matched random search", sum(_mech[s]["random_search"]["pretender"] for s in "ABC"), sum(_mech[s]["random_search"]["surrogate_pass"] for s in "ABC"), "seed 42, commitment rules"),
    ("single mid-box start", sum(_mech[s]["single_start"]["pretender"] for s in "ABC"), sum(_mech[s]["single_start"]["surrogate_pass"] for s in "ABC"), "seed 42, commitment rules"),
    ("as-submitted release, same protocol", _leg["totals"]["pretender"], _leg["totals"]["n_valid"], "superseded release, three seeds"),
]
fig, ax = plt.subplots(figsize=(DOUBLE, 3.2))
ys = []; y = 0; last_group = None; group_pos = {}
for label_, k, n, grp in rows8:
    if grp != last_group and last_group is not None: y += 0.8
    last_group = grp; group_pos.setdefault(grp, []).append(y)
    rate = 100 * k / n; lo, hi = _wilson(k, n)
    color = "0.55" if "release" in grp else ("#0072B2" if "three seeds" in grp else "#009E73")
    ax.barh(y, rate, color=color, edgecolor="k", linewidth=0.5, height=0.7)
    ax.errorbar(rate, y, xerr=[[rate - lo], [hi - rate]], color="k", capsize=3, lw=1)
    ax.text(hi + 1.2, y, f"{k}/{n}", va="center", fontsize=7)
    ys.append((y, label_)); y += 1
cb = 100 * np.array(_st["clustering"]["pooled"]["cluster_bootstrap_95"])
ax.errorbar(100 * _t["pretender"] / _t["n_valid"], ys[0][0] + 0.28, xerr=[[100 * _t["pretender"] / _t["n_valid"] - cb[0]], [cb[1] - 100 * _t["pretender"] / _t["n_valid"]]], color="#D55E00", capsize=2, lw=1, ls="none")
ax.text(cb[1] + 1.2, ys[0][0] + 0.28, "target-clustered", va="center", fontsize=6, color="#D55E00")
ax.set_yticks([p for p, _ in ys]); ax.set_yticklabels([l for _, l in ys], fontsize=7); ax.invert_yaxis()
for grp, pos in group_pos.items():
    ax.text(-1.5, min(pos) - 0.55, grp, fontsize=7, fontstyle="italic", ha="left", va="center", color="0.3")
ax.set_xlabel("pretenders / surrogate-claimed designs (%)"); ax.set_xlim(0, 75)
for s in ("top", "right"): ax.spines[s].set_visible(False)
save(fig, "fig8_pretender_summary", rect=[0, 0, 1, 1])
numbers["fig8_rows"] = {l: [int(k), int(n)] for l, k, n, _ in rows8}

(OUT / "figure_numbers.json").write_text(json.dumps(numbers, indent=1))
print("wrote", sorted(os.listdir(OUT)))
# Final numbering (T20): 1 schematic (make_fig_schematic_v9.py), 2 selfreport, 3 r-vs-reliability, 4 tau, 5 order-7, 6 spectrum, 7 mechanism; S1, S2.
