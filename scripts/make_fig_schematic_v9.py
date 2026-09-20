#!/usr/bin/env python3
"""T20 — protocol schematic (Figure 1): (a) pipeline flow, (b) T1–T4 threshold boxes.
matplotlib only.  Writes figures_v9/fig1_protocol_schematic.{pdf,png}."""
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
PROFILE = os.environ.get("INVERSETL_PROFILE", "legacy")
if PROFILE not in ("legacy", "pub"):
    raise ValueError(f"unknown profile: {PROFILE}")
OUT = ROOT / os.environ.get("INVERSETL_FIGURES_DIR", "figures_v11" if PROFILE == "pub" else "figures_v9")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
FILL = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]

fig = plt.figure(figsize=(5.9, 3.0))   # IOP double column 15 cm
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.06)
axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax, lab in ((axa, "(a)"), (axb, "(b)")):
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.04, 1.04); ax.axis("off")
    ax.text(-0.02, 1.02, lab, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")

def box(ax, x, y, w, h, text, color, fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="k", lw=0.5, alpha=0.25))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=False)

def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="k"))

# ---- (a) pipeline: seven boxes stacked top-to-bottom (full panel width)
texts = [("1  TMM checkpoint regenerated with recipe of [1]" if PROFILE == "pub" else "1  TMM-pretrained checkpoint [2]"),
         ("2  strict load + fine-tune\n(n_train 350 each; seeds 42/123/777)" if PROFILE == "pub" else "2  strict load + fine-tune\n(n_train 350/350/300; seeds 42/123/777)"),
         "3  20 held-out RCWA targets per structure",
         "4  Adam on u in [0,1]^d — 8 restarts, 500 + 300 iterations",
         "5  commit the restart with the lowest final surrogate loss",
         ("6  RCWA reference at the committed geometry\n(per-wavelength adaptive order 9 / 13 / 17)" if PROFILE == "pub" else "6  RCWA oracle at the committed geometry\n(order 5; order 7 for seed 42)"),
         "7  Tier 1A/1B/1C + T1–T4 accounting"]
cols = [FILL[0], FILL[0], FILL[1], FILL[1], FILL[1], FILL[2], FILL[3]]
n = len(texts); H = 0.105; gap = (1.0 - n * H) / (n - 1) if n > 1 else 0
pos = [(0.02, 1.0 - H - i * (H + gap)) for i in range(n)]
for (x, y), t, c in zip(pos, texts, cols):
    box(axa, x, y, 0.96, H, t, c, fs=6.5)
for i in range(n - 1):
    arrow(axa, (0.5, pos[i][1]), (0.5, pos[i + 1][1] + H))
# ---- (b) T1–T4 boxes
tb = ["T1  RCWA MAE > 3 × surrogate MAE",
      "T2  any u within ε = 0.05 of box edge\n(vs d-dim uniform null)",
      "T3  NN distance < 0.5 ×\nmedian pair distance (vs N = 20 null)",
      "T4  K = 100 perturbations\nσ = 1 % box width\nflag if mean MAE > 5 %"]
for i, t in enumerate(tb):
    box(axb, 0.02, 0.80 - i * 0.2, 0.96, 0.16, t, FILL[3], fs=6.5)
axb.text(0.5, 0.03, "τ = 5 %, k = 2\nΔ-flag threshold = k × forward MAE", ha="center", va="bottom", fontsize=6.5)

# ---- text-fits-box check
fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
for ax in (axa, axb):
    patches = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    for txt in ax.texts:
        bb = txt.get_window_extent(renderer=r)
        if txt.get_text() in ("(a)", "(b)") or txt.get_text().startswith("τ ="):
            continue
        inside = any((pb := p.get_window_extent(renderer=r)) is not None and pb.x0 - 2 <= bb.x0 and bb.x1 <= pb.x1 + 2 and pb.y0 - 2 <= bb.y0 and bb.y1 <= pb.y1 + 2 for p in patches)
        if not inside:
            bad.append(txt.get_text().split("\n")[0])
print("text outside its box:", bad if bad else "none")
fig.savefig(OUT / "fig1_protocol_schematic.pdf", metadata={"CreationDate": None})
fig.savefig(OUT / "fig1_protocol_schematic.png", dpi=600)
print("wrote fig1_protocol_schematic.{pdf,png}")
