"""Raster sensitivity of the companion's Structure-B dataset labels (Paper 1, [1]).

Companion to scripts/paper1_b_feature_census_v11.py.  The census shows that 94 % of the
Structure-B samples behind Paper 1's printed tables have a feature below the 50 nm premise of
its 64 x 64 raster.  This script re-solves a stratified sample of those dataset entries with
the pinned solver (same vendored module, adaptive order, materials, precision) at grid 64
(control: must reproduce the archived dataset spectrum) and at grid 256, on all 100
wavelengths, and reports the grid-to-grid change of the label itself.  Nothing in the
dataset is modified.

Strata (by the sample's smallest in-plane feature, pixel = P/64):
    lt1px   < 1 pixel          10 samples
    1to2px  1-2 pixels         10 samples
    2px_50  >= 2 px, < 50 nm    5 samples
    ge50    >= 50 nm (premise)  5 samples
drawn with numpy seed 0 inside each stratum.

Usage (server, GPU):
    INVERSETL_PROFILE=pub python scripts/paper1_b_grid_delta.py --npz <dataset> [--select-only]
Writes results_pub/grid_delta_paper1B/<sel>.json (selection), one npz per sample and a
summary paper1_b_grid_delta_v11.json.
"""
import argparse, os, sys, time, json, pathlib, importlib
os.environ.setdefault("INVERSETL_PROFILE", "pub")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_v8"))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", default=str(ROOT / "results_pub" / "grid_delta_paper1B"))
ap.add_argument("--grids", default="64,256")
ap.add_argument("--select-only", action="store_true")
ap.add_argument("--device", default="auto")
a = ap.parse_args()
OUT = pathlib.Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

z = np.load(a.npz, allow_pickle=True)
X = np.asarray(z["params"], dtype=np.float64); names = [str(n) for n in z["param_names"]]
wl = np.asarray(z["wavelengths"], dtype=np.float64)
A_arc = np.asarray(z["A"], dtype=np.float64); rel = np.asarray(z["reliable"], dtype=bool)
P, Ro, Ri, Rd = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
feat = np.min(np.stack([Ro - Ri, Ri - Rd, P / 2 - Ro, 2 * Rd], 1), 1); px = P / 64
strata = {"lt1px": (feat < px, 10), "1to2px": ((feat >= px) & (feat < 2 * px), 10),
          "2px_50": ((feat >= 2 * px) & (feat < 50), 5), "ge50": (feat >= 50, 5)}
rng = np.random.default_rng(0); sel = []
for name, (mask, n) in strata.items():
    idx = np.flatnonzero(mask & rel.all(1))          # only fully reliable spectra as controls
    pick = sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    sel += [(name, int(i)) for i in pick]
    print(f"stratum {name:7s}: {len(idx):3d} eligible, picked {pick}")
json.dump(dict(dataset=os.path.basename(a.npz), seed=0, strata={k: v[1] for k, v in strata.items()},
               selection=[dict(stratum=s, index=i, min_feature_nm=float(feat[i]), pixel_nm=float(px[i]),
                               ring_width_nm=float(Ro[i] - Ri[i])) for s, i in sel]),
          open(OUT / "selection.json", "w"), indent=2)
if a.select_only:
    sys.exit(0)

import torch, common
from oracle import configure_solver
device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
cfg = common.STRUCTS["B"]; assert not cfg["dual"]
mod = importlib.import_module(cfg["rcwa_module"]); configure_solver(mod, None)
grids = [int(g) for g in a.grids.split(",")]
rows = []
for stratum, i in sel:
    f = OUT / f"grid_delta_paper1B_{i}.npz"
    if f.exists():
        print(f"[B i{i}] exists, skipping", flush=True); d = dict(np.load(f, allow_pickle=True)); rows.append({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items() if k.startswith(("mae", "delta", "index", "stratum", "time"))}); continue
    pdict = {k: float(X[i, j]) for j, k in enumerate(names)}
    print(f"[B i{i} {stratum}] " + ", ".join(f"{k}={v:.1f}" for k, v in pdict.items()), flush=True)
    out = dict(index=i, stratum=stratum, params=X[i], param_names=np.array(names), wavelengths=wl, A_archived=A_arc[i], reliable=rel[i])
    res = {}
    for g in grids:
        mod.RCWA_SETTINGS["grid"] = (g, g); t0 = time.time()
        r = mod.simulate_single(pdict, wl, metal="Cr", device=device)
        Ag = np.asarray(r[0], dtype=np.float64); dt = time.time() - t0
        m = rel[i] & np.isfinite(Ag) & np.isfinite(A_arc[i])
        d_arc = 100 * np.mean(np.abs(Ag[m] - A_arc[i][m])); dmax_arc = 100 * np.max(np.abs(Ag[m] - A_arc[i][m]))
        print(f"  grid {g:>4}: {dt:7.1f} s | vs archived(64): mean {d_arc:5.3f} pp, max {dmax_arc:5.2f} pp", flush=True)
        out[f"grid{g}"] = Ag; out[f"mae_vs_archived_grid{g}"] = d_arc; out[f"max_vs_archived_grid{g}"] = dmax_arc; out[f"time_grid{g}"] = dt
        res[g] = Ag
    if len(grids) >= 2:
        g0, g1 = grids[0], grids[-1]; m = np.isfinite(res[g0]) & np.isfinite(res[g1]) & rel[i]
        out["delta_mean_pp"] = 100 * np.mean(np.abs(res[g1][m] - res[g0][m])); out["delta_max_pp"] = 100 * np.max(np.abs(res[g1][m] - res[g0][m]))
        out["delta_signed_mean_pp"] = 100 * np.mean(res[g1][m] - res[g0][m])
        print(f"  grid {g0} -> {g1}: mean |dA| {out['delta_mean_pp']:5.3f} pp, max {out['delta_max_pp']:5.2f} pp, signed {out['delta_signed_mean_pp']:+5.3f} pp", flush=True)
    np.savez(f, **out)
    rows.append({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items() if k.startswith(("mae", "delta", "index", "stratum", "time", "max"))})
summ = dict(dataset=os.path.basename(a.npz), grids=grids, n=len(rows), rows=rows)
if rows and "delta_mean_pp" in rows[0]:
    dm = np.array([r["delta_mean_pp"] for r in rows]); dx = np.array([r["delta_max_pp"] for r in rows])
    c = np.array([r[f"mae_vs_archived_grid{grids[0]}"] for r in rows])
    summ.update(control_grid64_vs_archived_mean_pp=dict(median=float(np.median(c)), max=float(c.max())),
                delta_mean_pp=dict(min=float(dm.min()), median=float(np.median(dm)), max=float(dm.max())),
                delta_max_pp=dict(min=float(dx.min()), median=float(np.median(dx)), max=float(dx.max())))
    per = {}
    for s in strata:
        v = [r["delta_mean_pp"] for r in rows if r["stratum"] == s]
        if v: per[s] = dict(n=len(v), delta_mean_pp_median=float(np.median(v)), delta_mean_pp_max=float(max(v)))
    summ["per_stratum"] = per
json.dump(summ, open(OUT / "paper1_b_grid_delta_v11.json", "w"), indent=2)
print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))
