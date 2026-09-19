"""Re-solve one committed design at the pinned solver settings on two real-space grids.

The pub pipeline rasterises geometry on a 64 x 64 grid; docs/notes_rcwa_grid_truncation_v11.md
asks how much that raster moves the oracle's absorptance.  This solves the archived design
with the vendored solver (same module, same adaptive order, same materials) at grid 64
(control: must reproduce the archived GPU spectrum) and at a finer grid, on a wavelength
subsample, and reports MAE-vs-target on each grid and the grid-to-grid difference.

Usage: INVERSETL_PROFILE=pub python scripts/grid_delta_resolve.py <S> <seed> <index> [--n-wl 20] [--grids 64,256] [--out DIR]
CPU is fine (a design at N <= 13 is minutes; N = 17 is ~5 min per wavelength on an 8 GB Mac).
Writes <out>/grid_delta_<S>_s<seed>_<index>.npz and prints one summary line per grid.
"""
import argparse, os, sys, time, pathlib, importlib
os.environ.setdefault("INVERSETL_PROFILE", "pub")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_v8"))
import numpy as np, torch
import common
from oracle import configure_solver

ap = argparse.ArgumentParser()
ap.add_argument("structure"); ap.add_argument("seed", type=int); ap.add_argument("index", type=int)
ap.add_argument("--n-wl", type=int, default=20, help="0 = all archived wavelengths")
ap.add_argument("--grids", default="64,256")
ap.add_argument("--out", default=str(ROOT / "results_pub" / "grid_delta"))
ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
a = ap.parse_args()
device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)

s, i = a.structure, a.index
sx = common.seed_sfx(a.seed)
cfg = common.STRUCTS[s]
mod = importlib.import_module(cfg["rcwa_module"])
prov = configure_solver(mod, None)
a_keys = ("A_TE", "A_TM") if cfg["dual"] else ("A",)

inv = np.load(common.RESULTS / f"inverse_{s}{sx}_v8.npz", allow_pickle=True)
arc = np.load(common.RESULTS / f"rcwa_{s}{sx}_v8.npz", allow_pickle=True)
names = [str(x) for x in inv["param_names"]]
pdict = {k: float(inv["best_params"][i, j]) for j, k in enumerate(names)}
wl_all = inv["wavelengths"].astype(np.float64)
sel = np.arange(len(wl_all)) if a.n_wl == 0 else np.linspace(0, len(wl_all) - 1, a.n_wl).round().astype(int)
wl = wl_all[sel]
tgt = {k: inv[f"A_target_{k}"][i][sel] for k in a_keys}
gpu = {k: arc[f"A_rcwa_{k}"][i][sel] for k in a_keys}

print(f"[{s} s{a.seed} i{i}] " + ", ".join(f"{k}={v:.1f}" for k, v in pdict.items()), flush=True)
print(f"  archived: surrogate MAE {float(arc['mae_surrogate'][i]):.2f} %, RCWA MAE {float(arc['mae_rcwa'][i]):.2f} % "
      f"(N {int(arc['orders_min'][i])}-{int(arc['orders_max'][i])}); {len(wl)}/{len(wl_all)} wavelengths", flush=True)

out = dict(structure=s, seed=a.seed, index=i, wavelengths=wl, wl_index=sel,
           params=np.array([pdict[k] for k in names]), param_names=np.array(names),
           mae_rcwa_archived=float(arc["mae_rcwa"][i]), mae_surrogate=float(arc["mae_surrogate"][i]))
for k in a_keys:
    out[f"target_{k}"] = tgt[k]; out[f"gpu64_{k}"] = gpu[k]
res_by_grid = {}
for g in [int(x) for x in a.grids.split(",")]:
    mod.RCWA_SETTINGS["grid"] = (g, g)
    t0 = time.time()
    r = mod.simulate_single(pdict, wl, metal="Cr", device=device)
    dt = time.time() - t0
    # same unpacking as oracle.simulate_cached: (A, R, T) or (A_TE, R, T, A_TM, R, T)
    A = ({"A_TE": np.asarray(r[0], dtype=np.float64), "A_TM": np.asarray(r[3], dtype=np.float64)}
         if cfg["dual"] else {"A": np.asarray(r[0], dtype=np.float64)})
    mae_t = 100 * np.mean([np.mean(np.abs(A[k] - tgt[k])) for k in a_keys])
    mae_g = 100 * np.mean([np.mean(np.abs(A[k] - gpu[k])) for k in a_keys])
    mx_g = 100 * max(np.max(np.abs(A[k] - gpu[k])) for k in a_keys)
    print(f"  grid {g:>4}: {dt:7.1f} s | MAE vs target {mae_t:5.2f} % | vs archived GPU(64): mean {mae_g:5.2f} pp, max {mx_g:5.2f} pp", flush=True)
    for k in a_keys: out[f"grid{g}_{k}"] = A[k]
    out[f"mae_grid{g}"] = mae_t; out[f"time_grid{g}"] = dt
    res_by_grid[g] = A
gs = sorted(res_by_grid)
if len(gs) == 1 and a.n_wl == 0:
    # single fine grid on the full wavelength set: the archived GPU solve is the grid-64 arm
    g = gs[0]
    d = 100 * np.mean([np.mean(np.abs(res_by_grid[g][k] - gpu[k])) for k in a_keys])
    dmax = 100 * max(np.max(np.abs(res_by_grid[g][k] - gpu[k])) for k in a_keys)
    verdict = lambda m: "PRETENDER" if m > 5.0 else "pass"
    print(f"  grid 64(archived) -> {g}: mean |dA| {d:5.2f} pp, max {dmax:5.2f} pp | MAE {out['mae_rcwa_archived']:.2f} -> {out[f'mae_grid{g}']:.2f} % "
          f"| verdict {verdict(out['mae_rcwa_archived'])} -> {verdict(out[f'mae_grid{g}'])}", flush=True)
    out["delta_mean_pp"] = d; out["delta_max_pp"] = dmax
if len(gs) >= 2:
    d = 100 * np.mean([np.mean(np.abs(res_by_grid[gs[-1]][k] - res_by_grid[gs[0]][k])) for k in a_keys])
    dmax = 100 * max(np.max(np.abs(res_by_grid[gs[-1]][k] - res_by_grid[gs[0]][k])) for k in a_keys)
    verdict = lambda m: "PRETENDER" if m > 5.0 else "pass"
    print(f"  grid {gs[0]} -> {gs[-1]}: mean |dA| {d:5.2f} pp, max {dmax:5.2f} pp | MAE {out[f'mae_grid{gs[0]}']:.2f} -> {out[f'mae_grid{gs[-1]}']:.2f} % "
          f"| verdict {verdict(out[f'mae_grid{gs[0]}'])} -> {verdict(out[f'mae_grid{gs[-1]}'])}", flush=True)
    out["delta_mean_pp"] = d; out["delta_max_pp"] = dmax
pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
np.savez(pathlib.Path(a.out) / f"grid_delta_{s}_s{a.seed}_{i}.npz", **out)
