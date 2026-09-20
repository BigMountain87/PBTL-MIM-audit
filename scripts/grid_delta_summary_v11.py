"""Summarise every raster re-solve of the audit (T78 + T82 + T98) into one evidence file.

Inputs: results_pub/grid_delta/*.npz (T78: 30 committed designs, mixed wavelength coverage;
T82: 23 near-threshold committed designs at all 100 wavelengths), results_pub/grid_delta_full/*.npz
(T98: the 11 T78 designs that had been subsampled, re-solved at all 100 wavelengths; used in
place of their subsampled record when present), results_pub/grid_delta_paper1B/paper1_b_grid_delta_v11.json
(30 stratified samples of the companion's Structure-B dataset, grid 64 control + 256), and the
archived verdicts.  Writes results_pub/grid_delta_summary_v11.json for the checker and W8.

    python3 scripts/grid_delta_summary_v11.py
"""
import glob, json, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results_pub"; TAU = 5.0
PILOT = dict(structure="A", seed=123, index=3, n_wl=10, mae_archived=4.44, mae_grid256=3.89, delta_mean_pp=0.60, delta_max_pp=0.85, source="T78 pilot (log only)")

def load(f):
    z = np.load(f, allow_pickle=True); n = int(z["wavelengths"].shape[0])
    d = dict(structure=str(z["structure"]), seed=int(z["seed"]), index=int(z["index"]), n_wl=n,
             mae_archived=float(z["mae_rcwa_archived"]), delta_mean_pp=float(z["delta_mean_pp"]), delta_max_pp=float(z["delta_max_pp"]))
    if "mae_grid256" in z.files: d["mae_grid256"] = float(z["mae_grid256"])
    if "mae_grid64" in z.files: d["mae_grid64_subsample"] = float(z["mae_grid64"])
    return d

designs = {}
for f in sorted(glob.glob(str(R / "grid_delta" / "grid_delta_*.npz"))):
    d = load(f); d["source"] = "T82" if d["n_wl"] == 100 and os.path.getmtime(f) > 1_758_200_000 else "T78"; designs[(d["structure"], d["seed"], d["index"])] = d
for f in sorted(glob.glob(str(R / "grid_delta_full" / "grid_delta_*.npz"))):
    d = load(f); d["source"] = "T98 (full-spectrum re-solve of a T78 subsampled design)"; designs[(d["structure"], d["seed"], d["index"])] = d
designs.setdefault(("A", 123, 3), PILOT)
rows = sorted(designs.values(), key=lambda d: (d["structure"], d["seed"], d["index"]))
def verdict(m): return "pretender" if m > TAU else "pass"
for d in rows:
    base = d["mae_archived"] if d["n_wl"] == 100 else d.get("mae_grid64_subsample", d["mae_archived"])
    d["verdict_grid64"] = verdict(base); d["verdict_grid256"] = verdict(d["mae_grid256"]); d["flip"] = d["verdict_grid64"] != d["verdict_grid256"]
    d["distance_to_tau_pp"] = abs(d["mae_archived"] - TAU)
full = [d for d in rows if d["n_wl"] == 100]; sub = [d for d in rows if d["n_wl"] < 100]
near1 = [d for d in rows if d["distance_to_tau_pp"] <= 1.0]; near2 = [d for d in rows if d["distance_to_tau_pp"] <= 2.0]
p1 = json.load(open(R / "grid_delta_paper1B" / "paper1_b_grid_delta_v11.json"))
out = dict(tau_pct=TAU, grid_fine=256, grid_pinned=64,
           committed=dict(n_designs=len(rows), n_full_spectrum=len(full), n_subsampled=len(sub),
                          n_flips=sum(d["flip"] for d in rows),
                          mean_abs_dA_pp=dict(min=min(d["delta_mean_pp"] for d in rows), max=max(d["delta_mean_pp"] for d in rows)),
                          worst_single_wavelength_pp=max(d["delta_max_pp"] for d in rows),
                          within_1pp_of_tau=dict(n=len(near1), n_full_spectrum=sum(d["n_wl"] == 100 for d in near1), flips=sum(d["flip"] for d in near1)),
                          within_2pp_of_tau=dict(n=len(near2), n_full_spectrum=sum(d["n_wl"] == 100 for d in near2), flips=sum(d["flip"] for d in near2)),
                          per_structure={s: dict(n=sum(d["structure"] == s for d in rows), flips=sum(d["flip"] for d in rows if d["structure"] == s)) for s in "ABCD"},
                          designs=rows),
           companion_B_dataset=dict(n=p1["n"], control_grid64_vs_archived_mean_pp=p1["control_grid64_vs_archived_mean_pp"],
                                    mean_abs_dA_pp=p1["delta_mean_pp"], max_abs_dA_pp=p1["delta_max_pp"], per_stratum=p1["per_stratum"]))
json.dump(out, open(R / "grid_delta_summary_v11.json", "w"), indent=2)
c = out["committed"]
print(f"committed designs re-solved: {c['n_designs']} ({c['n_full_spectrum']} at all 100 wavelengths, {c['n_subsampled']} subsampled); flips {c['n_flips']}; "
      f"mean|dA| {c['mean_abs_dA_pp']['min']:.2f}-{c['mean_abs_dA_pp']['max']:.2f} pp; worst wavelength {c['worst_single_wavelength_pp']:.2f} pp")
print(f"within 1 pp of tau: {c['within_1pp_of_tau']}  within 2 pp: {c['within_2pp_of_tau']}")
print("per structure:", c["per_structure"])
b = out["companion_B_dataset"]; print(f"companion B dataset: n={b['n']}, control median {b['control_grid64_vs_archived_mean_pp']['median']:.3f} pp, mean|dA| {b['mean_abs_dA_pp']['min']:.2f}-{b['mean_abs_dA_pp']['max']:.2f} pp, worst {b['max_abs_dA_pp']['max']:.2f} pp")
