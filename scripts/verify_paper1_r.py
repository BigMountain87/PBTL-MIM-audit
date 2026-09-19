#!/usr/bin/env python3
"""Verify Paper 1's TMM-RCWA Pearson r for Structures A/B/C from raw data.
Runs on CPU to avoid interfering with GPU C-inverse. Uses the same TMM engines
and same wavelength axes that the PBTL training scripts used."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

# Portable repo root: this script lives at scripts/verify_paper1_r.py
ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
if not UP.exists():
    # Fallback for environments where upstream is a sibling-symlink target
    UP = Path.home() / "InverseTL" / "upstream"
sys.path.insert(0, str(UP))

from src.simulation.tmm_struct_a import compute_tmm_batch as tmm_A
from src.simulation.tmm_struct_b import compute_tmm_batch as tmm_B
from src.simulation.tmm_struct_c_aniso import compute_tmm_batch as tmm_C  # aniso for dual-pol


def pearson(x, y):
    x = x.ravel().astype(np.float64); y = y.ravel().astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m] - x[m].mean(); y = y[m] - y[m].mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()) + 1e-30))


def per_sample_r(tmm_arr, rcwa_arr):
    rs = [pearson(tmm_arr[i], rcwa_arr[i]) for i in range(len(tmm_arr))]
    return np.array(rs)


def run(label, npz_path, tmm_fn, metal="Cr", n_use=300):
    print(f"\n=== {label}: {npz_path.name} ===", flush=True)
    d = np.load(npz_path)
    params = d["params"].astype(np.float64)
    wl = d["wavelengths"].astype(np.float64)
    n_use = min(n_use, len(params))
    params = params[:n_use]
    print(f"  n={n_use}, wl=({wl.min():.0f}-{wl.max():.0f}nm, {len(wl)} pts)", flush=True)

    # RCWA spectra (already in npz)
    if "A" in d.files:
        rcwa = d["A"][:n_use].astype(np.float64)
        # TMM
        tmm = tmm_fn(params, wl, metal)
        if isinstance(tmm, dict):
            tmm_a = np.asarray(tmm.get("A_tmm",
                          tmm.get("A", tmm.get("absorption"))))
        elif isinstance(tmm, tuple):
            tmm_a = np.asarray(tmm[0])
        else:
            tmm_a = np.asarray(tmm)
        # Pooled Pearson across all samples × wavelengths
        r_pooled = pearson(tmm_a, rcwa)
        # Per-sample
        r_per = per_sample_r(tmm_a, rcwa)
        print(f"  POOLED r(TMM, RCWA) = {r_pooled:+.4f}", flush=True)
        print(f"  PER-SAMPLE r: mean={r_per.mean():+.4f}  median={np.median(r_per):+.4f}  "
              f"std={r_per.std():.4f}  min={r_per.min():+.4f}  max={r_per.max():+.4f}", flush=True)
    elif "A_TE" in d.files:
        rcwa_te = d["A_TE"][:n_use].astype(np.float64)
        rcwa_tm = d["A_TM"][:n_use].astype(np.float64)
        tmm = tmm_fn(params, wl, metal)
        if isinstance(tmm, dict):
            tmm_te = tmm.get("A_tmm_te", tmm.get("A_TE", tmm.get("A_te")))
            tmm_tm = tmm.get("A_tmm_tm", tmm.get("A_TM", tmm.get("A_tm")))
            tmm_te = np.asarray(tmm_te); tmm_tm = np.asarray(tmm_tm)
            if tmm_te is None or tmm_tm is None:
                raise RuntimeError(f"unknown TMM C keys: {list(tmm.keys())}")
        else:
            tmm_te = np.asarray(tmm[0]); tmm_tm = np.asarray(tmm[1])
        r_te_pooled = pearson(tmm_te, rcwa_te)
        r_tm_pooled = pearson(tmm_tm, rcwa_tm)
        # combined: stack TE+TM
        r_combined = pearson(np.concatenate([tmm_te, tmm_tm]),
                             np.concatenate([rcwa_te, rcwa_tm]))
        r_per_te = per_sample_r(tmm_te, rcwa_te)
        r_per_tm = per_sample_r(tmm_tm, rcwa_tm)
        # Per-sample combined r: concatenate TE+TM spectra per sample
        r_per_combined = np.array([
            pearson(np.concatenate([tmm_te[i], tmm_tm[i]]),
                    np.concatenate([rcwa_te[i], rcwa_tm[i]]))
            for i in range(len(tmm_te))
        ])
        print(f"  POOLED r(TMM-TE, RCWA-TE) = {r_te_pooled:+.4f}", flush=True)
        print(f"  POOLED r(TMM-TM, RCWA-TM) = {r_tm_pooled:+.4f}", flush=True)
        print(f"  POOLED r(combined)        = {r_combined:+.4f}", flush=True)
        print(f"  PER-SAMPLE TE: mean={r_per_te.mean():+.4f}  median={np.median(r_per_te):+.4f}", flush=True)
        print(f"  PER-SAMPLE TM: mean={r_per_tm.mean():+.4f}  median={np.median(r_per_tm):+.4f}", flush=True)
        print(f"  PER-SAMPLE combined (TE+TM concat): "
              f"mean={r_per_combined.mean():+.4f}  median={np.median(r_per_combined):+.4f}",
              flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU to not disturb GPU
    raw = UP / "data" / "raw"
    # n_use=500 matches Paper 1's Table 4 sample pool (struct_*_500.npz; A uses vis_500)
    run("STRUCTURE A (vis, 380-780nm)", raw / "struct_A_vis_500.npz", tmm_A, metal="Cr", n_use=500)
    run("STRUCTURE B (400-1800nm)",     raw / "struct_B_500.npz",      tmm_B, metal="Cr", n_use=500)
    run("STRUCTURE C (aniso, 400-1800nm)", raw / "struct_C_500.npz",   tmm_C, metal="Cr", n_use=500)
