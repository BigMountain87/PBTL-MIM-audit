#!/usr/bin/env python3
"""Generate 500 samples of Structure A at unified 400-1800nm axis.

This produces struct_A_500_broadband.npz which, together with existing
struct_B_500.npz and struct_C_500.npz, supports a wavelength-unified
cross-structure comparison.

Protocol parallels step0_screen/generate_struct_a.py but at 400-1800nm
and 500 samples instead of 100 samples at 380-780nm.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch

UPSTREAM = Path.home() / "InverseTL" / "upstream"
sys.path.insert(0, str(UPSTREAM))

from src.simulation.rcwa_struct_a import generate_dataset
from src.simulation.tmm_struct_a import compute_tmm_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

N_SAMPLES = 500
SEED = 42
WAVELENGTHS = np.linspace(400, 1800, 100)
OUT = UPSTREAM / "data" / "raw" / "struct_A_500_broadband.npz"
OUT.parent.mkdir(parents=True, exist_ok=True)

print("=" * 70, flush=True)
print(f"Structure A broadband: {N_SAMPLES} samples at {WAVELENGTHS[0]:.0f}-{WAVELENGTHS[-1]:.0f} nm", flush=True)
print(f"Output: {OUT}", flush=True)
print("=" * 70, flush=True)

t0 = time.time()
data = generate_dataset(N_SAMPLES, WAVELENGTHS, metal="Cr", seed=SEED, device=device)
dt = time.time() - t0
print(f"\nRCWA time: {dt:.1f}s ({dt / N_SAMPLES:.2f}s/sample)", flush=True)

A_arr = data["A"]; R_arr = data["R"]; T_arr = data["T"]
params = data["params"]
energy_err = np.abs(A_arr + R_arr + T_arr - 1)
print(f"\n--- Validation ---", flush=True)
print(f"Energy conservation max error: {energy_err.max():.2e}", flush=True)
print(f"A range: [{A_arr.min():.4f}, {A_arr.max():.4f}]", flush=True)
bad_R = np.any(R_arr > 1.01, axis=1)
bad_T = np.any(T_arr > 1.01, axis=1)
n_bad = int(np.sum(bad_R | bad_T))
print(f"Samples with R>1.01 or T>1.01: {n_bad}/{N_SAMPLES}", flush=True)

# Save RCWA-only first
np.savez(OUT, **data)
print(f"Saved RCWA-only: {OUT}", flush=True)

# TMM baseline (CPU OK, fast)
print("\n--- TMM baseline ---", flush=True)
t1 = time.time()
tmm_out = compute_tmm_batch(params, WAVELENGTHS, metal="Cr")
dt1 = time.time() - t1
A_tmm = tmm_out["A_tmm"]; R_tmm = tmm_out["R_tmm"]; T_tmm = tmm_out["T_tmm"]
mae_A = float(np.mean(np.abs(A_tmm - A_arr)) * 100)
print(f"TMM time: {dt1:.1f}s", flush=True)
print(f"TMM-RCWA absorption MAE: {mae_A:.2f}%", flush=True)

# Pearson r per sample
rs = []
for i in range(N_SAMPLES):
    if np.std(A_tmm[i]) > 1e-6 and np.std(A_arr[i]) > 1e-6:
        rs.append(float(np.corrcoef(A_tmm[i], A_arr[i])[0, 1]))
rs = np.array(rs)
print(f"TMM-RCWA Pearson r: mean={rs.mean():+.4f}, median={np.median(rs):+.4f}, "
      f"std={rs.std():.4f}, n_valid={len(rs)}", flush=True)
print(f"  range: [{rs.min():+.4f}, {rs.max():+.4f}]", flush=True)

# Save with TMM
out = dict(data)
out["A_tmm"] = A_tmm.astype(np.float32)
out["R_tmm"] = R_tmm.astype(np.float32)
out["T_tmm"] = T_tmm.astype(np.float32)
np.savez(OUT, **out)
print(f"\nFinal saved (with TMM): {OUT}", flush=True)
print("DONE", flush=True)
