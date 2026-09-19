#!/usr/bin/env python3
"""Assert src_v8/phys_features_D11 reproduces the upstream Structure-D features.

    python3 scripts/check_phys_D11.py --upstream <PBTL-MIM working copy>

Checks (a) our numpy version against upstream compute_physics_features_D on random
geometries, and (b) our torch version against our numpy version wavelength by
wavelength, which is what the inverse loop actually calls.  The surrogate is trained on
the upstream features, so a silent mismatch here would poison every D result.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np


def main(up):
    up = Path(up).expanduser()
    sys.path.insert(0, str(up))
    sys.path.insert(0, str(up / "structure_D"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_v8"))
    import src.simulation.materials as _mat
    _mat.MATERIAL_MODEL = "jc"
    from structure_D.physics_features_D import compute_physics_features_D
    import phys_features_D11 as ours
    import torch

    rng = np.random.default_rng(0)
    lo = np.array([200.0, 50.0, 20.0, 10.0, 50.0, 0.0])
    hi = np.array([900.0, 400.0, 200.0, 80.0, 400.0, 30.0])
    params = lo + rng.random((32, 6)) * (hi - lo)
    wl = np.linspace(400.0, 1800.0, 100)

    up_f = compute_physics_features_D(params, wl, metal="Cr")
    our_f = ours.compute_phys_D11_np(params, wl, metal="Cr")
    assert up_f.shape == our_f.shape == (32, 100, 11), (up_f.shape, our_f.shape)
    d_np = float(np.abs(up_f - our_f).max())
    print(f"numpy vs upstream: max |diff| = {d_np:.3e}  shape {our_f.shape}")

    geo = torch.tensor(params, dtype=torch.float32)
    d_t = 0.0
    for j in (0, 37, 99):
        t = ours.compute_phys_D11_torch(geo, float(wl[j]), metal="Cr").numpy()
        d_t = max(d_t, float(np.abs(t - our_f[:, j, :]).max()))
    print(f"torch vs numpy (3 wavelengths): max |diff| = {d_t:.3e}")

    g = torch.tensor(params, dtype=torch.float32, requires_grad=True)
    ours.compute_phys_D11_torch(g, 900.0).sum().backward()
    finite = bool(torch.isfinite(g.grad).all())
    print(f"gradient finite through every feature: {finite}")

    ok = d_np < 1e-5 and d_t < 1e-4 and finite
    print("phys_D11:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="<PBTL-MIM working copy>")
    raise SystemExit(main(ap.parse_args().upstream))
