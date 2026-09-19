"""Structure-C physics features of the PUBLISHED Paper-1 pipeline — the 18-feature
`compute_physics_features_C_v2` defined inline in upstream
`step0_screen/pbtl_C_v2_redesign.py` (13 original + 5 polarization-specific).
The first 13 columns are exactly `phys_features_C13` (asserted in selfcheck).

    numpy: (N, Nlam, 18)          torch (per wavelength): (B, 18)

Upstream code root (materials, jc tables) is put on sys.path by common.py before
this module is imported; importing common here would be circular.
"""
from __future__ import annotations
import numpy as np
import torch

from src.simulation.materials import (  # noqa: E402
    get_sio2_permittivity, get_metal_permittivity,
)
from phys_features_C13 import compute_phys_C13_np, compute_phys_C13_torch  # noqa: E402


def compute_phys_C18_np(params, wavelengths_nm, metal="Cr"):
    """Return (N, Nlam, 18). params shape (N, 7) = [P, Wx, Wy, t_Cr, d_SiO2, theta, phi]."""
    base = compute_phys_C13_np(params, wavelengths_nm, metal)          # (N, L, 13)
    N, Nlam = len(params), len(wavelengths_nm)
    P, Wx, Wy = params[:, 0], params[:, 1], params[:, 2]
    f_x = Wx / P
    f_y = Wy / P
    aniso = np.abs(f_x - f_y) / (f_x + f_y + 1e-10)
    lam = wavelengths_nm[None, :]
    extra = [
        np.tile(f_x[:, None], (1, Nlam)),                 # 13: f_x = Wx/P
        np.tile(f_y[:, None], (1, Nlam)),                 # 14: f_y = Wy/P
        np.tile(aniso[:, None], (1, Nlam)),               # 15: normalized anisotropy
        Wx[:, None] ** 2 / (P[:, None] * lam),            # 16: Wx^2/(P*lam)
        Wy[:, None] ** 2 / (P[:, None] * lam),            # 17: Wy^2/(P*lam)
    ]
    return np.concatenate([base, np.stack(extra, axis=-1)], axis=-1).astype(np.float32)


def compute_phys_C18_torch(geo_t, lam_scalar, metal="Cr"):
    """Differentiable: geo_t (B, 7), lam_scalar float -> (B, 18)."""
    base = compute_phys_C13_torch(geo_t, lam_scalar, metal)            # (B, 13)
    P, Wx, Wy = geo_t[:, 0], geo_t[:, 1], geo_t[:, 2]
    f_x = Wx / P
    f_y = Wy / P
    aniso = torch.abs(f_x - f_y) / (f_x + f_y + 1e-10)
    extra = torch.stack([f_x, f_y, aniso,
                         Wx ** 2 / (P * lam_scalar),
                         Wy ** 2 / (P * lam_scalar)], dim=-1)
    return torch.cat([base, extra], dim=-1)
