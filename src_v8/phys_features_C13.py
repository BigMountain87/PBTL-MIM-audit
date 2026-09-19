"""Structure-C physics features as used by Paper 1's
`pretrained_mphys_tmm_C.pt` — the first 13 of `compute_physics_features_C_v2`
in `step0_screen/pbtl_C_v2_10seed.py`. Both numpy (training) and torch
(differentiable for inverse design) versions, returning shapes:

    numpy: (N, Nlam, 13)
    torch (looped per-wavelength): (Nlam, 13) for a single geo
"""

from __future__ import annotations

import numpy as np
import torch

# Upstream code root is put on sys.path by src_v8/common.py before this module
# is imported (vendored tree src_v8/upstream_920b1bd by default). Importing
# common here would be circular, so no path manipulation happens in this file.

from src.simulation.materials import (  # noqa: E402
    get_sio2_permittivity, get_metal_permittivity,
)


def compute_phys_C13_np(params, wavelengths_nm, metal="Cr"):
    """Return (N, Nlam, 13). params shape (N, 7)."""
    N = len(params); Nlam = len(wavelengths_nm)
    P, Wx, Wy, t_Cr, d_SiO2, theta, phi = [params[:, i] for i in range(7)]
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    eps_sio2 = get_sio2_permittivity(wavelengths_nm)
    eps_metal = get_metal_permittivity(wavelengths_nm, metal)
    n_sio2 = np.sqrt(np.real(eps_sio2))
    k_metal = np.imag(np.sqrt(eps_metal))
    skin_depth = wavelengths_nm / (4 * np.pi * k_metal + 1e-30)
    alpha = 4 * np.pi * k_metal / wavelengths_nm

    feats = []
    sin_ti = np.clip(np.sin(theta_rad[:, None]) / n_sio2[None, :], -1, 1)
    cos_ti = np.sqrt(1 - sin_ti ** 2)
    phase = 4 * np.pi * n_sio2[None, :] * d_SiO2[:, None] * cos_ti / wavelengths_nm[None, :]
    feats.append(np.cos(phase))
    feats.append(np.sin(phase))
    feats.append(np.tile((Wx * Wy / P ** 2)[:, None], (1, Nlam)))
    feats.append(P[:, None] / wavelengths_nm[None, :])
    feats.append(Wx[:, None] / wavelengths_nm[None, :])
    feats.append(Wy[:, None] / wavelengths_nm[None, :])
    feats.append(t_Cr[:, None] / skin_depth[None, :])
    feats.append(n_sio2[None, :] * d_SiO2[:, None] / wavelengths_nm[None, :])
    feats.append(np.tile(np.cos(theta_rad)[:, None], (1, Nlam)))
    feats.append(np.tile(np.cos(phi_rad)[:, None], (1, Nlam)))
    feats.append(np.tile(np.sin(phi_rad)[:, None], (1, Nlam)))
    feats.append(np.tile((Wy / (Wx + 1e-10))[:, None], (1, Nlam)))
    feats.append(np.tile(alpha[None, :], (N, 1)))
    return np.stack(feats, axis=-1).astype(np.float32)


def compute_phys_C13_torch(geo_t, lam_scalar, metal="Cr"):
    """Differentiable: geo_t shape (B, 7), lam_scalar float → (B, 13)."""
    P, Wx, Wy = geo_t[:, 0], geo_t[:, 1], geo_t[:, 2]
    t_Cr, d_SiO2 = geo_t[:, 3], geo_t[:, 4]
    theta, phi = geo_t[:, 5], geo_t[:, 6]
    theta_rad = theta * (np.pi / 180.0)
    phi_rad = phi * (np.pi / 180.0)
    lam_np = np.array([lam_scalar], dtype=np.float64)
    n_sio2 = float(np.sqrt(np.real(get_sio2_permittivity(lam_np))[0]))
    eps_m = complex(get_metal_permittivity(lam_np, metal)[0])
    k_m = float(np.imag(np.sqrt(eps_m)))
    sd = lam_scalar / (4 * np.pi * k_m + 1e-30)
    alpha_m = 4 * np.pi * k_m / lam_scalar

    feats = []
    sin_ti = torch.clamp(torch.sin(theta_rad) / n_sio2, -1, 1)
    cos_ti = torch.sqrt(1 - sin_ti ** 2)
    phase = 4 * np.pi * n_sio2 * d_SiO2 * cos_ti / lam_scalar
    feats.append(torch.cos(phase))
    feats.append(torch.sin(phase))
    feats.append(Wx * Wy / P ** 2)
    feats.append(P / lam_scalar)
    feats.append(Wx / lam_scalar)
    feats.append(Wy / lam_scalar)
    feats.append(t_Cr / sd)
    feats.append(n_sio2 * d_SiO2 / lam_scalar)
    feats.append(torch.cos(theta_rad))
    feats.append(torch.cos(phi_rad))
    feats.append(torch.sin(phi_rad))
    feats.append(Wy / (Wx + 1e-10))
    feats.append(torch.full_like(P, alpha_m))
    return torch.stack(feats, dim=-1)
