"""Structure-D (cross patch) physics features, matching the upstream training script
`structure_D/physics_features_D.py:compute_physics_features_D`, which produces the
checkpoint the D surrogate fine-tunes from.  11 features, numpy and torch versions in
the same order:

    0  cos(phase)              (Fabry-Perot round trip in the SiO2 cavity)
    1  sin(phase)
    2  fill fraction f = (2*L*w - w^2) / P^2      (cross patch)
    3  P / lam
    4  L / lam
    5  w / lam
    6  t_Cr / skin_depth
    7  n_sio2 * d_SiO2 / lam
    8  cos(theta)
    9  arm aspect w / L
   10  alpha_metal = 4*pi*k_metal / lam

The upstream module builds 0-1, 6 and 10 from helpers in
`src/utils/physics_features.py` (`_cavity_phase_features`, `_skin_depth`,
`_metal_alpha`); those are inlined here exactly as `phys_features_B13` inlines them,
and `scripts/check_phys_D11.py` asserts this file reproduces the upstream array.
"""
from __future__ import annotations
import numpy as np
import torch

# The upstream code root is on sys.path via src_v8/common.py before this is imported.
from src.simulation.materials import (  # noqa: E402
    get_sio2_permittivity, get_metal_permittivity,
)

N_FEATURES = 11


def _material_constants(wavelengths_nm, metal):
    eps_sio2 = get_sio2_permittivity(wavelengths_nm)
    eps_metal = get_metal_permittivity(wavelengths_nm, metal)
    n_sio2 = np.sqrt(np.real(eps_sio2))
    k_metal = np.imag(np.sqrt(eps_metal))
    skin_depth = wavelengths_nm / (4 * np.pi * k_metal + 1e-30)
    alpha = 4 * np.pi * k_metal / wavelengths_nm
    return n_sio2, skin_depth, alpha


def compute_phys_D11_np(params, wavelengths_nm, metal="Cr"):
    """Return (N, Nlam, 11).  params shape (N, 6) = [P, L, w, t_Cr, d_SiO2, theta]."""
    N, Nlam = len(params), len(wavelengths_nm)
    P, L, w = params[:, 0], params[:, 1], params[:, 2]
    t_Cr, d_SiO2, theta = params[:, 3], params[:, 4], params[:, 5]
    theta_rad = np.deg2rad(theta)
    n_sio2, skin_depth, alpha = _material_constants(wavelengths_nm, metal)

    feats = []
    sin_ti = np.clip(np.sin(theta_rad[:, None]) / n_sio2[None, :], -1, 1)
    cos_ti = np.sqrt(1 - sin_ti ** 2)
    phase = 4 * np.pi * n_sio2[None, :] * d_SiO2[:, None] * cos_ti / wavelengths_nm[None, :]
    feats.append(np.cos(phase))
    feats.append(np.sin(phase))
    f = (2.0 * L * w - w * w) / P ** 2
    feats.append(np.tile(f[:, None], (1, Nlam)))
    feats.append(P[:, None] / wavelengths_nm[None, :])
    feats.append(L[:, None] / wavelengths_nm[None, :])
    feats.append(w[:, None] / wavelengths_nm[None, :])
    feats.append(t_Cr[:, None] / skin_depth[None, :])
    feats.append(n_sio2[None, :] * d_SiO2[:, None] / wavelengths_nm[None, :])
    feats.append(np.tile(np.cos(theta_rad)[:, None], (1, Nlam)))
    feats.append(np.tile((w / (L + 1e-10))[:, None], (1, Nlam)))
    feats.append(np.tile(alpha[None, :], (N, 1)))
    return np.stack(feats, axis=-1).astype(np.float32)


def compute_phys_D11_torch(geo_t, lam_scalar, metal="Cr"):
    """Differentiable.  geo_t shape (B, 6); lam_scalar float.  Returns (B, 11)."""
    P, L, w = geo_t[:, 0], geo_t[:, 1], geo_t[:, 2]
    t_Cr, d_SiO2, theta = geo_t[:, 3], geo_t[:, 4], geo_t[:, 5]
    theta_rad = theta * (np.pi / 180.0)
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
    feats.append((2.0 * L * w - w * w) / P ** 2)
    feats.append(P / lam_scalar)
    feats.append(L / lam_scalar)
    feats.append(w / lam_scalar)
    feats.append(t_Cr / sd)
    feats.append(n_sio2 * d_SiO2 / lam_scalar)
    feats.append(torch.cos(theta_rad))
    feats.append(w / (L + 1e-10))
    feats.append(torch.full_like(P, alpha_m))
    return torch.stack(feats, dim=-1)
