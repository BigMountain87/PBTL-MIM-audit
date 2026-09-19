"""Structure-B physics features matching the training script
`pbtl_B_10seed.py:compute_physics_features_B` (which produced
`pretrained_mphys_tmm_B.pt`). 13 features. Both numpy and torch
versions returning the same feature order:

    0  cos(phase)              (cavity, SiO2 single-spacer)
    1  sin(phase)
    2  ring_area / P^2         (ring fill fraction)
    3  disk_area / P^2         (disk fill fraction)
    4  (ring_area + disk_area) / P^2   (total fill)
    5  P / lam
    6  R_out / lam
    7  t_Cr / skin_depth
    8  n_sio2 * d_SiO2 / lam
    9  cos(theta)
   10  gap_ratio   = (R_out - R_in) / (R_out + 1e-10)
   11  disk_ring   = R_disk / (R_in + 1e-10)
   12  alpha_metal = 4*pi*k_metal / lam
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


def compute_phys_B13_np(params, wavelengths_nm, metal="Cr"):
    """Return (N, Nlam, 13). params shape (N, 8)."""
    N, Nlam = len(params), len(wavelengths_nm)
    P, R_out, R_in, R_disk = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    t_Cr, d_SiO2 = params[:, 4], params[:, 5]
    theta, _phi = params[:, 6], params[:, 7]
    theta_rad = np.deg2rad(theta)
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
    ring_area = np.pi * (R_out ** 2 - R_in ** 2)
    disk_area = np.pi * R_disk ** 2
    feats.append(np.tile((ring_area / P ** 2)[:, None], (1, Nlam)))
    feats.append(np.tile((disk_area / P ** 2)[:, None], (1, Nlam)))
    feats.append(np.tile(((ring_area + disk_area) / P ** 2)[:, None], (1, Nlam)))
    feats.append(P[:, None] / wavelengths_nm[None, :])
    feats.append(R_out[:, None] / wavelengths_nm[None, :])
    feats.append(t_Cr[:, None] / skin_depth[None, :])
    feats.append(n_sio2[None, :] * d_SiO2[:, None] / wavelengths_nm[None, :])
    feats.append(np.tile(np.cos(theta_rad)[:, None], (1, Nlam)))
    gap_ratio = (R_out - R_in) / (R_out + 1e-10)
    feats.append(np.tile(gap_ratio[:, None], (1, Nlam)))
    disk_ring = R_disk / (R_in + 1e-10)
    feats.append(np.tile(disk_ring[:, None], (1, Nlam)))
    feats.append(np.tile(alpha[None, :], (N, 1)))
    return np.stack(feats, axis=-1).astype(np.float32)


def compute_phys_B13_torch(geo_t, lam_scalar, metal="Cr"):
    """Differentiable. geo_t shape (B, 8); lam_scalar float. Returns (B, 13)."""
    P, R_out, R_in, R_disk = geo_t[:, 0], geo_t[:, 1], geo_t[:, 2], geo_t[:, 3]
    t_Cr, d_SiO2 = geo_t[:, 4], geo_t[:, 5]
    theta = geo_t[:, 6]
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
    pi_t = torch.tensor(np.pi, dtype=geo_t.dtype, device=geo_t.device)
    ring_area = pi_t * (R_out ** 2 - R_in ** 2)
    disk_area = pi_t * R_disk ** 2
    feats.append(ring_area / P ** 2)
    feats.append(disk_area / P ** 2)
    feats.append((ring_area + disk_area) / P ** 2)
    feats.append(P / lam_scalar)
    feats.append(R_out / lam_scalar)
    feats.append(t_Cr / sd)
    feats.append(n_sio2 * d_SiO2 / lam_scalar)
    feats.append(torch.cos(theta_rad))
    feats.append((R_out - R_in) / (R_out + 1e-10))
    feats.append(R_disk / (R_in + 1e-10))
    feats.append(torch.full_like(P, alpha_m))
    return torch.stack(feats, dim=-1)
