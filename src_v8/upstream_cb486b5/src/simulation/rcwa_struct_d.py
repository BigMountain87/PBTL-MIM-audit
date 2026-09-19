"""
Structure D: Cross (plus) shaped MIM absorber (TORCWA GPU) -- CONTINGENCY 4th structure.

Layer stack:
  Air | Patterned Cr (cross L,w, thickness t_Cr) | SiO2 (d) | Cr mirror (100nm) | Glass

6 Parameters: P, L (arm length), w (arm width), t_Cr, d_SiO2, theta
Output: single-polarization (A, R, T) -- the cross is 4-fold symmetric, hence
polarization-independent at normal incidence; TE is reported at oblique incidence.

Mirrors src/simulation/rcwa_struct_c.py: SAME rcwa_utils (grazing-fixed COS_MIN
power normalization), SAME materials, SAME complex64 + adaptive-order pipeline.
The ONLY structural change is the patterned geometry (union of two rectangles).
Run on a CUDA GPU (home server); the corrected-pipeline conventions are inherited.
"""

import numpy as np
import torch
import torcwa
from tqdm import tqdm
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.simulation.materials import get_metal_permittivity, get_sio2_permittivity
from src.simulation.rcwa_utils import compute_RT_batch

DESIGN_SPACE = {
    "P":      {"min": 300, "max": 800, "unit": "nm"},
    "L":      {"min": 100, "max": 720, "unit": "nm"},  # arm length, <= 0.9 P
    "w":      {"min": 40,  "max": 300, "unit": "nm"},  # arm width, <= L
    "t_Cr":   {"min": 20,  "max": 80,  "unit": "nm"},
    "d_SiO2": {"min": 50,  "max": 200, "unit": "nm"},
    "theta":  {"min": 0,   "max": 60,  "unit": "deg"},
}
PARAM_NAMES = list(DESIGN_SPACE.keys())
PARAM_MIN = np.array([DESIGN_SPACE[k]["min"] for k in PARAM_NAMES], dtype=np.float64)
PARAM_MAX = np.array([DESIGN_SPACE[k]["max"] for k in PARAM_NAMES], dtype=np.float64)

WAVELENGTH = {"start": 400, "stop": 1800, "n_pts": 100}

# Inherit the corrected-pipeline settings from Structure C (adaptive order + c64).
RCWA_SETTINGS = {"grid": (64, 64), "order": [5, 5], "adaptive_order": True,
                 "dtype": torch.complex64}

_AZI_PERTURB = np.deg2rad(0.01)


def adaptive_order(lam_nm, P, L, w):
    """Feature-size-adaptive Fourier order for the cross (same policy as Structure C).

    Convergence is governed by the narrowest metal/air feature: the arm width w and
    the air gap (P - L)/2. A cross has MORE re-entrant corners than a rectangle, so
    it is expected to sit at the harder end -- err on the side of higher order.
      hard  (w < 150 nm  OR  air gap (P-L)/2 < 150 nm)  -> N = 17
      benign, short lambda (< 700 nm)                    -> N = 13
      benign, long  lambda (>= 700 nm)                   -> N = 9
    Residual note (as for C): N=17 is the 16 GB-GPU cap; narrowest-arm crosses may
    not be fully converged (true value needs N>=21). Documented, not eliminated --
    the convergence study below MUST be run before trusting absolute fidelity.
    """
    gap = 0.5 * (P - L)
    hard = (w < 150.0) or (gap < 150.0)
    if hard:
        N = 17
    elif lam_nm < 700.0:
        N = 13
    else:
        N = 9
    return [N, N]


def _cross_geometry(P, L, w):
    """Union of a horizontal (L x w) and vertical (w x L) bar, centered."""
    torcwa.rcwa_geo.Lx = P
    torcwa.rcwa_geo.Ly = P
    torcwa.rcwa_geo.nx = RCWA_SETTINGS["grid"][0]
    torcwa.rcwa_geo.ny = RCWA_SETTINGS["grid"][1]
    torcwa.rcwa_geo.grid()
    horiz = torcwa.rcwa_geo.rectangle(Wx=L, Wy=w, Cx=P / 2, Cy=P / 2)
    vert = torcwa.rcwa_geo.rectangle(Wx=w, Wy=L, Cx=P / 2, Cy=P / 2)
    return torcwa.rcwa_geo.union(horiz, vert)


def _build_sim(freq, geo, t_Cr, d_SiO2, eps_m, eps_sio2,
               theta_rad, phi_rad, order, P, sim_dtype, device):
    eps_pat = (geo * eps_m + (1.0 - geo) * 1.0).to(dtype=sim_dtype, device=device)
    sim = torcwa.rcwa(freq=freq, order=order, L=[P, P],
                      dtype=sim_dtype, device=device, stable_eig_grad=False)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=theta_rad, azi_ang=phi_rad)
    sim.source_planewave(amplitude=[0.0, 1.0], direction='forward')
    sim.add_layer(thickness=t_Cr, eps=eps_pat)
    sim.add_layer(thickness=d_SiO2, eps=eps_sio2)
    sim.add_layer(thickness=100.0, eps=eps_m)
    sim.solve_global_smatrix()
    return sim


def simulate_single(params, wavelengths_nm, metal="Cr", device=None):
    """Single parameter set -> (A, R, T) arrays over wavelengths."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P = float(params["P"]); L = float(params["L"]); w = float(params["w"])
    t_Cr = float(params["t_Cr"]); d_SiO2 = float(params["d_SiO2"])
    theta_deg = float(params["theta"])
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = _AZI_PERTURB if theta_deg > 0.1 else 0.0

    use_adaptive = RCWA_SETTINGS.get("adaptive_order", True)
    fixed_order = RCWA_SETTINGS["order"]
    sim_dtype = RCWA_SETTINGS.get("dtype", torch.complex64)

    eps_metal_all = get_metal_permittivity(wavelengths_nm, metal)
    eps_sio2_all = get_sio2_permittivity(wavelengths_nm)

    geo = _cross_geometry(P, L, w)  # geometry is wavelength-independent
    n_wl = len(wavelengths_nm)
    A = np.zeros(n_wl); R = np.zeros(n_wl); T = np.zeros(n_wl)
    for i, lam in enumerate(wavelengths_nm):
        freq = 1.0 / lam
        eps_m = complex(eps_metal_all[i])
        eps_sio2 = float(np.real(eps_sio2_all[i]))
        order = adaptive_order(lam, P, L, w) if use_adaptive else fixed_order
        sim = _build_sim(freq, geo, t_Cr, d_SiO2, eps_m, eps_sio2,
                         theta_rad, phi_rad, order, P, sim_dtype, device)
        r, t = compute_RT_batch(sim, order)
        A[i] = 1.0 - r - t; R[i] = r; T[i] = t
        del sim
    return A, R, T


def generate_dataset(n_samples, wavelengths_nm, metal="Cr", seed=42, device=None):
    """Latin-Hypercube dataset, same conventions as Structure C's generator."""
    from scipy.stats import qmc
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=rng)
    u = sampler.random(n=n_samples)

    all_params = []
    for i in range(n_samples):
        p = {name: PARAM_MIN[j] + u[i, j] * (PARAM_MAX[j] - PARAM_MIN[j])
             for j, name in enumerate(PARAM_NAMES)}
        p["L"] = min(p["L"], 0.9 * p["P"])   # arm fits in the cell
        p["w"] = min(p["w"], p["L"])         # width <= length
        all_params.append(p)

    n_wl = len(wavelengths_nm)
    A_all = np.zeros((n_samples, n_wl)); R_all = np.zeros((n_samples, n_wl))
    T_all = np.zeros((n_samples, n_wl)); params_array = np.zeros((n_samples, len(PARAM_NAMES)))
    for i in tqdm(range(n_samples), desc=f"RCWA-D ({metal})"):
        p = all_params[i]
        a, r, t = simulate_single(p, wavelengths_nm, metal=metal, device=device)
        A_all[i] = a; R_all[i] = r; T_all[i] = t
        for j, name in enumerate(PARAM_NAMES):
            params_array[i, j] = p[name]

    return {
        "params": params_array, "param_names": PARAM_NAMES,
        "A": A_all, "R": R_all, "T": T_all,
        "wavelengths": wavelengths_nm, "metal": metal,
        "structure": "D_cross", "redesign_version": "structD-v1-jc+adaptive+c64+grazingfix",
    }
