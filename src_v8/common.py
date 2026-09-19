#!/usr/bin/env python3
"""v8 unified pipeline — shared structure registry, data hygiene, splits,
TMM-faithful physics-feature statistics, and model classes.

Single source of truth for everything that differs between Structures A/B/C.
Protocol: docs/v8_protocol.md (frozen 2026-06-10).

Faithfulness contract with the Paper-1 pretrained checkpoints:
  * geometry input is [wl_norm, params_norm]  (wavelength FIRST)
  * params_norm uses TRAIN_BOUNDS (the bounds the pretrain scripts used),
    which for C is Wx,Wy in [50,400] even though the dataset is sampled wider
  * physics features are normalized with TMM-set statistics, recomputed
    deterministically from rng(99) — features depend only on params + grid
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

V8_ROOT = Path(os.environ.get("INVERSETL_ROOT", Path(__file__).resolve().parent.parent))
# Two distinct upstream roots (T02):
#   UPSTREAM       — Paper-1 *inputs* (data/raw/*.npz, results/pretrained_*.pt).
#                    These are gitignored upstream and are not part of any commit,
#                    so they cannot be vendored from git; they are hashed in
#                    results_v8/INPUTS_MANIFEST.txt instead.
#   UPSTREAM_CODE  — Paper-1 *code* (src/), pinned to the vendored archive of
#                    commit 920b1bd so that solver settings, material models and
#                    physics features cannot drift under the audit.
# PROFILE selects which Paper-1 release is audited (docs/PLAN_v10_amendment_pub.md):
#   legacy — as-submitted release: legacy pools, 2026-03-15 checkpoints, code 920b1bd,
#            fixed Fourier order 5.  Artifacts: results_v8/ (frozen, never rewritten).
#   pub    — printed pipeline: *_redesign pools, regenerated checkpoints, code cb486b5
#            (J&C materials, adaptive Fourier order).  Artifacts: results_pub/.
PROFILE = os.environ.get("INVERSETL_PROFILE", "legacy")
if PROFILE not in ("legacy", "pub"):
    raise SystemExit(f"INVERSETL_PROFILE must be 'legacy' or 'pub', got {PROFILE!r}")
_PINNED = {"legacy": "upstream_920b1bd", "pub": "upstream_cb486b5"}
_UPSTREAM_DEFAULT = V8_ROOT / ("upstream_inputs_v8" if PROFILE == "legacy" else "upstream_inputs_pub")
if not _UPSTREAM_DEFAULT.exists():
    _UPSTREAM_DEFAULT = Path.home() / "mim_novel"      # the audit host kept the companion checkout there
UPSTREAM = Path(os.environ.get("INVERSETL_UPSTREAM_ROOT", _UPSTREAM_DEFAULT))
UPSTREAM_CODE = Path(os.environ.get(
    "INVERSETL_UPSTREAM_CODE", Path(__file__).resolve().parent / _PINNED[PROFILE]))
UPSTREAM_COMMIT = {"upstream_920b1bd": "920b1bd", "upstream_cb486b5": "cb486b5"}.get(
    UPSTREAM_CODE.name, "UNPINNED")
RESULTS = Path(os.environ.get(
    "INVERSETL_RESULTS_DIR", V8_ROOT / ("results_v8" if PROFILE == "legacy" else "results_pub")))
LOGS = V8_ROOT / ("logs_v8" if PROFILE == "legacy" else "logs_pub")
FIGS = V8_ROOT / "figures_v8"
ARCHIVED_PUB = V8_ROOT / "archived_inputs" / "pub"   # regenerated printed-pipeline checkpoints
for _d in (RESULTS, LOGS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

# the vendored code root takes priority over anything else on sys.path
if str(UPSTREAM_CODE) in sys.path:
    sys.path.remove(str(UPSTREAM_CODE))
sys.path.insert(0, str(UPSTREAM_CODE))

from src.utils.data_utils import get_bounds                                   # noqa: E402
from src.utils.seed_utils import set_global_seed                              # noqa: E402
import src.utils.physics_features as _upf                                      # noqa: E402
from src.utils.physics_features import (                                       # noqa: E402
    compute_physics_features_A, compute_physics_features_A_torch)
# upstream module references torch in its *_torch functions without importing
# it at module level — inject rather than edit the Paper-1 repo
if not hasattr(_upf, "torch"):
    _upf.torch = torch
from phys_features_B13 import compute_phys_B13_np, compute_phys_B13_torch      # noqa: E402
from phys_features_C13 import compute_phys_C13_np, compute_phys_C13_torch      # noqa: E402
from phys_features_C18 import compute_phys_C18_np, compute_phys_C18_torch      # noqa: E402
from src.simulation.materials import (                                          # noqa: E402
    get_sio2_permittivity, get_tio2_permittivity, get_metal_permittivity)


def compute_phys_A17_torch(geo, lam, metal="Cr"):
    """Corrected differentiable Structure-A features.

    The upstream `compute_physics_features_A_torch` permutes features 12-14
    relative to the canonical numpy implementation (which generated the TMM
    statistics and all training features): numpy order is
    [..., n_s*d1/lam, n_t*d2/lam, cos(theta), ...] but the upstream torch
    version emits [..., cos(theta), n_s*d1/lam, n_t*d2/lam, ...].
    This port follows the numpy order exactly (selfcheck-gated)."""
    P, Wx, Wy, W2 = geo[:, 0], geo[:, 1], geo[:, 2], geo[:, 3]
    t1, t2, t_mid = geo[:, 4], geo[:, 5], geo[:, 6]
    d1, d2, theta = geo[:, 7], geo[:, 8], geo[:, 9]
    theta_rad = theta * (np.pi / 180.0)
    lam_np = np.array([lam])
    n_s = float(np.sqrt(np.real(get_sio2_permittivity(lam_np))[0]))
    n_t = float(np.sqrt(np.real(get_tio2_permittivity(lam_np))[0]))
    k_m = float(np.imag(np.sqrt(get_metal_permittivity(lam_np, metal)[0])))
    sd = lam / (4 * np.pi * k_m + 1e-30)
    alpha = 4 * np.pi * k_m / lam
    feats = []
    for n_c, d_c in ((n_s, d1), (n_t, d2)):
        sin_ti = torch.clamp(torch.sin(theta_rad) / n_c, -1, 1)
        cos_ti = torch.sqrt(1 - sin_ti ** 2)
        phase = 4 * np.pi * n_c * d_c * cos_ti / lam
        feats += [torch.cos(phase), torch.sin(phase)]
    feats += [Wx * Wy / P ** 2, W2 ** 2 / P ** 2,
              P / lam, Wx / lam, W2 / lam,
              t1 / sd, t2 / sd, t_mid / sd,
              n_s * d1 / lam, n_t * d2 / lam,
              torch.cos(theta_rad), Wy / (Wx + 1e-10),
              torch.full_like(P, alpha)]
    return torch.stack(feats, dim=-1)

DEVICE = torch.device(os.environ.get("INVERSETL_DEVICE",
                                     "cuda" if torch.cuda.is_available() else "cpu"))

# ---------------------------------------------------------------- registry

def _bounds_arr(lo, hi):
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

_BA = _bounds_arr(*get_bounds("A")[1:])
_BB = _bounds_arr(*get_bounds("B")[1:])
# Structure D is not in the companion's data_utils: its design space lives in the
# contingency generator (structure_D/rcwa_struct_d.py:DESIGN_SPACE, 2026-07-03).
_D_NAMES = ["P", "L", "w", "t_Cr", "d_SiO2", "theta"]
_BD = np.array([[300.0, 100.0, 40.0, 20.0, 50.0, 0.0],
                [800.0, 720.0, 300.0, 80.0, 200.0, 60.0]], dtype=np.float64)
_BC_DESIGN = _bounds_arr(*get_bounds("C")[1:])
# pbtl_C.py BOUNDS_C — the normalization box the C checkpoint was trained with
_BC_TRAIN = _bounds_arr([300., 50., 50., 20., 50., 0., 0.],
                        [800., 400., 400., 80., 200., 60., 45.])

STRUCTS = {
    "A": dict(
        d=10, phys_dim=17, dual=False,
        param_names=get_bounds("A")[0],
        dataset=UPSTREAM / "data/raw/struct_A_vis_500.npz",
        ckpt=UPSTREAM / "results/pretrained_mphys_tmm.pt",
        design_bounds=_BA, train_bounds=_BA,
        np_feat=compute_physics_features_A,
        torch_feat=compute_phys_A17_torch,   # corrected port; upstream torch fn permutes cols 12-14
        tmm_wavelengths=np.linspace(380, 780, 100).astype(np.float32),
        tmm_params_dtype=np.float32,   # pbtl_A passes float32 params to the feature fn
        rcwa_module="src.simulation.rcwa_struct_a",
        expected_good=479,
    ),
    "B": dict(
        d=8, phys_dim=13, dual=False,
        param_names=get_bounds("B")[0],
        dataset=UPSTREAM / "data/raw/struct_B_500.npz",
        ckpt=UPSTREAM / "results/pretrained_mphys_tmm_B.pt",
        design_bounds=_BB, train_bounds=_BB,
        np_feat=compute_phys_B13_np,
        torch_feat=compute_phys_B13_torch,
        tmm_wavelengths=np.linspace(400, 1800, 100).astype(np.float32),
        tmm_params_dtype=np.float64,   # pbtl_B casts to float64 before the feature fn
        rcwa_module="src.simulation.rcwa_struct_b",
        expected_good=461,
    ),
    "C": dict(
        d=7, phys_dim=13, dual=True,
        param_names=get_bounds("C")[0],
        dataset=UPSTREAM / "data/raw/struct_C_500.npz",
        ckpt=UPSTREAM / "results/pretrained_mphys_tmm_C.pt",
        design_bounds=_BC_DESIGN, train_bounds=_BC_TRAIN,
        np_feat=compute_phys_C13_np,
        torch_feat=compute_phys_C13_torch,
        tmm_wavelengths=np.linspace(400, 1800, 100).astype(np.float32),
        tmm_params_dtype=np.float64,
        rcwa_module="src.simulation.rcwa_struct_c",
        expected_good=400,
    ),
}


def _dataset_wavelengths(path, fallback):
    try:
        return np.load(path, allow_pickle=True)["wavelengths"].astype(np.float32)
    except Exception:                                   # dataset absent on this host
        return fallback


if PROFILE == "pub":
    # Printed-pipeline overrides (upstream step0_screen/pbtl_{A,B}_redesign.py and
    # pbtl_C_v2_redesign.py): *_redesign pools with the `reliable` mask as the
    # hygiene filter (499/498/487), unified 400-1800 nm grid, regenerated
    # checkpoints, full-box C training bounds and the 18-feature C set.
    _pub_wl = np.linspace(400, 1800, 100).astype(np.float32)
    for _s, _ds, _ck, _n in (("A", "struct_A_500_redesign.npz", "pretrained_mphys_tmm_A_pub.pt", 499),
                             ("B", "struct_B_500_redesign.npz", "pretrained_mphys_tmm_B_pub.pt", 498),
                             ("C", "struct_C_500_redesign.npz", "pretrained_mphys_tmm_C_pub.pt", 487)):
        _cfg = STRUCTS[_s]
        _cfg["dataset"] = UPSTREAM / "data/raw" / _ds
        _cfg["ckpt"] = ARCHIVED_PUB / _ck
        _cfg["expected_good"] = _n
        _cfg["filter"] = "reliable"
        _cfg["tmm_wavelengths"] = _dataset_wavelengths(_cfg["dataset"], _pub_wl)
    STRUCTS["C"].update(phys_dim=18, np_feat=compute_phys_C18_np,
                        torch_feat=compute_phys_C18_torch, train_bounds=_BC_DESIGN)
    # Structure D (cross patch): the contingency fourth structure, generated 2026-07-08
    # with these same printed-pipeline settings.  It exists only in the pub arm -- there
    # is no as-submitted release of it -- so it is added here, not in the base table.
    from phys_features_D11 import compute_phys_D11_np, compute_phys_D11_torch  # noqa: E402
    STRUCTS["D"] = dict(
        d=6, phys_dim=11, dual=False,
        param_names=_D_NAMES,
        dataset=UPSTREAM / "data/raw/struct_D_500_redesign.npz",
        ckpt=ARCHIVED_PUB / "pretrained_mphys_tmm_D_pub.pt",
        design_bounds=_BD, train_bounds=_BD,
        np_feat=compute_phys_D11_np,
        torch_feat=compute_phys_D11_torch,
        tmm_wavelengths=_pub_wl,
        tmm_params_dtype=np.float64,
        rcwa_module="src.simulation.rcwa_struct_d",
        expected_good=494,
        filter="reliable",
    )
    STRUCTS["D"]["tmm_wavelengths"] = _dataset_wavelengths(STRUCTS["D"]["dataset"], _pub_wl)
else:
    for _cfg in STRUCTS.values():
        _cfg["filter"] = "legacy"

SPLIT_SEED = 42        # upstream rng_sp
TRAIN_SUBSET_SEED = 42 # upstream SEEDS[0]
TARGET_SEED = 42
TRAIN_SEED = 42        # set_global_seed before fine-tune
N_TEST, N_VAL, N_TRAIN_CAP = 50, 50, 350
N_TARGETS_BASE = 20                    # the pre-specified draw, frozen in the protocol
N_TARGETS = int(os.environ.get("INVERSETL_N_TARGETS", N_TARGETS_BASE))
N_RESTARTS = 8

# ---------------------------------------------------------------- models
# Exact upstream architectures (state_dict-compatible with the checkpoints).

class BaseResNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
            for _ in range(n_blocks)])
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks:
            h = h + self.act(b(h))
        return h


class MPhys(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd + pd)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(),
                                  nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x, p], -1))
        R = self.head(h).squeeze(-1)
        return {"A": 1 - R, "R": R}


class MPhysDual(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd + pd)
        self.head_te = nn.Sequential(nn.Linear(256, 128), nn.SiLU(),
                                     nn.Linear(128, 1), nn.Sigmoid())
        self.head_tm = nn.Sequential(nn.Linear(256, 128), nn.SiLU(),
                                     nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x, p], -1))
        Rte = self.head_te(h).squeeze(-1)
        Rtm = self.head_tm(h).squeeze(-1)
        return {"A_TE": 1 - Rte, "R_TE": Rte, "A_TM": 1 - Rtm, "R_TM": Rtm}


def make_model(s):
    cfg = STRUCTS[s]
    cls = MPhysDual if cfg["dual"] else MPhys
    return cls(1 + cfg["d"], cfg["phys_dim"]).to(DEVICE)


# ---------------------------------------------------------------- data

def load_filtered(s):
    """Load dataset, apply the Paper-1 hygiene filter, clip to [0,1].
    Returns dict with params, per-channel arrays, wavelengths, and index maps."""
    cfg = STRUCTS[s]
    raw = np.load(cfg["dataset"], allow_pickle=True)
    params = raw["params"].astype(np.float64)
    wl = raw["wavelengths"].astype(np.float64)
    if cfg["filter"] == "reliable":
        # printed pipeline: per-(sample, wavelength) reliability mask, all-true rows
        if cfg["dual"]:
            good = raw["reliable_TE"].all(axis=1) & raw["reliable_TM"].all(axis=1)
        else:
            good = raw["reliable"].all(axis=1)
    elif cfg["dual"]:
        good = (np.all(raw["A_TE"] >= -0.01, axis=1)
                & np.all(raw["A_TM"] >= -0.01, axis=1))
    else:
        good = np.all(raw["A"] >= -0.01, axis=1)
    keys = ("A_TE", "R_TE", "A_TM", "R_TM") if cfg["dual"] else ("A", "R")
    chans = {k: np.clip(raw[k][good], 0, 1).astype(np.float32) for k in keys}
    gi = np.where(good)[0]
    if len(gi) != cfg["expected_good"]:
        raise RuntimeError(f"[{s}] good-sample count {len(gi)} != expected "
                           f"{cfg['expected_good']} — dataset changed?")
    return dict(params=params[good], chans=chans, wavelengths=wl,
                orig_idx=gi, n_good=len(gi))


def make_split(n_good, subset_seed=TRAIN_SUBSET_SEED):
    """Upstream stage-2 split: rng(42) permutation; test=last 50, val=50 before,
    train = first min(350, pool) of an rng(subset_seed) permutation of the rest.
    Test/val are carved out with the fixed SPLIT_SEED first, so they are
    IDENTICAL across training seeds — multi-seed runs share the same targets."""
    perm = np.random.default_rng(SPLIT_SEED).permutation(n_good)
    test_idx = perm[-N_TEST:]
    val_idx = perm[-(N_TEST + N_VAL):-N_TEST]
    remaining = perm[:-(N_TEST + N_VAL)]
    rng2 = np.random.default_rng(subset_seed)
    train_idx = remaining[rng2.permutation(len(remaining))[:N_TRAIN_CAP]]
    return train_idx, val_idx, test_idx


def pick_targets(test_idx, n=None):
    """Targets for one run, in an order that never moves an existing design.

    The protocol draws 20 with rng(TARGET_SEED) from the held-out split.  Raising
    INVERSETL_N_TARGETS appends the rest of that split in index order instead of
    redrawing, so design i keeps its geometry -- and therefore its oracle cache entry --
    when the run is extended from 20 targets to the full 50."""
    n = N_TARGETS if n is None else n
    test_idx = np.asarray(test_idx)
    base = np.random.default_rng(TARGET_SEED).choice(test_idx, size=min(n, N_TARGETS_BASE),
                                                     replace=False)
    if n <= len(base):
        return base
    rest = np.array([i for i in np.sort(test_idx) if i not in set(base.tolist())], dtype=base.dtype)
    if n - len(base) > len(rest):
        raise ValueError(f"asked for {n} targets but the held-out split has "
                         f"{len(test_idx)}")
    return np.concatenate([base, rest[:n - len(base)]])


def load_surrogate(s, sx):
    """Load surrogate_<s><sx>_v8.pt and return (model, identity).  The sha256 is taken of the
    exact bytes handed to torch.load -- not of the file re-read later, which could differ if
    the checkpoint were replaced mid-run -- and a missing file is an error, never a
    "missing" identity (codex full-scope review, 2026-09-12, slices 2 and 6)."""
    import hashlib, io, torch
    ck = RESULTS / f"surrogate_{s}{sx}_v8.pt"
    if not ck.exists():
        raise FileNotFoundError(f"surrogate checkpoint {ck} does not exist; refusing to fall back")
    raw = ck.read_bytes()
    model = make_model(s)
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location=DEVICE, weights_only=True))
    model.eval()
    ident = dict(surrogate_sha256=hashlib.sha256(raw).hexdigest(), surrogate_path=ck.name,
                 finetune_log=f"finetune_{s}{sx}_v8.json", input_sfx=sx)
    return model, ident


def surrogate_identity(s, sx):
    """Identity of the checkpoint on disk right now.  Only for comparison against a recorded
    identity; producers must record the identity returned by load_surrogate instead."""
    import hashlib
    ck = RESULTS / f"surrogate_{s}{sx}_v8.pt"
    if not ck.exists():
        raise FileNotFoundError(f"surrogate checkpoint {ck} does not exist")
    return dict(surrogate_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(), surrogate_path=ck.name,
                finetune_log=f"finetune_{s}{sx}_v8.json", input_sfx=sx)


def assert_same_surrogate(recorded, s, sx, what):
    """recorded: the identity dict (or npz field) an upstream artifact carries."""
    now = surrogate_identity(s, sx)
    rec = str(recorded)
    if rec == "missing":
        raise RuntimeError(f"{what}: upstream artifact records a 'missing' surrogate identity; "
                           "that value is never legitimate for a freshly produced artifact")
    if rec != now["surrogate_sha256"]:
        raise RuntimeError(f"{what}: upstream artifact was produced by surrogate "
                           f"{rec[:12]}… but this run resolved {now['surrogate_path']} "
                           f"({now['surrogate_sha256'][:12]}…). Refusing to mix them.")


def seed_sfx(seed):
    """Artifact-name suffix: '' for the canonical seed 42, '_s<seed>' otherwise, plus an
    optional experiment tag from INVERSETL_TAG (e.g. '_m0', '_feas') so that new
    experiments never overwrite the archived artifacts (T04)."""
    return ("" if int(seed) == 42 else f"_s{seed}") + os.environ.get("INVERSETL_TAG", "")


def input_sfx(seed, kind):
    """Suffix for READING an upstream-stage artifact `kind` (e.g. 'finetune', 'stats',
    'surrogate', 'inverse') for this seed: the tagged name if that stage was itself run
    under the current INVERSETL_TAG (e.g. the M0 control re-fine-tunes with '_m0'),
    otherwise the untagged seed-only name (e.g. the feasible rerun '_feas' reuses the
    archived surrogate).  Raises if neither exists."""
    tagged = seed_sfx(seed)
    plain = "" if int(seed) == 42 else f"_s{seed}"
    ext = {"finetune": ".json", "stats": ".npz", "surrogate": ".pt", "inverse": ".npz"}[kind]
    for sfx in (tagged, plain):
        if (RESULTS / f"{kind}_{{s}}{sfx}_v8{ext}").exists() or any(
                (RESULTS / f"{kind}_{st}{sfx}_v8{ext}").exists() for st in STRUCTS):
            return sfx
    raise FileNotFoundError(f"no {kind}_*{tagged}_v8{ext} or {kind}_*{plain}_v8{ext} in {RESULTS}")


def feasible_mask(s, params):
    """Generator feasibility constraints of the Paper-1 datasets (upstream
    rcwa_struct_{a,b,c}.py sampling clamps):  A: Wx, Wy, W2 <= 0.9 P;  C: Wx, Wy <= 0.9 P;
    B: R_out <= 0.45 P, R_in <= R_out - 10, R_disk <= R_in - 10;  D: L <= 0.9 P, w <= L.
    params (N, d) in
    physical units, columns ordered as STRUCTS[s]['param_names'].  Returns bool (N,)."""
    params = np.asarray(params, dtype=np.float64)
    names = list(STRUCTS[s]["param_names"])
    col = {n: params[:, i] for i, n in enumerate(names)}
    # Relative, because the geometries come out of a float32 pipeline: a restart clamped to
    # u = 1 lands on the cap, and lo + 1.0*(min(0.9 P, hi) - lo) rounds up to ~1e-5 nm above
    # 0.9 P.  An absolute 1e-9 rejected those (2026-09-10, inverse --feasible on A), which is
    # a tolerance for float64 applied to float32 output.  1e-6 relative is ~5e-4 nm at these
    # magnitudes -- far below anything physical, far above the rounding, and still orders of
    # magnitude tighter than a real reparametrization bug, which misses by nanometres.
    tol = 1e-6 * float(np.abs(params).max() if params.size else 1.0)
    if s == "D":
        # structure_D/rcwa_struct_d.py:146 clamps L to 0.9 P and w to L when sampling
        return (col["L"] <= 0.9 * col["P"] + tol) & (col["w"] <= col["L"] + tol)
    if s == "B":
        return ((col["R_out"] <= 0.45 * col["P"] + tol)
                & (col["R_in"] <= col["R_out"] - 10 + tol)
                & (col["R_disk"] <= col["R_in"] - 10 + tol))
    ws = ("Wx", "Wy", "W2") if s == "A" else ("Wx", "Wy")
    ok = np.ones(len(params), dtype=bool)
    for w in ws:
        ok &= col[w] <= 0.9 * col["P"] + tol
    return ok


def env_info():
    """Software/hardware provenance written into every stage's output (T04)."""
    import platform
    info = dict(profile=PROFILE, upstream_commit=UPSTREAM_COMMIT, upstream_code=str(UPSTREAM_CODE),
                python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device=str(DEVICE))
    try:
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["gpu"] = "unknown"
    try:
        import torcwa
        info["torcwa"] = getattr(torcwa, "__version__", "unknown")
    except Exception:
        info["torcwa"] = "unavailable"
    return info


def tmm_phys_stats(s):
    """Deterministic reconstruction of the TMM-set feature statistics the
    pretrained checkpoint was normalized with (rng(99), N=5000, train bounds)."""
    cfg = STRUCTS[s]
    lo, hi = cfg["train_bounds"]
    rng = np.random.default_rng(99)
    p = rng.uniform(lo, hi, (5000, cfg["d"])).astype(np.float32)
    wl = cfg["tmm_wavelengths"]
    if cfg["tmm_params_dtype"] == np.float64:
        feats = cfg["np_feat"](p.astype(np.float64), wl.astype(np.float64), "Cr")
    else:
        feats = cfg["np_feat"](p, wl, "Cr")
    flat = feats.reshape(-1, cfg["phys_dim"]).astype(np.float32)
    return flat.mean(0), flat.std(0) + 1e-8


def normalize_params_train(s, params):
    lo, hi = STRUCTS[s]["train_bounds"]
    return (params - lo) / (hi - lo)


def build_xgeo(s, params_phys, wavelengths):
    """Row-major (n*L, 1+d) geometry input, wavelength FIRST (upstream order)."""
    cfg = STRUCTS[s]
    n, L = len(params_phys), len(wavelengths)
    pn = normalize_params_train(s, params_phys).astype(np.float32)
    wln = ((wavelengths - wavelengths.min())
           / (wavelengths.max() - wavelengths.min())).astype(np.float32)
    X = np.empty((n * L, 1 + cfg["d"]), dtype=np.float32)
    for i in range(n):
        X[i * L:(i + 1) * L, 0] = wln
        X[i * L:(i + 1) * L, 1:] = pn[i]
    return X


def build_phys(s, params_phys, wavelengths, phys_mean, phys_std):
    cfg = STRUCTS[s]
    feats = cfg["np_feat"](params_phys.astype(np.float64),
                           wavelengths.astype(np.float64), "Cr").astype(np.float32)
    return ((feats - phys_mean) / phys_std).reshape(-1, cfg["phys_dim"])


def a_channel_keys(s):
    return ("A_TE", "A_TM") if STRUCTS[s]["dual"] else ("A",)


def spectrum_mae_pct(s, pred_chans, true_chans):
    """Protocol Tier-1 spectrum MAE in % — A channel; C: mean of TE/TM."""
    keys = a_channel_keys(s)
    return float(np.mean([np.mean(np.abs(pred_chans[k] - true_chans[k]))
                          for k in keys]) * 100.0)
