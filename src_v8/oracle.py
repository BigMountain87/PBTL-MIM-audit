"""RCWA reference-solver plumbing shared by rcwa_validate / random_baseline /
restart0_validate (T53).

* configure_solver(mod, order)  — pins the solver per profile:
    legacy : vendored 920b1bd, fixed Fourier order [5,5] (or --order N), complex128 local;
             asserts the module exposes no 'adaptive_order' key.
    pub    : vendored cb486b5, the data-generating fidelity = per-wavelength ADAPTIVE
             order (N in {9,13,17}, complex64, J&C materials); --order N switches the
             module to a fixed order N with adaptive off (explicit convergence probes).
* simulate_cached(...)          — one design = one cache file
    RESULTS/oracle_cache/<kind>_<S><sx>_<i>.npz written atomically right after the
    solve, so any driver can be killed and resumed and several shard workers can own
    disjoint designs (owned = {i : i % nshards == shard}); the final artifact is
    assembled from the cache by whichever worker finishes last.
* wait_for_vram(need_gb)        — pre-solve VRAM gate (adaptive N=17 needs ~7 GB for A).
"""
from __future__ import annotations
import inspect, json, os, re, time
import numpy as np
import torch

from common import RESULTS, PROFILE, UPSTREAM_CODE, UPSTREAM_COMMIT, DEVICE
import hashlib, pathlib

CACHE = RESULTS / "oracle_cache"
# Peak VRAM per solve at the adaptive order in complex64 (upstream generate_redesign.py quotes
# 7.5/5.0/5.0 GB for complex128; c64 halves it, and measured RSS on this host is 4.5-6.5 GB).
PEAK_GB = {"A": 5.0, "B": 3.5, "C": 3.5, "D": 3.5}   # D: 6 params, single channel, C-like
MEM_WAIT_MAX_S = 1800.0
ALLOW_UNSTAMPED = os.environ.get("INVERSETL_ALLOW_UNSTAMPED_CACHE") == "1"   # migration window only
OOM_RETRY = 3                                  # retries per design on CUDA OOM
_DTYPE_RX = re.compile(r"sim_dtype\s*=\s*torch\.(complex\d+)")


def solver_dtype(mod) -> str:
    """Complex dtype the module actually uses: RCWA_SETTINGS['dtype'] at cb486b5, a
    hard-coded local at 920b1bd (read from source)."""
    if "dtype" in getattr(mod, "RCWA_SETTINGS", {}):
        return str(mod.RCWA_SETTINGS["dtype"])
    try:
        hits = set(_DTYPE_RX.findall(inspect.getsource(mod)))
    except OSError:
        return "unknown"
    return f"torch.{hits.pop()}" if len(hits) == 1 else ("mixed:" + ",".join(sorted(hits)) if hits else "not-found")


def torcwa_version() -> str:
    try:
        import torcwa
        return getattr(torcwa, "__version__", "unknown")
    except Exception as exc:                                    # pragma: no cover
        return f"unavailable ({exc.__class__.__name__})"


_PROVENANCE = None            # set by configure_solver; part of every cache fingerprint

ADAPTIVE_N_MAX = 17           # ceiling of every vendored adaptive_order rule (rcwa_struct_{a,b,c,d}.py)


def grid_sufficiency(st, adaptive, order):
    """torcwa's _material_conv reads the FFT of the nx*ny raster at harmonic differences
    up to +-2N, but an nx-point FFT only holds -nx/2 .. nx/2-1, so nx < 4N+2 wraps the
    outermost couplings around (docs/notes_rcwa_grid_truncation_v11.md).  Reported, not
    enforced: the pinned solver is what generated the labels and must stay as it is."""
    n_max = ADAPTIVE_N_MAX if adaptive else int(order)
    nx = int(min(st["grid"]))
    need = 4 * n_max + 2
    return dict(grid_n_max=n_max, grid_required=need, grid_sufficient=nx >= need)


def configure_solver(mod, order=None):
    """Pin mod.RCWA_SETTINGS per profile; returns a provenance dict."""
    global _PROVENANCE
    if not hasattr(mod, "RCWA_SETTINGS"):
        raise RuntimeError(f"{mod.__name__} has no RCWA_SETTINGS")
    st = mod.RCWA_SETTINGS
    if PROFILE == "legacy":
        assert "adaptive_order" not in st, (
            f"{mod.__name__} exposes 'adaptive_order' — code root is not the vendored "
            f"920b1bd tree (UPSTREAM_CODE={UPSTREAM_CODE})")
        st["order"] = [order or 5] * 2
        adaptive = False
    else:
        assert "adaptive_order" in st, (
            f"{mod.__name__} lacks 'adaptive_order' — code root is not the vendored "
            f"cb486b5 tree (UPSTREAM_CODE={UPSTREAM_CODE})")
        if order is None:
            st["adaptive_order"] = True          # data-generating fidelity of the printed release
            adaptive = True
        else:
            st["adaptive_order"] = False
            st["order"] = [order, order]
            adaptive = False
    _PROVENANCE = live_provenance(mod)
    enforce_profile_contract(_PROVENANCE)
    # Kept out of _PROVENANCE on purpose: it feeds every cache fingerprint, and adding a
    # key there would invalidate every archived solve.
    grid = grid_sufficiency(st, adaptive, None if adaptive else st["order"][0])
    if not grid["grid_sufficient"]:
        print(f"  [oracle] NOTE: real-space grid {tuple(st['grid'])} < 4*N_max+2 = {grid['grid_required']} "
              f"for N_max = {grid['grid_n_max']}: couplings at |m-n| > {min(st['grid']) // 2 - 1} alias "
              f"(pinned data-generating setting, reported not changed)", flush=True)
    return dict(_PROVENANCE, **grid)


def _materials():
    try:
        import src.simulation.materials as mats
        return str(getattr(mats, "MATERIAL_MODEL", "legacy-analytic (no MATERIAL_MODEL attribute)"))
    except Exception:                                          # pragma: no cover
        return "unknown"


def live_provenance(mod):
    """Solver identity read from the module AS IT IS NOW, not from a snapshot.  The
    fingerprint used to hash a copy taken at configure time, so mutating RCWA_SETTINGS
    afterwards left old cache entries acceptable (codex full-scope review, 2026-09-12)."""
    st = mod.RCWA_SETTINGS
    adaptive = bool(st.get("adaptive_order", False))
    mf = getattr(mod, "__file__", None)
    try:                      # identity of the implementation, not just its settings
        src_sha = hashlib.sha256(pathlib.Path(mf).read_bytes()).hexdigest() if mf else "no-file"
    except OSError:
        src_sha = "unreadable"
    return dict(profile=PROFILE, upstream_commit=UPSTREAM_COMMIT, rcwa_settings=str(dict(st)),
                sim_dtype=solver_dtype(mod), adaptive=adaptive,
                fixed_order=None if adaptive else int(st["order"][0]),
                materials=_materials(), torcwa_version=torcwa_version(),
                module_file=str(mf or "?"), module_sha256=src_sha)


# What each profile's solver must look like.  configure_solver used to record these
# and enforce nothing, so a run under the pub profile with complex128 or the legacy
# material model would have been accepted and merely logged.
CONTRACT = {"legacy": dict(sim_dtype="torch.complex128", materials_startswith="legacy-analytic"),
            "pub":    dict(sim_dtype="torch.complex64",  materials_startswith="jc")}


def enforce_profile_contract(prov):
    if os.environ.get("INVERSETL_ALLOW_SOLVER_DEVIATION") == "1":
        print(f"  [oracle] profile contract NOT enforced (INVERSETL_ALLOW_SOLVER_DEVIATION=1): {prov['sim_dtype']}, {prov['materials']}", flush=True)
        return
    want = CONTRACT[PROFILE]
    bad = []
    if prov["sim_dtype"] != want["sim_dtype"]:
        bad.append(f"dtype {prov['sim_dtype']} != {want['sim_dtype']}")
    if not prov["materials"].startswith(want["materials_startswith"]):
        bad.append(f"materials {prov['materials']!r} not {want['materials_startswith']!r}")
    root = UPSTREAM_CODE.resolve()
    try:
        inside = pathlib.Path(prov["module_file"]).resolve().is_relative_to(root)
    except (ValueError, OSError):
        inside = False
    if not inside:
        bad.append(f"module loaded from {prov['module_file']}, not inside the pinned tree {root}")
    if prov.get("module_sha256") in (None, "no-file", "unreadable"):
        bad.append("solver module source could not be hashed")
    if bad:
        raise RuntimeError(f"solver does not match the {PROFILE} profile contract: " + "; ".join(bad))


def orders_used(mod, s, pdict, wl):
    """Per-wavelength Fourier order the module will use for this design."""
    st = mod.RCWA_SETTINGS
    if not st.get("adaptive_order", False):
        return [int(st["order"][0])] * len(wl)
    f = mod.adaptive_order
    out = []
    for lam in wl:
        if s == "A":
            o = f(float(lam), pdict["P"], pdict["Wx"], pdict["Wy"], pdict["W2"])
        elif s == "D":
            o = f(float(lam), pdict["P"], pdict["L"], pdict["w"])
        elif s == "C":
            o = f(float(lam), pdict["P"], pdict["Wx"], pdict["Wy"])
        else:
            o = f(float(lam), pdict["P"])
        out.append(int(o[0]) if isinstance(o, (list, tuple)) else int(o))
    return out


def wait_for_vram(need_gb, tag=""):
    """Block until need_gb of VRAM is free.  Releasing our own cached blocks first is what makes
    this safe with several workers: without it every worker sits on its allocator cache and they
    all wait for each other (observed deadlock, 2026-09-06)."""
    if not torch.cuda.is_available():
        return
    t0 = time.time()
    while True:
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        if free / 1e9 >= need_gb or time.time() - t0 > MEM_WAIT_MAX_S:
            return
        if int(time.time() - t0) % 60 < 5:
            print(f"  [{tag}] waiting for VRAM: {free/1e9:.1f} GB free < {need_gb} GB", flush=True)
        time.sleep(5)


def owned(n, shard=0, nshards=1):
    return [i for i in range(n) if i % nshards == shard]


def cache_path(kind, s, sx, i):
    return CACHE / f"{kind}_{s}{sx}_{i:03d}.npz"


def fingerprint(s, pdict, wl, cfg, mod=None):
    """Identity of a cached solve: the pinned solver configuration, the geometry and the
    wavelength grid.  Without this the cache is keyed by index alone, so re-running under
    a different --order would return the old spectra while the artifact metadata claimed
    the newly requested solver (codex review 2026-09-09, item 10)."""
    if _PROVENANCE is None:
        raise RuntimeError("configure_solver() must run before the cache is consulted")
    prov = live_provenance(mod) if mod is not None else _PROVENANCE
    h = hashlib.sha256()
    h.update(json.dumps(prov, sort_keys=True).encode())
    h.update(np.asarray([pdict[k] for k in cfg["param_names"]], dtype=np.float64).tobytes())
    h.update(np.asarray(wl, dtype=np.float64).tobytes())
    h.update(s.encode())
    return h.hexdigest()


def simulate_cached(kind, s, sx, i, pdict, wl, cfg, mod, targets=None):
    """Simulate one design (or load it from the cache).  Returns dict with keys
    A (dict channel -> (L,) array), failed, error, elapsed, orders, and mae (if targets given).
    `targets` is a dict channel -> (L,) target absorptance used for the MAE."""
    a_keys = ("A_TE", "A_TM") if cfg["dual"] else ("A",)
    path = cache_path(kind, s, sx, i)
    enforce_profile_contract(live_provenance(mod))       # the contract holds per solve, not per configure
    fp = fingerprint(s, pdict, wl, cfg, mod)
    stale = None
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if "fingerprint" in z.files:
            if str(z["fingerprint"]) != fp:
                stale = f"fingerprint {str(z['fingerprint'])[:12]} != {fp[:12]}"
        else:
            # written before fingerprints existed: check the fields those files do record.
            # The per-wavelength orders catch a changed solver configuration on their own.
            if not np.array_equal(np.asarray(z["wavelengths"], float), np.asarray(wl, float)):
                stale = "wavelength grid differs"
            elif not np.allclose(np.asarray(z["params"], float),
                                 [pdict[k] for k in cfg["param_names"]], rtol=0, atol=1e-9):
                stale = "geometry differs"
            elif np.asarray(z["orders"]).tolist() != orders_used(mod, s, pdict, wl):
                stale = "per-wavelength Fourier orders differ"
        if not stale:
            # whatever wrote the entry, its spectra must be finite and the right length
            for k in a_keys:
                v = np.asarray(z[f"A_{k}"])
                if not bool(z["failed"]) and (v.shape != (len(wl),) or not np.all(np.isfinite(v))):
                    stale = f"cached spectrum {k} is nonfinite or wrong length"
                    break
        if not stale and "fingerprint" not in z.files and not ALLOW_UNSTAMPED:
            # An entry with no solver identity cannot be shown to come from this solver.
            # T54's entries were stamped by scripts/stamp_oracle_cache_v10.py after their
            # assembled artifacts were checked; anything else unstamped is recomputed.
            stale = "no solver identity recorded (run stamp_oracle_cache_v10.py for T54 entries)"
        if stale:
            print(f"  [{kind} {s} {i}] cache entry rejected ({stale}) — recomputing", flush=True)
            path.unlink()      # a rejected file must not remain to be mistaken for completion
    if path.exists() and not stale:
        res = dict(A={k: z[f"A_{k}"] for k in a_keys}, failed=bool(z["failed"]),
                   error=str(z["error"]), elapsed=float(z["elapsed"]), orders=z["orders"].tolist(),
                   cached=True)
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        orders = orders_used(mod, s, pdict, wl)
        wait_for_vram(PEAK_GB[s] if max(orders) >= 17 else 2.0, tag=f"{kind} {s} {i}")
        t0 = time.time()
        A = {k: np.full(len(wl), np.nan) for k in a_keys}
        failed, error, was_oom = False, "", False
        for attempt in range(OOM_RETRY + 1):
            try:
                out = mod.simulate_single(pdict, np.asarray(wl, dtype=np.float64), metal="Cr", device=DEVICE)
                if cfg["dual"]:
                    A["A_TE"], A["A_TM"] = np.asarray(out[0]), np.asarray(out[3])
                else:
                    A["A"] = np.asarray(out[0])
                # a solve that returns NaN or the wrong length is a failure, not a success
                # with a NaN MAE (it used to be marked failed=False and cached as such)
                bad = [k for k, v in A.items()
                       if v.shape != (len(wl),) or not np.all(np.isfinite(v))]
                if bad:
                    failed, error = True, f"nonfinite or wrong-shape output in {bad}"
                else:
                    failed, error = False, ""
                break
            except Exception as exc:
                failed, error = True, f"{exc.__class__.__name__}: {str(exc)[:200]}"
                was_oom = "out of memory" in str(exc).lower()     # decided on the full message
                # transient shared-GPU contention: free, wait, retry — never cache an OOM
                if was_oom and attempt < OOM_RETRY:
                    print(f"  [{kind} {s} {i}] CUDA OOM (attempt {attempt+1}) — retrying in 60 s", flush=True)
                    torch.cuda.empty_cache(); time.sleep(60)
                    wait_for_vram(PEAK_GB[s], tag=f"{kind} {s} {i}")
                    continue
                break
        elapsed = time.time() - t0
        torch.cuda.empty_cache() if torch.cuda.is_available() else None   # hand memory back between designs
        if failed and was_oom:
            # every retry hit OOM: a resource problem, not a property of the design.
            # Return the failure so the caller can retry the run, but never write it to
            # the cache, where it would become a permanent hole in the denominator.
            print(f"  [{kind} {s} {i}] OOM after {OOM_RETRY + 1} attempts — NOT cached", flush=True)
            res = dict(A=A, failed=True, error=error, elapsed=elapsed, orders=orders, cached=False,
                       transient=True)                       # structured: assembly must not treat as done
            res["mae"] = float("nan")
            return res
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **{f"A_{k}": v for k, v in A.items()}, failed=failed, error=error,
                 elapsed=elapsed, orders=np.array(orders), params=np.array([pdict[k] for k in cfg["param_names"]]),
                 wavelengths=np.asarray(wl), kind=kind, structure=s, index=i,
                 fingerprint=fp, solver=json.dumps(live_provenance(mod), sort_keys=True))
        os.replace(tmp, path)
        res = dict(A=A, failed=failed, error=error, elapsed=elapsed, orders=orders, cached=False)
    if targets is not None and not res["failed"]:
        res["mae"] = float(np.mean([np.mean(np.abs(res["A"][k] - targets[k])) for k in a_keys]) * 100)
    else:
        res["mae"] = float("nan")
    return res


def all_cached(kind, s, sx, idxs):
    """True only if every design has a cache entry that carries a solver identity (or the
    migration window is open).  Existence alone used to count, so a stale file left by a
    rejected entry, or a run that ended in a transient failure, looked complete."""
    for i in idxs:
        f = cache_path(kind, s, sx, i)
        if not f.exists():
            return False
        if not ALLOW_UNSTAMPED:
            with np.load(f, allow_pickle=True) as z:
                if "fingerprint" not in z.files:
                    return False
    return True
