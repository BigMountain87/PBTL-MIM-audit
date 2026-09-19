#!/usr/bin/env python3
"""Reference-solver (RCWA) validation of committed inverse-design geometries.

Profile legacy: dataset-generation settings of the as-submitted release
  (64×64 grid, fixed Fourier order [5,5], complex128, vendored 920b1bd).
Profile pub   : data-generating fidelity of the printed release — per-wavelength
  adaptive Fourier order (N ∈ {9,13,17}), complex64, J&C materials, vendored cb486b5.
--order N forces a fixed order N (legacy: order override; pub: adaptive OFF).
--subset "3,9,16" --tag conv7 re-simulates a subset into a tagged artifact.
--shard k --nshards N: this worker owns designs {i : i % N == k}; every design is
cached under results*/oracle_cache/ as soon as it is solved (resume-safe), and the
worker that completes the set assembles the final npz.
"""
from __future__ import annotations
import argparse, importlib, json, os, time
import numpy as np

from common import STRUCTS, RESULTS, DEVICE, seed_sfx, UPSTREAM_COMMIT, env_info
from oracle import configure_solver, simulate_cached, owned, all_cached


def main(s, order=None, subset=None, tag="", seed=42, materials="legacy", shard=0, nshards=1):
    if subset is not None and not tag:
        raise SystemExit("--subset requires --tag (would overwrite the main artifact)")
    cfg = STRUCTS[s]
    sx = seed_sfx(seed)
    mod = importlib.import_module(cfg["rcwa_module"])
    try:
        import src.simulation.materials as mats
        if hasattr(mats, "MATERIAL_MODEL") and materials != "legacy":
            mats.MATERIAL_MODEL = materials
    except Exception:                                          # pragma: no cover
        pass
    prov = configure_solver(mod, order)
    print(f"[rcwa {s} v8] upstream={UPSTREAM_COMMIT} profile={prov['profile']} "
          f"settings={prov['rcwa_settings']} dtype={prov['sim_dtype']} "
          f"materials={prov['materials']} torcwa={prov['torcwa_version']}", flush=True)

    inv = np.load(RESULTS / f"inverse_{s}{sx}_v8.npz", allow_pickle=True)
    best = inv["best_params"]
    wl = inv["wavelengths"].astype(np.float64)
    names = [str(x) for x in inv["param_names"]]
    a_keys = ("A_TE", "A_TM") if cfg["dual"] else ("A",)
    n = len(best)
    idxs = list(range(n)) if subset is None else [int(x) for x in subset.split(",")]
    mine = [i for i in idxs if i in set(owned(n, shard, nshards))]
    kind = "rcwa" + (f"_{tag}" if tag else "")
    print(f"[rcwa {s} v8] {len(mine)}/{len(idxs)} geometries (shard {shard}/{nshards}), device={DEVICE}, "
          f"wl {wl.min():.0f}-{wl.max():.0f} nm, "
          f"order={'adaptive' if prov['adaptive'] else prov['fixed_order']}", flush=True)

    t_all = time.time()
    for i in mine:
        pdict = {k: float(best[i, j]) for j, k in enumerate(names)}
        tgt = {k: inv[f"A_target_{k}"][i] for k in a_keys}
        r = simulate_cached(kind, s, sx, i, pdict, wl, cfg, mod, targets=tgt)
        if r["failed"]:
            print(f"  [{i+1:2d}/{n}] FAILED ({r['elapsed']:.1f}s): {r['error']}", flush=True)
        else:
            print(f"  [{i+1:2d}/{n}] RCWA MAE={r['mae']:6.2f}%  N={min(r['orders'])}-{max(r['orders'])}  "
                  f"({r['elapsed']:.1f}s{', cached' if r['cached'] else ''})", flush=True)
    print(f"[rcwa {s} v8] shard done in {time.time()-t_all:.0f}s", flush=True)

    if not all_cached(kind, s, sx, idxs):
        print(f"[rcwa {s} v8] other shards still running — final artifact not assembled by this worker", flush=True)
        return

    # ---- assemble the final artifact from the cache (identical layout to the archived npz)
    A_rcwa = {k: np.full((n, len(wl)), np.nan) for k in a_keys}
    elapsed = np.zeros(n); failed = np.ones(n, dtype=bool)
    orders_min = np.zeros(n, dtype=int); orders_max = np.zeros(n, dtype=int)
    mae_rcwa = np.full(n, np.nan)
    for i in idxs:
        pdict = {k: float(best[i, j]) for j, k in enumerate(names)}
        tgt = {k: inv[f"A_target_{k}"][i] for k in a_keys}
        r = simulate_cached(kind, s, sx, i, pdict, wl, cfg, mod, targets=tgt)   # cache hit
        elapsed[i] = r["elapsed"]; failed[i] = r["failed"]
        orders_min[i], orders_max[i] = min(r["orders"]), max(r["orders"])
        if not r["failed"]:
            for k in a_keys:
                A_rcwa[k][i] = r["A"][k]
            mae_rcwa[i] = r["mae"]
    mae_surr = inv["mae_surrogate"]
    delta = mae_rcwa - mae_surr
    print(f"\n=== Per-sample ({s} v8{' '+tag if tag else ''}) ===", flush=True)
    for i in idxs:
        if not failed[i]:
            print(f"  [{i+1:2d}] surr={mae_surr[i]:6.2f}%  rcwa={mae_rcwa[i]:6.2f}%  Δ={delta[i]:+6.2f}%", flush=True)

    out_d = dict(best_params=best, mae_surrogate=mae_surr, mae_rcwa=mae_rcwa, delta=delta,
                 failed=failed, elapsed=elapsed, wavelengths=wl, param_names=inv["param_names"],
                 # provenance carried forward from the inverse artifact so every downstream
                 # file names the surrogate that produced these geometries
                 **{k: inv[k] for k in ("surrogate_sha256", "surrogate_path", "feasible",
                                        "parametrization") if k in inv.files},
                 order=(-1 if prov["adaptive"] else prov["fixed_order"]),
                 orders_min=orders_min, orders_max=orders_max, adaptive=prov["adaptive"],
                 rcwa_settings=prov["rcwa_settings"], sim_dtype=prov["sim_dtype"],
                 torcwa_version=prov["torcwa_version"], upstream_commit=UPSTREAM_COMMIT,
                 materials=prov["materials"], env=json.dumps(env_info()))
    for k in a_keys:
        out_d[f"A_rcwa_{k}"] = A_rcwa[k]
        out_d[f"A_target_{k}"] = inv[f"A_target_{k}"]
    path = RESULTS / f"rcwa_{s}{sx}_v8{('_'+tag) if tag else ''}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out_d); os.replace(tmp, path)
    ok = int((~failed[idxs]).sum())
    print(f"[rcwa {s} v8] wrote {path.name}; {ok}/{len(idxs)} success, "
          f"{elapsed[idxs].sum():.0f}s solver time", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True, choices=list(STRUCTS))
    ap.add_argument("--order", type=int, default=None,
                    help="fixed Fourier order (legacy default 5; pub default = adaptive)")
    ap.add_argument("--subset", default=None, help="comma-separated 0-based sample indices")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--materials", default="legacy", choices=["legacy", "jc"],
                    help="applied only if the vendored materials module exposes MATERIAL_MODEL")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    main(a.structure, order=a.order, subset=a.subset, tag=a.tag, seed=a.seed,
         materials=a.materials, shard=a.shard, nshards=a.nshards)
