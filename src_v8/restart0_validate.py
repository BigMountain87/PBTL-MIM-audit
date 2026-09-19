#!/usr/bin/env python3
"""v8 single-start control — oracle-validate the restart-0 endpoints.

Restart 0 of every target starts at the deterministic mid-box point, so its
endpoint is exactly what a practitioner running single-start gradient descent
would commit. Comparing its pretender rate against the best-of-8 committed
geometries isolates the *selection-pressure* contribution of multi-start
("did picking the deepest surrogate minimum make pretenders worse?").

Reads inverse_{S}_v8.npz (endpoints_u archived by inverse.py), evaluates the
surrogate at restart-0 endpoints, RCWA-validates them, and writes
restart0_{S}_v8.{npz,json}.
"""
from __future__ import annotations
import argparse, importlib, json, time
import numpy as np
import torch

from common import (STRUCTS, DEVICE, RESULTS, make_model, a_channel_keys, seed_sfx,
                    input_sfx, env_info, assert_same_surrogate, load_surrogate)
from oracle import configure_solver, simulate_cached, owned, all_cached

SUCCESS_TAU_PCT = 5.0


def main(s, seed=42, shard=0, nshards=1):
    cfg = STRUCTS[s]
    sx = seed_sfx(seed)                 # output suffix; the inverse run being validated
    ix = input_sfx(seed, "finetune")    # carries the same tag, its surrogate may not
    inv = np.load(RESULTS / f"inverse_{s}{sx}_v8.npz")
    stats = np.load(RESULTS / f"stats_{s}{ix}_v8.npz")
    model, ident = load_surrogate(s, ix)
    a_keys = a_channel_keys(s)
    wl = stats["wavelengths"].astype(np.float64)
    L = len(wl)
    d = cfg["d"]
    n = inv["endpoints_u"].shape[0]

    lo, hi = stats["design_lo"], stats["design_hi"]
    tlo = torch.tensor(stats["train_lo"], dtype=torch.float32, device=DEVICE)
    thi = torch.tensor(stats["train_hi"], dtype=torch.float32, device=DEVICE)
    pm = torch.tensor(stats["phys_mean"], dtype=torch.float32, device=DEVICE)
    ps = torch.tensor(stats["phys_std"], dtype=torch.float32, device=DEVICE)
    wln = torch.tensor((wl - wl.min()) / (wl.max() - wl.min()),
                       dtype=torch.float32, device=DEVICE)

    # Restart-0 geometry.  A feasible inverse run maps u through a NESTED parametrization
    # (inverse.py u_to_phys), so reconstructing x from u with the plain box mapping gives a
    # different geometry -- up to 250 nm off on A (codex review 2026-09-12, slice 2).
    # inverse.py now archives the physical endpoints; use them, and refuse to guess for
    # older feasible artifacts that lack them.
    if "endpoints_phys" in inv.files:
        x0 = inv["endpoints_phys"][:, 0, :].astype(np.float64)
    elif bool(inv["feasible"]) if "feasible" in inv.files else False:
        raise RuntimeError("feasible inverse artifact without endpoints_phys: the box "
                           "mapping would reconstruct the wrong geometries; rerun inverse")
    else:
        u0_end = inv["endpoints_u"][:, 0, :].astype(np.float64)     # restart 0
        x0 = lo + u0_end * (hi - lo)                                # (n, d) physical
    if "surrogate_sha256" in inv.files and str(inv["surrogate_sha256"]) != ident["surrogate_sha256"]:
        raise RuntimeError("restart0_validate: the inverse artifact was produced by a different "
                           "surrogate than the one just loaded; refusing to classify its endpoints")

    def spectra(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
        B = x.shape[0]
        p_norm = (x - tlo) / (thi - tlo)
        feats = torch.stack([cfg["torch_feat"](x, lam, metal="Cr")
                             for lam in wl.tolist()], dim=0)
        feats = (feats - pm) / ps
        xg = torch.cat([wln.view(L, 1, 1).expand(L, B, 1),
                        p_norm.unsqueeze(0).expand(L, B, d)], dim=-1)
        with torch.no_grad():
            out = model(xg.reshape(L * B, -1), p=feats.reshape(L * B, -1))
        return {k: out[k].view(L, B).cpu().numpy() for k in a_keys}

    pred = spectra(x0)
    A_tgt = {k: inv[f"A_target_{k}"] for k in a_keys}
    mae_surr = np.mean([np.mean(np.abs(pred[k].T - A_tgt[k]), axis=1)
                        for k in a_keys], axis=0) * 100

    mod = importlib.import_module(cfg["rcwa_module"])
    prov = configure_solver(mod)
    names = cfg["param_names"]
    kind = "restart0"
    mae_rcwa = np.full(n, np.nan)
    print(f"[restart0 {s}] validating {n} single-start endpoints (shard {shard}/{nshards})", flush=True)
    for i in owned(n, shard, nshards):
        pdict = {k: float(x0[i, j]) for j, k in enumerate(names)}
        r = simulate_cached(kind, s, sx, i, pdict, wl, cfg, mod, targets={k: A_tgt[k][i] for k in a_keys})
        print(f"  [{i+1:2d}/{n}] surr={mae_surr[i]:5.2f}%  "
              f"rcwa={('FAILED ' + r['error']) if r['failed'] else ('%6.2f%%' % r['mae'])} "
              f"N={min(r['orders'])}-{max(r['orders'])} ({r['elapsed']:.0f}s{', cached' if r['cached'] else ''})", flush=True)
    if not all_cached(kind, s, sx, range(n)):
        print(f"[restart0 {s}] other shards still running — summary not written by this worker", flush=True)
        return
    for i in range(n):
        pdict = {k: float(x0[i, j]) for j, k in enumerate(names)}
        mae_rcwa[i] = simulate_cached(kind, s, sx, i, pdict, wl, cfg, mod,
                                      targets={k: A_tgt[k][i] for k in a_keys})["mae"]

    valid = ~np.isnan(mae_rcwa)
    surr_pass = (mae_surr <= SUCCESS_TAU_PCT) & valid
    rcwa_pass = (mae_rcwa <= SUCCESS_TAU_PCT) & valid
    pretender = surr_pass & ~rcwa_pass
    summary = dict(structure=s, variant="single-start (restart-0, mid-box init)",
                   seed=int(seed), env=env_info(), oracle=prov, n=int(valid.sum()),
                   mean_surr=float(mae_surr[valid].mean()),
                   mean_rcwa=float(mae_rcwa[valid].mean()),
                   surrogate_pass=int(surr_pass.sum()),
                   rcwa_pass=int(rcwa_pass.sum()),
                   pretender=int(pretender.sum()),
                   per_sample_surr=mae_surr.tolist(),
                   per_sample_rcwa=mae_rcwa.tolist())
    np.savez(RESULTS / f"restart0_{s}{sx}_v8.npz",
             params=x0, mae_surrogate=mae_surr, mae_rcwa=mae_rcwa,
             wavelengths=wl)
    (RESULTS / f"restart0_{s}{sx}_v8.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== RESTART-0 {s}: surr pass {int(surr_pass.sum())}/{n}, "
          f"oracle pass {int(rcwa_pass.sum())}/{n}, pretender {int(pretender.sum())}/{n}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True, choices=list(STRUCTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    main(a.structure, seed=a.seed, shard=a.shard, nshards=a.nshards)
