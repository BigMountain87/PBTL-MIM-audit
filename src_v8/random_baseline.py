#!/usr/bin/env python3
"""v8 random-search baseline — mechanism control for the pretender finding.

For each of the SAME 20 targets, draw M = 6400 uniform geometries from the
design box (matching the gradient pipeline's surrogate-call budget of
8 restarts × 800 iterations), evaluate the surrogate, commit the argmin,
and RCWA-validate the committed geometry.

Interpretation: if random search shows far fewer pretenders than gradient
descent at equal budget, the mechanism is *active optimizer exploitation* of
surrogate error; if pretender rates are similar, the mechanism is generic
surrogate extrapolation error. Either way the result disambiguates §4 of the
protocol.
"""
from __future__ import annotations
import argparse, importlib, json, os, time
import numpy as np
import torch

from common import (STRUCTS, DEVICE, RESULTS, N_TARGETS, TARGET_SEED, pick_targets,
                    make_model, load_filtered, a_channel_keys, seed_sfx, input_sfx, load_surrogate,
                    feasible_mask, env_info)
from oracle import configure_solver, simulate_cached, owned, all_cached

M_BUDGET = 6400
CHUNK = 256
SUCCESS_TAU_PCT = 5.0


def main(s, seed=42, feasible=False, shard=0, nshards=1):
    cfg = STRUCTS[s]
    sx = seed_sfx(seed)                 # output suffix (seed + INVERSETL_TAG)
    ix = input_sfx(seed, "finetune")    # fine-tune inputs: tagged if present, else seed-only
    ft = json.loads((RESULTS / f"finetune_{s}{ix}_v8.json").read_text())
    stats = np.load(RESULTS / f"stats_{s}{ix}_v8.npz")
    model, ident = load_surrogate(s, ix)         # hash of the bytes actually loaded
    data = load_filtered(s)
    wl = stats["wavelengths"].astype(np.float64)
    L = len(wl)
    d = cfg["d"]
    a_keys = a_channel_keys(s)

    lo, hi = stats["design_lo"], stats["design_hi"]
    tlo = torch.tensor(stats["train_lo"], dtype=torch.float32, device=DEVICE)
    thi = torch.tensor(stats["train_hi"], dtype=torch.float32, device=DEVICE)
    pm = torch.tensor(stats["phys_mean"], dtype=torch.float32, device=DEVICE)
    ps = torch.tensor(stats["phys_std"], dtype=torch.float32, device=DEVICE)
    wln = torch.tensor((wl - wl.min()) / (wl.max() - wl.min()),
                       dtype=torch.float32, device=DEVICE)
    wl_list = wl.tolist()

    test_idx = np.array(ft["test_idx"], dtype=int)
    rng_t = np.random.default_rng(TARGET_SEED)
    targets = pick_targets(test_idx)
    orig_idx_map = np.array(ft["orig_dataset_idx"], dtype=int)
    print(f"[random-baseline {s}] M={M_BUDGET} per target, device={DEVICE}, seed={seed}, "
          f"{'feasible-only (rejection sampling)' if feasible else 'design box'}", flush=True)

    def draw(rng):
        """M_BUDGET uniform draws from the design box; with --feasible, rejection-sample
        until M_BUDGET ACCEPTED candidates (same generator constraints as the datasets)."""
        if not feasible:
            return rng.uniform(lo, hi, (M_BUDGET, d)), 1.0
        acc, n_drawn = [], 0
        while sum(len(a) for a in acc) < M_BUDGET:
            c = rng.uniform(lo, hi, (M_BUDGET, d)); n_drawn += M_BUDGET
            acc.append(c[feasible_mask(s, c)])
        return np.concatenate(acc)[:M_BUDGET], M_BUDGET / n_drawn

    def spectra(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
        B = x.shape[0]
        p_norm = (x - tlo) / (thi - tlo)
        feats = torch.stack([cfg["torch_feat"](x, lam, metal="Cr")
                             for lam in wl_list], dim=0)
        feats = (feats - pm) / ps
        xg = torch.cat([wln.view(L, 1, 1).expand(L, B, 1),
                        p_norm.unsqueeze(0).expand(L, B, d)], dim=-1)
        with torch.no_grad():
            out = model(xg.reshape(L * B, -1), p=feats.reshape(L * B, -1))
        return {k: out[k].view(L, B) for k in a_keys}

    best_params = np.empty((N_TARGETS, d))
    mae_surr = np.empty(N_TARGETS)
    A_tgt = {k: np.empty((N_TARGETS, L)) for k in a_keys}
    A_best = {k: np.empty((N_TARGETS, L)) for k in a_keys}

    for i, gidx in enumerate(targets):
        oidx = int(orig_idx_map[gidx])
        tgt_t = {k: torch.tensor(data["chans"][k][gidx], dtype=torch.float32,
                                 device=DEVICE) for k in a_keys}
        rng = np.random.default_rng(5000 + oidx)
        cand, acc_rate = draw(rng)
        best_loss, best_j, best_spec = np.inf, -1, None
        t0 = time.time()
        for st in range(0, M_BUDGET, CHUNK):
            ch = cand[st:st + CHUNK]
            pred = spectra(ch)
            losses = sum(torch.mean((pred[k] - tgt_t[k].unsqueeze(1)) ** 2, dim=0)
                         for k in a_keys) / len(a_keys)
            j = int(torch.argmin(losses))
            lv = float(losses[j])
            if lv < best_loss:
                best_loss = lv
                best_j = st + j
                best_spec = {k: pred[k][:, j].cpu().numpy() for k in a_keys}
        best_params[i] = cand[best_j]
        maes = []
        for k in a_keys:
            A_best[k][i] = best_spec[k]
            A_tgt[k][i] = data["chans"][k][gidx]
            maes.append(np.mean(np.abs(best_spec[k] - A_tgt[k][i])))
        mae_surr[i] = float(np.mean(maes) * 100)
        print(f"  [{i+1:2d}/{N_TARGETS}] orig idx={oidx:3d}  "
              f"surr MAE={mae_surr[i]:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)

    # RCWA-validate committed geometries (cached per design; shard-able; resume-safe)
    mod = importlib.import_module(cfg["rcwa_module"])
    prov = configure_solver(mod)
    names = cfg["param_names"]
    kind = "random_baseline"
    mae_rcwa = np.full(N_TARGETS, np.nan)
    for i in owned(N_TARGETS, shard, nshards):
        pdict = {k: float(best_params[i, j]) for j, k in enumerate(names)}
        r = simulate_cached(kind, s, sx, i, pdict, wl, cfg, mod, targets={k: A_tgt[k][i] for k in a_keys})
        print(f"  rcwa [{i+1:2d}] {('FAILED ' + r['error']) if r['failed'] else ('%6.2f%%' % r['mae'])} "
              f"N={min(r['orders'])}-{max(r['orders'])} ({r['elapsed']:.0f}s{', cached' if r['cached'] else ''})", flush=True)
    if not all_cached(kind, s, sx, range(N_TARGETS)):
        print(f"[random-baseline {s}] other shards still running — summary not written by this worker", flush=True)
        return
    for i in range(N_TARGETS):
        pdict = {k: float(best_params[i, j]) for j, k in enumerate(names)}
        r = simulate_cached(kind, s, sx, i, pdict, wl, cfg, mod, targets={k: A_tgt[k][i] for k in a_keys})
        mae_rcwa[i] = r["mae"]

    valid = ~np.isnan(mae_rcwa)
    surr_pass = (mae_surr <= SUCCESS_TAU_PCT) & valid
    rcwa_pass = (mae_rcwa <= SUCCESS_TAU_PCT) & valid
    pretender = surr_pass & ~rcwa_pass
    summary = dict(structure=s, budget=M_BUDGET, seed=int(seed), feasible=bool(feasible),
                   env=env_info(), oracle=prov, n=int(valid.sum()),
                   mean_surr=float(mae_surr[valid].mean()),
                   mean_rcwa=float(mae_rcwa[valid].mean()),
                   surrogate_pass=int(surr_pass.sum()),
                   rcwa_pass=int(rcwa_pass.sum()),
                   pretender=int(pretender.sum()),
                   per_sample_surr=mae_surr.tolist(),
                   per_sample_rcwa=mae_rcwa.tolist())
    np.savez(RESULTS / f"random_baseline_{s}{sx}_v8.npz", **ident,
             best_params=best_params, mae_surrogate=mae_surr, mae_rcwa=mae_rcwa,
             targets=targets, wavelengths=wl, feasible=bool(feasible))
    (RESULTS / f"random_baseline_{s}{sx}_v8.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n=== RANDOM BASELINE {s}: surr pass {int(surr_pass.sum())}/20, "
          f"oracle pass {int(rcwa_pass.sum())}/20, pretender {int(pretender.sum())}/20, "
          f"mean surr {mae_surr[valid].mean():.2f}% / rcwa {mae_rcwa[valid].mean():.2f}%",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True, choices=list(STRUCTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feasible", action="store_true",
                    help="rejection-sample candidates from the generator's feasible region")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    main(a.structure, seed=a.seed, feasible=a.feasible, shard=a.shard, nshards=a.nshards)
