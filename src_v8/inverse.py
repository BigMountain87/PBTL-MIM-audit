#!/usr/bin/env python3
"""v8 unified inverse design — normalized variables + multi-start.

Fixes the v4a/v7 frozen-optimizer bug: optimization runs on u ∈ [0,1]^d
(clamped each step) mapped onto the design box, so Adam step sizes are
commensurate with the box in every coordinate. R=8 restarts per target
(1 mid-box + 7 uniform from rng(1000+orig_idx)) run as one batched tensor;
the committed geometry is the restart with the lowest final surrogate loss.
All restart endpoints and loss histories are archived (Tier-1C diagnostics).
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch

from common import (STRUCTS, DEVICE, RESULTS, N_TARGETS, N_RESTARTS, pick_targets, load_surrogate,
                    TARGET_SEED, make_model, load_filtered, seed_sfx, input_sfx,
                    env_info, feasible_mask)

PHASE1_ITERS, PHASE2_ITERS = 500, 300
LR1, LR2 = 0.05, 0.02


def load_artifacts(s, seed=42):
    """Fine-tune artifacts for this seed: tagged if the fine-tune ran under the current
    INVERSETL_TAG, else the untagged seed-only files (see common.input_sfx)."""
    sx = input_sfx(seed, "finetune")
    log = json.loads((RESULTS / f"finetune_{s}{sx}_v8.json").read_text())
    stats = np.load(RESULTS / f"stats_{s}{sx}_v8.npz")
    model, ident = load_surrogate(s, sx)          # identity of the bytes actually loaded
    return model, log, stats, ident


def main(s, seed=42, feasible=False):
    cfg = STRUCTS[s]
    sx = seed_sfx(seed)
    model, ft_log, stats, ident = load_artifacts(s, seed)
    data = load_filtered(s)
    wl = stats["wavelengths"].astype(np.float64)
    L = len(wl)
    d = cfg["d"]

    design_lo = torch.tensor(stats["design_lo"], dtype=torch.float32, device=DEVICE)
    design_hi = torch.tensor(stats["design_hi"], dtype=torch.float32, device=DEVICE)
    train_lo = torch.tensor(stats["train_lo"], dtype=torch.float32, device=DEVICE)
    train_hi = torch.tensor(stats["train_hi"], dtype=torch.float32, device=DEVICE)
    pm = torch.tensor(stats["phys_mean"], dtype=torch.float32, device=DEVICE)
    ps = torch.tensor(stats["phys_std"], dtype=torch.float32, device=DEVICE)
    wln = torch.tensor((wl - wl.min()) / (wl.max() - wl.min()),
                       dtype=torch.float32, device=DEVICE)
    wl_list = wl.tolist()
    torch_feat = cfg["torch_feat"]
    a_keys = ("A_TE", "A_TM") if cfg["dual"] else ("A",)

    test_idx = np.array(ft_log["test_idx"], dtype=int)
    rng = np.random.default_rng(TARGET_SEED)
    targets = pick_targets(test_idx)
    orig_idx_map = np.array(ft_log["orig_dataset_idx"], dtype=int)
    print(f"[inverse {s} v8] device={DEVICE}  N={N_TARGETS} targets × "
          f"{N_RESTARTS} restarts  wl {wl.min():.0f}-{wl.max():.0f} nm  "
          f"parametrization={'feasible (nested box)' if feasible else 'design box'}", flush=True)
    names = list(cfg["param_names"])
    ci = {n: i for i, n in enumerate(names)}

    def u_to_phys(u):
        """u (R,d) in [0,1]^d -> physical geometry (R,d).

        Default: the pre-registered design box  x = lo + u (hi - lo).
        --feasible (T04/T09): a nested reparametrization whose image is exactly the
        generator's feasible region, so every u yields a manufacturable geometry:
          A/C  W_i   = lo_W + u_i (min(0.9 P, hi_W) - lo_W)          for Wx, Wy (A: also W2)
          D    L     = lo + u (min(0.9 P, hi) - lo);  w = lo + u (min(L, hi) - lo)
          B    R_out = lo + u (min(0.45 P, hi) - lo)
               R_in  = lo + u (min(R_out - 10, hi) - lo)
               R_disk= lo + u (min(R_in - 10, hi) - lo)
        with lo/hi the design-box bounds of that parameter (stats design_lo/hi)."""
        x = design_lo + u * (design_hi - design_lo)
        if not feasible:
            return x
        cols = list(x.unbind(dim=1))
        P = cols[ci["P"]]
        def nested(name, cap):
            i = ci[name]
            hi_eff = torch.minimum(cap, design_hi[i])
            cols[i] = design_lo[i] + u[:, i] * (hi_eff - design_lo[i])
            return cols[i]
        if s == "D":
            L = nested("L", 0.9 * P)
            nested("w", L)
        elif s == "B":
            r_out = nested("R_out", 0.45 * P)
            r_in = nested("R_in", r_out - 10.0)
            nested("R_disk", r_in - 10.0)
        else:
            for w in (("Wx", "Wy", "W2") if s == "A" else ("Wx", "Wy")):
                nested(w, 0.9 * P)
        return torch.stack(cols, dim=1)

    def predict(u):
        """u (R,d) in [0,1] → dict of A-channel predictions, each (L,R)."""
        x_phys = u_to_phys(u)
        p_norm = (x_phys - train_lo) / (train_hi - train_lo)          # (R,d)
        feats = torch.stack([torch_feat(x_phys, lam, metal="Cr")
                             for lam in wl_list], dim=0)               # (L,R,F)
        feats = (feats - pm) / ps
        R = u.shape[0]
        xg = torch.cat([wln.view(L, 1, 1).expand(L, R, 1),
                        p_norm.unsqueeze(0).expand(L, R, d)], dim=-1)  # (L,R,1+d)
        out = model(xg.reshape(L * R, 1 + d), p=feats.reshape(L * R, -1))
        return {k: out[k].view(L, R) for k in a_keys}

    def loss_per_restart(pred, tgt):
        """tgt: dict key->(L,) tensor.  Returns (R,) MSE averaged over channels."""
        terms = [torch.mean((pred[k] - tgt[k].unsqueeze(1)) ** 2, dim=0)
                 for k in a_keys]
        return sum(terms) / len(terms)

    n_iters = PHASE1_ITERS + PHASE2_ITERS
    best_params = np.empty((N_TARGETS, d))
    best_restart = np.empty(N_TARGETS, dtype=int)
    endpoints_u = np.empty((N_TARGETS, N_RESTARTS, d), dtype=np.float32)
    endpoints_phys = np.empty((N_TARGETS, N_RESTARTS, d), dtype=np.float64)   # under THIS run's parametrization
    final_losses = np.empty((N_TARGETS, N_RESTARTS))
    loss_hist = np.empty((N_TARGETS, n_iters, N_RESTARTS), dtype=np.float32)
    A_surr = {k: np.empty((N_TARGETS, L)) for k in a_keys}
    A_tgt = {k: np.empty((N_TARGETS, L)) for k in a_keys}
    mae_surr = np.empty(N_TARGETS)
    true_params = np.empty((N_TARGETS, d))

    t_all = time.time()
    for i, gidx in enumerate(targets):
        oidx = int(orig_idx_map[gidx])
        tgt = {k: torch.tensor(data["chans"][k][gidx], dtype=torch.float32,
                               device=DEVICE) for k in a_keys}
        r_rng = np.random.default_rng(1000 + oidx)
        u0 = np.vstack([np.full((1, d), 0.5),
                        r_rng.uniform(0, 1, (N_RESTARTS - 1, d))]).astype(np.float32)
        u = torch.tensor(u0, device=DEVICE, requires_grad=True)
        opt = torch.optim.Adam([u], lr=LR1)
        t0 = time.time()
        for it in range(n_iters):
            if it == PHASE1_ITERS:
                for g in opt.param_groups:
                    g["lr"] = LR2
            opt.zero_grad()
            lpr = loss_per_restart(predict(u), tgt)
            lpr.sum().backward()
            opt.step()
            with torch.no_grad():
                u.clamp_(0.0, 1.0)
            loss_hist[i, it] = lpr.detach().cpu().numpy()
        with torch.no_grad():
            lpr = loss_per_restart(predict(u), tgt).cpu().numpy()
            r_best = int(np.argmin(lpr))
            x_phys = u_to_phys(u).cpu().numpy()
            pred = predict(u)
        final_losses[i] = lpr
        best_restart[i] = r_best
        endpoints_u[i] = u.detach().cpu().numpy()
        endpoints_phys[i] = x_phys            # every restart, already mapped through u_to_phys
        best_params[i] = x_phys[r_best]
        true_params[i] = data["params"][gidx]
        maes = []
        for k in a_keys:
            A_surr[k][i] = pred[k][:, r_best].cpu().numpy()
            A_tgt[k][i] = data["chans"][k][gidx]
            maes.append(np.mean(np.abs(A_surr[k][i] - A_tgt[k][i])))
        mae_surr[i] = float(np.mean(maes) * 100)
        spread = np.abs(endpoints_u[i] - u0).max()
        print(f"  [{i+1:2d}/{N_TARGETS}] orig idx={oidx:3d}  "
              f"surr MAE={mae_surr[i]:5.2f}%  best r={r_best}  "
              f"loss range [{lpr.min():.2e},{lpr.max():.2e}]  "
              f"max|Δu|={spread:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    out = dict(test_indices=targets,
               orig_indices=orig_idx_map[targets],
               true_params=true_params, best_params=best_params,
               best_restart=best_restart, endpoints_u=endpoints_u, endpoints_phys=endpoints_phys,
               final_losses=final_losses, loss_hist=loss_hist,
               mae_surrogate=mae_surr, wavelengths=wl,
               design_lo=stats["design_lo"], design_hi=stats["design_hi"],
               param_names=np.array(cfg["param_names"]),
               structure=s, n_restarts=N_RESTARTS,
               iters=np.array([PHASE1_ITERS, PHASE2_ITERS]),
               lrs=np.array([LR1, LR2]),
               feasible=bool(feasible),
               parametrization="feasible-nested" if feasible else "design-box",
               **ident,
               env=json.dumps(env_info()))
    if feasible:
        assert feasible_mask(s, best_params).all(), "feasible reparametrization violated"
    for k in a_keys:
        out[f"A_surr_{k}"] = A_surr[k]
        out[f"A_target_{k}"] = A_tgt[k]
    path = RESULTS / f"inverse_{s}{sx}_v8.npz"
    np.savez(path, **out)
    print(f"[inverse {s} v8] wrote {path.name}  mean surr MAE "
          f"{mae_surr.mean():.2f}% (range {mae_surr.min():.2f}-{mae_surr.max():.2f}%)  "
          f"total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True, choices=list(STRUCTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feasible", action="store_true",
                    help="optimize over the generator's feasible region (nested "
                         "reparametrization) instead of the raw design box")
    a = ap.parse_args()
    main(a.structure, seed=a.seed, feasible=a.feasible)
