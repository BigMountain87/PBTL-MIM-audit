#!/usr/bin/env python3
"""v8 unified reliability analysis — protocol §4 metrics + §5 statistics.

Differences from v7: T4 is actually measured (K=100 Gaussian perturbations);
the primary discrimination statistic is ρ(RCWA MAE, Surr MAE) with exact p and
a Bonett–Wright Fisher-z CI; ρ(Δ,·) is reported as secondary with the
mathematical-coupling caveat; Tier-1C reports multi-start dispersion.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
from scipy import stats as sps

from common import (STRUCTS, RESULTS, DEVICE, make_model, a_channel_keys, seed_sfx,
                    input_sfx, env_info, assert_same_surrogate, load_surrogate)

SUCCESS_TAU_PCT = 5.0
FLAG_K = 2.0
T1_FACTOR = 3.0
T2_EPS = 0.05
T3_FACTOR = 0.5
T4_K = 100
T4_SIGMA_FRAC = 0.01
T4_THRESHOLD_PCT = 5.0
T4_SEED = 777


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, c - m), min(1.0, c + m))


def spearman_with_ci(a, b):
    rho, p = sps.spearmanr(a, b)
    n = len(a)
    if n > 3 and abs(rho) < 1:
        se = np.sqrt((1 + rho ** 2 / 2) / (n - 3))   # Bonett–Wright
        z = np.arctanh(rho)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    else:
        lo = hi = float("nan")
    return dict(rho=float(rho), p=float(p), ci95=[float(lo), float(hi)], n=int(n))


def t2_baseline(d, n_random=10_000, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, (n_random, d))
    near = (u < T2_EPS) | (u > 1 - T2_EPS)
    return dict(empirical=float(near.any(axis=1).mean()),
                analytical=float(1 - (1 - 2 * T2_EPS) ** d), d=d)


def t3_baseline(n_samples, d, n_batches=1000, seed=0):
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_batches):
        x = rng.uniform(0, 1, (n_samples, d))
        dist = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        med = np.median(dist[np.triu_indices(n_samples, 1)])
        nn = (dist + np.eye(n_samples) * 1e9).min(axis=1)
        rates.append((nn < T3_FACTOR * med).mean())
    return dict(mean=float(np.mean(rates)),
                p2_5=float(np.percentile(rates, 2.5)),
                p97_5=float(np.percentile(rates, 97.5)))


def mode_count(u, thr):
    """Connected components under distance < thr (the T3 threshold)."""
    n = len(u)
    dist = np.linalg.norm(u[:, None] - u[None, :], axis=-1)
    seen, comps = np.zeros(n, bool), 0
    for i in range(n):
        if seen[i]:
            continue
        comps += 1
        stack = [i]
        while stack:
            j = stack.pop()
            if seen[j]:
                continue
            seen[j] = True
            stack.extend(np.where((dist[j] < thr) & ~seen)[0].tolist())
    return comps


def measure_t4(s, best_params, stats, a_keys, sx=""):
    """K Gaussian perturbations (σ = 1% box width, clipped to box); per-sample
    mean surrogate-spectrum MAE vs the unperturbed surrogate spectrum."""
    cfg = STRUCTS[s]
    model, _ = load_surrogate(s, sx)
    lo, hi = stats["design_lo"], stats["design_hi"]
    tlo = torch.tensor(stats["train_lo"], dtype=torch.float32, device=DEVICE)
    thi = torch.tensor(stats["train_hi"], dtype=torch.float32, device=DEVICE)
    pm = torch.tensor(stats["phys_mean"], dtype=torch.float32, device=DEVICE)
    ps = torch.tensor(stats["phys_std"], dtype=torch.float32, device=DEVICE)
    wl = stats["wavelengths"].astype(np.float64)
    L = len(wl)
    wln = torch.tensor((wl - wl.min()) / (wl.max() - wl.min()),
                       dtype=torch.float32, device=DEVICE)
    sigma = T4_SIGMA_FRAC * (hi - lo)
    rng = np.random.default_rng(T4_SEED)
    t4_mean = np.empty(len(best_params))

    def spectra(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)   # (B,d)
        p_norm = (x - tlo) / (thi - tlo)
        feats = torch.stack([cfg["torch_feat"](x, lam, metal="Cr")
                             for lam in wl.tolist()], dim=0)          # (L,B,F)
        feats = (feats - pm) / ps
        B = x.shape[0]
        xg = torch.cat([wln.view(L, 1, 1).expand(L, B, 1),
                        p_norm.unsqueeze(0).expand(L, B, cfg["d"])], dim=-1)
        with torch.no_grad():
            out = model(xg.reshape(L * B, -1), p=feats.reshape(L * B, -1))
        return {k: out[k].view(L, B).cpu().numpy() for k in a_keys}

    for i, x0 in enumerate(best_params):
        base = spectra(x0[None, :])
        pert = np.clip(x0[None, :] + rng.normal(0, 1, (T4_K, len(x0))) * sigma,
                       lo, hi)
        per = spectra(pert)
        maes = [np.mean(np.abs(per[k] - base[k][:, :1]), axis=0) for k in a_keys]
        t4_mean[i] = float(np.mean(maes) * 100)
    return t4_mean


def main(s, seed=42):
    cfg = STRUCTS[s]
    sx = seed_sfx(seed)                       # this run's own outputs (inverse, rcwa, results)
    # The fine-tune artifacts belong to the surrogate the run used, which for a tagged
    # condition that reuses the base surrogate -- the feasibility rerun does -- is the
    # untagged one.  inverse.py resolves them the same way; reading them with seed_sfx
    # here is what made every feasible reliability run fail on a missing finetune log.
    fx = input_sfx(seed, "finetune")
    inv = np.load(RESULTS / f"inverse_{s}{sx}_v8.npz")
    rcwa = np.load(RESULTS / f"rcwa_{s}{sx}_v8.npz")
    # the flag threshold and T4 come from the surrogate this run resolves; it must be the
    # one that produced the designs, or the reliability file describes a different model
    if "surrogate_sha256" in inv.files:
        assert_same_surrogate(inv["surrogate_sha256"], s, fx, "reliability")
    # the inverse and oracle artifacts being joined must describe the same run: same
    # surrogate (when both record it) and the same committed geometries
    if "surrogate_sha256" in inv.files and "surrogate_sha256" in rcwa.files \
            and str(inv["surrogate_sha256"]) != str(rcwa["surrogate_sha256"]):
        raise RuntimeError("reliability: inverse and rcwa artifacts record different surrogates")
    if not np.allclose(np.asarray(inv["best_params"], float), np.asarray(rcwa["best_params"], float),
                       rtol=0, atol=1e-6):
        raise RuntimeError("reliability: inverse and rcwa artifacts carry different committed "
                           "geometries; they are not from the same run")
    ft = json.loads((RESULTS / f"finetune_{s}{fx}_v8.json").read_text())
    stats = np.load(RESULTS / f"stats_{s}{fx}_v8.npz")
    a_keys = a_channel_keys(s)

    mae_surr = inv["mae_surrogate"].astype(float)
    mae_rcwa = rcwa["mae_rcwa"].astype(float)
    valid = ~rcwa["failed"].astype(bool)
    n, nv = len(mae_surr), int(valid.sum())
    test_mae = float(ft["final_test_mae_pct"])

    surr_pass = (mae_surr <= SUCCESS_TAU_PCT) & valid
    rcwa_pass = (mae_rcwa <= SUCCESS_TAU_PCT) & valid
    pretender = surr_pass & ~rcwa_pass
    delta = mae_rcwa - mae_surr
    flag_thr = FLAG_K * test_mae
    flagged = (delta > flag_thr) & valid
    amp = np.where(valid, mae_rcwa / np.maximum(mae_surr, 1e-6), np.nan)

    # taxonomy on committed geometries (u-space of the DESIGN box)
    lo, hi = stats["design_lo"], stats["design_hi"]
    u = (inv["best_params"] - lo) / (hi - lo)
    t1 = (mae_rcwa > T1_FACTOR * mae_surr) & valid
    t2 = ((u < T2_EPS) | (u > 1 - T2_EPS)).any(axis=1)
    dist = np.linalg.norm(u[:, None] - u[None, :], axis=-1)
    med = float(np.median(dist[np.triu_indices(n, 1)]))
    nn_dist = (dist + np.eye(n) * 1e9).min(axis=1)
    t3 = nn_dist < T3_FACTOR * med
    t4_mean = measure_t4(s, inv["best_params"], stats, a_keys, sx=fx)
    t4 = t4_mean > T4_THRESHOLD_PCT

    # Tier 1C — multi-start dispersion + loss-tail RSD of committed restart
    fl = inv["final_losses"]                      # (n, R)
    restart_loss_std = fl.std(axis=1)
    eu = inv["endpoints_u"]                       # (n, R, d)
    ep_spread = np.array([np.linalg.norm(e[:, None] - e[None, :], axis=-1).max()
                          for e in eu])
    lh = inv["loss_hist"]                         # (n, iters, R)
    br = inv["best_restart"].astype(int)
    tail = lh[:, int(lh.shape[1] * 0.8):, :]
    tail_rsd = np.array([float(tail[i, :, br[i]].std()
                               / max(tail[i, :, br[i]].mean(), 1e-12))
                         for i in range(n)])

    # discrimination (protocol §5): primary = rho(RCWA, Surr)
    m = valid
    disc = dict(
        rcwa_vs_surr_PRIMARY=spearman_with_ci(mae_rcwa[m], mae_surr[m]),
        delta_vs_rcwa_secondary=spearman_with_ci(delta[m], mae_rcwa[m]),
        delta_vs_surr_secondary_COUPLED=spearman_with_ci(delta[m], mae_surr[m]),
        pearson_rcwa_vs_surr=float(sps.pearsonr(mae_rcwa[m], mae_surr[m])[0]),
        coupling_caveat=("delta = rcwa - surr contains -surr by construction; "
                         "rho(delta, surr) is negatively biased and must not "
                         "be headlined (v4a lesson)"),
    )

    out = dict(
        meta=dict(structure=s, version="v8", n=n, n_valid=nv,
                  forward_test_mae_pct=test_mae,
                  n_train=ft["n_train"], wavelengths_nm=[float(stats["wavelengths"].min()),
                                                         float(stats["wavelengths"].max())],
                  n_restarts=int(inv["n_restarts"]), env=env_info()),
        tier1a=dict(tau_pct=SUCCESS_TAU_PCT,
                    surrogate_pass=int(surr_pass.sum()),
                    rcwa_pass=int(rcwa_pass.sum()),
                    pretender=int(pretender.sum()),
                    pretender_indices=pretender.nonzero()[0].tolist(),
                    surrogate_pass_ci=wilson_ci(int(surr_pass.sum()), nv),
                    rcwa_pass_ci=wilson_ci(int(rcwa_pass.sum()), nv),
                    pretender_ci=wilson_ci(int(pretender.sum()), nv)),
        tier1b=dict(flag_threshold_pct=flag_thr, k=FLAG_K,
                    flagged=int(flagged.sum()),
                    flagged_indices=flagged.nonzero()[0].tolist(),
                    flagged_ci=wilson_ci(int(flagged.sum()), nv),
                    max_delta_pct=float(np.nanmax(delta[m])) if nv else None,
                    argmax_delta=int(np.nanargmax(np.where(m, delta, -np.inf))),
                    max_amplification=float(np.nanmax(amp[m])) if nv else None,
                    amplification_caveat="ratio metric; small surrogate-MAE "
                                         "denominators inflate it"),
        tier1c=dict(restart_loss_std_mean=float(restart_loss_std.mean()),
                    endpoint_spread_u_mean=float(ep_spread.mean()),
                    endpoint_spread_u_per_sample=ep_spread.tolist(),
                    tail_rsd_mean=float(tail_rsd.mean())),
        taxonomy=dict(t1=int(t1.sum()), t2=int(t2.sum()), t3=int(t3.sum()),
                      t4=int(t4.sum()),
                      t4_measured=True, t4_mean_pert_mae_pct=t4_mean.tolist(),
                      t2_null=t2_baseline(cfg["d"]),
                      t3_null=t3_baseline(n, cfg["d"]),
                      mode_count=mode_count(u, T3_FACTOR * med),
                      median_pair_dist_u=med,
                      per_sample=dict(t1=t1.tolist(), t2=t2.tolist(),
                                      t3=t3.tolist(), t4=t4.tolist())),
        discrimination=disc,
        per_sample=dict(orig_indices=inv["orig_indices"].tolist(),
                        mae_surr_pct=mae_surr.tolist(),
                        mae_rcwa_pct=mae_rcwa.tolist(),
                        delta_pct=delta.tolist(),
                        amplification=amp.tolist()),
    )
    out["meta"]["seed"] = int(seed)
    path = RESULTS / f"reliability_{s}{sx}_v8.json"
    path.write_text(json.dumps(out, indent=2))

    # CSV table for the manuscript
    rows = ["sample,orig_idx,surr_mae,rcwa_mae,delta,amp,flagged,pretender,"
            "t1,t2,t3,t4,restart_loss_std,endpoint_spread_u"]
    for i in range(n):
        rows.append(f"{i+1},{int(inv['orig_indices'][i])},{mae_surr[i]:.2f},"
                    f"{mae_rcwa[i]:.2f},{delta[i]:.2f},{amp[i]:.2f},"
                    f"{int(flagged[i])},{int(pretender[i])},{int(t1[i])},"
                    f"{int(t2[i])},{int(t3[i])},{int(t4[i])},"
                    f"{restart_loss_std[i]:.2e},{ep_spread[i]:.3f}")
    (RESULTS / f"table_{s}{sx}_v8.csv").write_text("\n".join(rows))

    print(f"\n=== RELIABILITY {s} v8 (n_valid={nv}/{n}) ===")
    print(f"Tier1A τ=5%: surr {int(surr_pass.sum())}, oracle {int(rcwa_pass.sum())}, "
          f"pretender {int(pretender.sum())} {out['tier1a']['pretender_indices']}")
    if nv:
        print(f"Tier1B Δ>{flag_thr:.2f}%: flagged {int(flagged.sum())}, "
              f"max Δ={out['tier1b']['max_delta_pct']:.2f}%, "
              f"max amp={out['tier1b']['max_amplification']:.2f}x")
    print(f"Taxonomy: T1={int(t1.sum())} T2={int(t2.sum())} "
          f"(null {out['taxonomy']['t2_null']['analytical']*100:.0f}%) "
          f"T3={int(t3.sum())} (null {out['taxonomy']['t3_null']['mean']*100:.0f}%) "
          f"T4={int(t4.sum())} [measured]  modes={out['taxonomy']['mode_count']}")
    pr = disc["rcwa_vs_surr_PRIMARY"]
    print(f"PRIMARY rho(RCWA,Surr)={pr['rho']:+.3f} (p={pr['p']:.3f}, "
          f"CI {pr['ci95'][0]:+.2f}..{pr['ci95'][1]:+.2f})")
    print(f"Wrote {path.name} + table_{s}{sx}_v8.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True, choices=list(STRUCTS))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.structure, seed=a.seed)
