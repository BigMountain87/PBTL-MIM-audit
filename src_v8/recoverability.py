#!/usr/bin/env python3
"""T12(2) — recoverability metrics for every run (A/B/C x 42/123/777).

Per committed design: surrogate MAE at the TRUE generating geometry vs the target
(how well the surrogate would have scored the actual answer), the claimed MAE at the
committed geometry, ||u_best - u_true|| and per-coordinate |du| in design-box units,
the fraction of targets whose committed surrogate MAE undercuts the surrogate MAE at
truth, u-space distances among oracle-confirmed designs, a support-violation flag
(A/C: any of Wx, Wy(, W2) >= P) and its overlap with the pretender set, and from the
reference-solver spectra the peak-wavelength shift and peak-absorptance error
(C per channel).  Writes RESULTS/recoverability_v8.json.  Run with INVERSETL_DEVICE=cpu.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch

from common import STRUCTS, RESULTS, DEVICE, make_model, a_channel_keys, PROFILE

SEEDS = {"": 42, "_s123": 123, "_s777": 777}
TAU = 5.0


def spectra_fn(s, cfg, model, stats, wl):
    L, d = len(wl), cfg["d"]
    tlo = torch.tensor(stats["train_lo"], dtype=torch.float32, device=DEVICE)
    thi = torch.tensor(stats["train_hi"], dtype=torch.float32, device=DEVICE)
    pm = torch.tensor(stats["phys_mean"], dtype=torch.float32, device=DEVICE)
    ps = torch.tensor(stats["phys_std"], dtype=torch.float32, device=DEVICE)
    wln = torch.tensor((wl - wl.min()) / (wl.max() - wl.min()), dtype=torch.float32, device=DEVICE)
    keys = a_channel_keys(s)

    def f(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE); B = x.shape[0]
        p_norm = (x - tlo) / (thi - tlo)
        feats = torch.stack([cfg["torch_feat"](x, lam, metal="Cr") for lam in wl.tolist()], dim=0)
        feats = (feats - pm) / ps
        xg = torch.cat([wln.view(L, 1, 1).expand(L, B, 1), p_norm.unsqueeze(0).expand(L, B, d)], dim=-1)
        with torch.no_grad():
            out = model(xg.reshape(L * B, -1), p=feats.reshape(L * B, -1))
        return {k: out[k].view(L, B).cpu().numpy().T for k in keys}   # (B, L)
    return f


def peak_stats(A_pred, A_tgt, wl):
    """peak-wavelength shift (nm) and peak-absorptance error for (n, L) arrays; NaN rows -> NaN."""
    shift, aerr = np.full(len(A_pred), np.nan), np.full(len(A_pred), np.nan)
    for i in range(len(A_pred)):
        if np.all(np.isfinite(A_pred[i])):
            shift[i] = wl[int(np.argmax(A_pred[i]))] - wl[int(np.argmax(A_tgt[i]))]
            aerr[i] = float(A_pred[i].max() - A_tgt[i].max())
    return shift, aerr


def main():
    out = dict(profile=PROFILE, tau_pct=TAU, runs={})
    for s in "ABC":
        cfg = STRUCTS[s]; keys = a_channel_keys(s); names = list(cfg["param_names"])
        for sx, seed in SEEDS.items():
            inv = np.load(RESULTS / f"inverse_{s}{sx}_v8.npz", allow_pickle=True)
            rc = np.load(RESULTS / f"rcwa_{s}{sx}_v8.npz", allow_pickle=True)
            rel = json.load(open(RESULTS / f"reliability_{s}{sx}_v8.json"))
            stats = np.load(RESULTS / f"stats_{s}{sx}_v8.npz")
            model = make_model(s)
            model.load_state_dict(torch.load(RESULTS / f"surrogate_{s}{sx}_v8.pt", map_location=DEVICE, weights_only=True)); model.eval()
            wl = stats["wavelengths"].astype(np.float64)
            f = spectra_fn(s, cfg, model, stats, wl)
            tgt = {k: inv[f"A_target_{k}"] for k in keys}
            pred_true, pred_best = f(inv["true_params"]), f(inv["best_params"])
            mae_at_truth = np.mean([np.mean(np.abs(pred_true[k] - tgt[k]), axis=1) for k in keys], axis=0) * 100
            mae_at_best = np.mean([np.mean(np.abs(pred_best[k] - tgt[k]), axis=1) for k in keys], axis=0) * 100
            assert np.allclose(mae_at_best, inv["mae_surrogate"], atol=0.05), (s, sx, np.abs(mae_at_best - inv["mae_surrogate"]).max())
            lo, hi = inv["design_lo"], inv["design_hi"]
            u_true = (inv["true_params"] - lo) / (hi - lo); u_best = (inv["best_params"] - lo) / (hi - lo)
            du = np.abs(u_best - u_true); dist = np.linalg.norm(u_best - u_true, axis=1)
            mae_rcwa = rc["mae_rcwa"].astype(float); valid = (~rc["failed"].astype(bool)) & np.isfinite(mae_rcwa)
            ms = inv["mae_surrogate"].astype(float)
            pret = (ms <= TAU) & (mae_rcwa > TAU) & valid; conf = (ms <= TAU) & (mae_rcwa <= TAU) & valid
            assert sorted(np.where(pret)[0].tolist()) == sorted(rel["tier1a"]["pretender_indices"]), (s, sx)
            # support violation (A/C): a patch wider than the period is outside the physical support of the generator
            if s in ("A", "C"):
                P = inv["best_params"][:, names.index("P")]
                sv = np.zeros(len(P), bool)
                for w in (("Wx", "Wy", "W2") if s == "A" else ("Wx", "Wy")):
                    sv |= inv["best_params"][:, names.index(w)] >= P
            else:
                sv = inv["best_params"][:, names.index("R_in")] >= inv["best_params"][:, names.index("R_out")]
            # peak statistics from the reference solver
            peaks = {}
            for k in keys:
                sh, ae = peak_stats(rc[f"A_rcwa_{k}"], tgt[k], wl)
                peaks[k] = dict(peak_wavelength_shift_nm=sh.tolist(), peak_absorptance_error=ae.tolist(),
                                abs_shift_median_nm=float(np.nanmedian(np.abs(sh))),
                                abs_shift_median_pretenders_nm=float(np.nanmedian(np.abs(sh[pret]))) if pret.any() else None,
                                abs_shift_median_confirmed_nm=float(np.nanmedian(np.abs(sh[conf]))) if conf.any() else None)
            ci = np.where(conf)[0]
            if len(ci) >= 2:
                D = np.linalg.norm(u_best[ci][:, None] - u_best[ci][None, :], axis=-1)
                conf_dist = dict(n=int(len(ci)), mean_pair=float(D[np.triu_indices(len(ci), 1)].mean()),
                                 min_nn=float((D + np.eye(len(ci)) * 1e9).min(axis=1).mean()))
            else:
                conf_dist = dict(n=int(len(ci)))
            out["runs"][f"{s}{sx}"] = dict(
                structure=s, seed=seed, n=int(len(ms)), n_valid=int(valid.sum()),
                surr_mae_at_truth_pct=mae_at_truth.tolist(), surr_mae_at_best_pct=mae_at_best.tolist(),
                surr_mae_at_truth_median_pct=float(np.median(mae_at_truth)),
                n_truth_le_tau=int((mae_at_truth <= TAU).sum()),
                frac_claimed_below_truth=float((mae_at_best < mae_at_truth).mean()),
                u_dist_best_true=dist.tolist(), u_dist_median=float(np.median(dist)),
                du_per_coord_median={n: float(np.median(du[:, i])) for i, n in enumerate(names)},
                u_dist_median_pretenders=float(np.median(dist[pret])) if pret.any() else None,
                u_dist_median_confirmed=float(np.median(dist[conf])) if conf.any() else None,
                confirmed_u_distances=conf_dist,
                support_violation_flag=sv.tolist(), n_support_violation=int(sv[valid].sum()),
                support_violation_among_pretenders=[int((sv & pret).sum()), int(pret.sum())],
                support_violation_among_confirmed=[int((sv & conf).sum()), int(conf.sum())],
                pretender_indices=np.where(pret)[0].tolist(), confirmed_indices=np.where(conf)[0].tolist(),
                peaks=peaks)
            r = out["runs"][f"{s}{sx}"]
            print(f"{s}{sx:6s} truth-MAE median {r['surr_mae_at_truth_median_pct']:.2f}% (<=tau {r['n_truth_le_tau']}/20)  "
                  f"claimed<truth {r['frac_claimed_below_truth']:.2f}  |du| median {r['u_dist_median']:.2f}  "
                  f"support-viol pret {r['support_violation_among_pretenders']} conf {r['support_violation_among_confirmed']}", flush=True)
    (RESULTS / "recoverability_v8.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS/'recoverability_v8.json'}")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); main()
