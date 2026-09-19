#!/usr/bin/env python3
"""T07 — surrogate-side detector benchmark: can any oracle-free quantity separate pretenders
(Surr MAE <= tau & RCWA MAE > tau) from oracle-confirmed designs?

Per run (S x seed) and per committed geometry the detectors are:
  ens_std          mean over wavelengths of the std across the three seed surrogates
  heldout_mae      mean MAE vs target of the two NON-committing seeds
  ensmean_mae      MAE vs target of the 3-seed ensemble-mean spectrum
  knn_train        min Euclidean distance (train-bounds-normalized u) to the fine-tune training rows
  knn_pool         same, to the whole filtered dataset pool
  tmm_mae_vs_surr  MAE between the TMM spectrum of the committed geometry and the surrogate's
  tmm_mae_vs_target  MAE between that TMM spectrum and the target
  t4_mean_pert     reliability taxonomy t4_mean_pert_mae_pct (local smoothness)
  t2_flag          box-edge flag (binary)
  endpoint_spread  std of the 8 restart final losses
  mae_surr         the surrogate's own claimed MAE
  infeasible_flag  generator-constraint violation (binary, common.feasible_mask)
AUROC = P(detector higher for a pretender) via Mann-Whitney U/(n1 n0); binary flags use
(sens+spec)/2. Per run, stratified pooled per structure (within-run AUROCs weighted by
n1*n0), naive pooled, 2000-resample target-cluster bootstrap 95 % CI, Spearman(detector, RCWA MAE).
Writes RESULTS/detector_bench_v8.json and detector_bench_v8.md.  Run with INVERSETL_DEVICE=cpu.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
from scipy import stats as sps

from common import (STRUCTS, RESULTS, DEVICE, PROFILE, load_surrogate, a_channel_keys, load_filtered, feasible_mask)

SEEDS = {"": 42, "_s123": 123, "_s777": 777}
TAU = 5.0
BINARY = {"t2_flag", "infeasible_flag"}


def tmm_engine(s):
    if s == "A":
        from src.simulation.tmm_struct_a import compute_tmm_batch
    elif s == "B":
        from src.simulation.tmm_struct_b import compute_tmm_batch
    else:
        from src.simulation.tmm_struct_c_aniso import compute_tmm_batch
    return compute_tmm_batch


def spectra_fn(s, cfg, model, stats, wl):
    L, d = len(wl), cfg["d"]; keys = a_channel_keys(s)
    tlo = torch.tensor(stats["train_lo"], dtype=torch.float32, device=DEVICE); thi = torch.tensor(stats["train_hi"], dtype=torch.float32, device=DEVICE)
    pm = torch.tensor(stats["phys_mean"], dtype=torch.float32, device=DEVICE); ps = torch.tensor(stats["phys_std"], dtype=torch.float32, device=DEVICE)
    wln = torch.tensor((wl - wl.min()) / (wl.max() - wl.min()), dtype=torch.float32, device=DEVICE)

    def f(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE); B = x.shape[0]
        feats = torch.stack([cfg["torch_feat"](x, lam, metal="Cr") for lam in wl.tolist()], dim=0); feats = (feats - pm) / ps
        xg = torch.cat([wln.view(L, 1, 1).expand(L, B, 1), ((x - tlo) / (thi - tlo)).unsqueeze(0).expand(L, B, d)], dim=-1)
        with torch.no_grad():
            out = model(xg.reshape(L * B, -1), p=feats.reshape(L * B, -1))
        return {k: out[k].view(L, B).cpu().numpy().T for k in keys}
    return f


def auroc(x, y, binary=False):
    """x: detector values, y: bool labels (True = pretender). Higher x -> pretender."""
    x, y = np.asarray(x, float), np.asarray(y, bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    if binary:
        sens, spec = (x[y] > 0.5).mean(), (x[~y] <= 0.5).mean()
        return float((sens + spec) / 2)
    return float(sps.mannwhitneyu(x[y], x[~y])[0] / (n1 * n0))


def main():
    out = dict(profile=PROFILE, tau_pct=TAU, detectors=[], runs={}, pooled={})
    rng = np.random.default_rng(0)
    values = {}   # (S, seed) -> dict detector -> array; labels
    for s in "ABC":
        cfg = STRUCTS[s]; keys = a_channel_keys(s); data = load_filtered(s); names = list(cfg["param_names"])
        models = {}
        for sx2, seed2 in SEEDS.items():
            st = np.load(RESULTS / f"stats_{s}{sx2}_v8.npz")
            m, ident = load_surrogate(s, sx2)                # hashed at load; must be the checkpoint that committed the designs
            inv2 = np.load(RESULTS / f"inverse_{s}{sx2}_v8.npz", allow_pickle=True)
            if "surrogate_sha256" in inv2.files and str(inv2["surrogate_sha256"]) != ident["surrogate_sha256"]:
                raise SystemExit(f"{s}{sx2}: surrogate on disk is not the one that committed inverse_{s}{sx2}_v8.npz")
            models[seed2] = (m, st)
        tmm = tmm_engine(s)
        for sx, seed in SEEDS.items():
            inv = np.load(RESULTS / f"inverse_{s}{sx}_v8.npz", allow_pickle=True); rel = json.load(open(RESULTS / f"reliability_{s}{sx}_v8.json"))
            ft = json.load(open(RESULTS / f"finetune_{s}{sx}_v8.json")); st = models[seed][1]
            wl = st["wavelengths"].astype(np.float64); X = inv["best_params"]; tgt = {k: inv[f"A_target_{k}"] for k in keys}
            preds = {sd: spectra_fn(s, cfg, models[sd][0], models[sd][1], wl)(X) for sd in SEEDS.values()}
            stack = {k: np.stack([preds[sd][k] for sd in SEEDS.values()]) for k in keys}           # (3, n, L)
            ens_std = np.mean([stack[k].std(axis=0).mean(axis=1) for k in keys], axis=0) * 100
            others = [sd for sd in SEEDS.values() if sd != seed]
            heldout = np.mean([np.mean([np.mean(np.abs(preds[sd][k] - tgt[k]), axis=1) for k in keys], axis=0) for sd in others], axis=0) * 100
            ensmean = np.mean([np.mean(np.abs(stack[k].mean(axis=0) - tgt[k]), axis=1) for k in keys], axis=0) * 100
            tlo, thi = st["train_lo"], st["train_hi"]; u = (X - tlo) / (thi - tlo)
            train_rows = np.asarray(ft["train_idx"]); pool_u = (data["params"] - tlo) / (thi - tlo)
            knn_train = np.linalg.norm(u[:, None] - pool_u[train_rows][None], axis=-1).min(axis=1)
            knn_pool = np.linalg.norm(u[:, None] - pool_u[None], axis=-1).min(axis=1)
            t = tmm(X.astype(np.float32 if cfg["tmm_params_dtype"] == np.float32 else np.float64), wl, "Cr")
            tmm_spec = {"A": np.clip(t["A_tmm"], 0, 1)} if not cfg["dual"] else {"A_TE": np.clip(t["A_tmm_te"], 0, 1), "A_TM": np.clip(t["A_tmm_tm"], 0, 1)}
            surr_self = preds[seed]
            tmm_vs_surr = np.mean([np.mean(np.abs(tmm_spec[k] - surr_self[k]), axis=1) for k in keys], axis=0) * 100
            tmm_vs_tgt = np.mean([np.mean(np.abs(tmm_spec[k] - tgt[k]), axis=1) for k in keys], axis=0) * 100
            ps = rel["per_sample"]; tx = rel["taxonomy"]
            mae_surr = np.array(ps["mae_surr_pct"], float); mae_rcwa = np.array([np.nan if v is None else v for v in ps["mae_rcwa_pct"]], float)
            valid = np.isfinite(mae_rcwa); pret = (mae_surr <= TAU) & (mae_rcwa > TAU) & valid; severe = (mae_rcwa > 10) & valid
            det = dict(ens_std=ens_std, heldout_mae=heldout, ensmean_mae=ensmean, knn_train=knn_train, knn_pool=knn_pool,
                       tmm_mae_vs_surr=tmm_vs_surr, tmm_mae_vs_target=tmm_vs_tgt,
                       t4_mean_pert=np.array([np.nan if v is None else v for v in tx["t4_mean_pert_mae_pct"]], float),
                       t2_flag=np.array(tx["per_sample"]["t2"], float), endpoint_spread=inv["final_losses"].std(axis=1),
                       mae_surr=mae_surr, infeasible_flag=(~feasible_mask(s, X)).astype(float))
            values[(s, seed)] = dict(det={k: v[valid] for k, v in det.items()}, pret=pret[valid], severe=severe[valid], rcwa=mae_rcwa[valid],
                                     valid=valid, target=np.arange(len(valid))[valid])   # target position: the cluster id shared across seeds
            out["detectors"] = list(det)
            run = {}
            for k, v in values[(s, seed)]["det"].items():
                run[k] = dict(auroc=auroc(v, values[(s, seed)]["pret"], k in BINARY), auroc_severe=auroc(v, values[(s, seed)]["severe"], k in BINARY),
                              spearman_vs_rcwa_mae=float(sps.spearmanr(v, values[(s, seed)]["rcwa"])[0]) if np.std(v) > 0 else None,
                              n_pretender=int(values[(s, seed)]["pret"].sum()), n_confirmed=int((~values[(s, seed)]["pret"]).sum()))
            run["_values"] = {k: [None if not np.isfinite(x) else float(x) for x in v] for k, v in values[(s, seed)]["det"].items()}
            run["_labels"] = dict(pretender=values[(s, seed)]["pret"].tolist(), severe=values[(s, seed)]["severe"].tolist(),
                                  mae_rcwa_pct=values[(s, seed)]["rcwa"].tolist())
            out["runs"][f"{s}_{seed}"] = run
            # a pub-arm run can have zero pretenders (A seed 123), which leaves AUROC undefined
            print(f"{s} {seed} ({int(values[(s, seed)]['pret'].sum())} pretenders): "
                  + "  ".join(f"{k}=" + ("n/a" if run[k]["auroc"] is None else f"{run[k]['auroc']:.2f}")
                              for k in ("ens_std", "heldout_mae", "knn_train", "tmm_mae_vs_surr", "infeasible_flag")), flush=True)
    # ---- pooled per structure: stratified (weighted by n1*n0), naive, bootstrap CI
    for s in "ABC":
        out["pooled"][s] = {}
        for k in out["detectors"]:
            w, a = [], []
            for seed in SEEDS.values():
                r = values[(s, seed)]; n1, n0 = r["pret"].sum(), (~r["pret"]).sum(); au = out["runs"][f"{s}_{seed}"][k]["auroc"]
                if au is not None:
                    w.append(n1 * n0); a.append(au)
            strat = float(np.average(a, weights=w)) if a else None
            xs = np.concatenate([values[(s, sd)]["det"][k] for sd in SEEDS.values()]); ys = np.concatenate([values[(s, sd)]["pret"] for sd in SEEDS.values()])
            naive = auroc(xs, ys, k in BINARY)
            # Cluster bootstrap over TARGETS: the three seeds of a structure share the same 20
            # targets (make_split fixes the held-out set), so a target's three appearances travel
            # together in every replicate.  Resampling each seed independently treated repeated
            # appearances as independent evidence (codex full-scope review, slice 3, 2026-09-12).
            n_targets = len(values[(s, SEEDS[""])]["valid"])
            boots = []
            for _ in range(2000):
                tsel = rng.integers(0, n_targets, n_targets)
                ww, aa = [], []
                for seed in SEEDS.values():
                    r = values[(s, seed)]
                    pos = {t_: i for i, t_ in enumerate(r["target"])}          # skip a target that failed in this seed
                    idx = np.array([pos[t_] for t_ in tsel if t_ in pos], int)
                    if idx.size == 0:
                        continue
                    au = auroc(r["det"][k][idx], r["pret"][idx], k in BINARY)
                    if au is not None:
                        ww.append(r["pret"][idx].sum() * (~r["pret"][idx]).sum()); aa.append(au)
                boots.append(np.average(aa, weights=ww) if aa and sum(ww) > 0 else np.nan)
            boots = np.array(boots); boots = boots[np.isfinite(boots)]
            rho = float(sps.spearmanr(xs, np.concatenate([values[(s, sd)]["rcwa"] for sd in SEEDS.values()]))[0]) if np.std(xs) > 0 else None
            lo, hi = ((float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
                      if boots.size else (None, None))     # every resample degenerate: too few events
            out["pooled"][s][k] = dict(auroc=strat, ci_lo=lo, ci_hi=hi, n_bootstrap=int(boots.size),
                                       ci_method="percentile bootstrap over target clusters (a target's three seeds resampled together); "
                                                 "36 detector x structure intervals are screened, so an interval excluding 0.5 is a "
                                                 "selection, not a pointwise confirmation",
                                       auroc_naive=naive, spearman=rho,
                                       ci_excludes_0p5=bool(lo is not None and (lo > 0.5 or hi < 0.5)))
    (RESULTS / "detector_bench_v8.json").write_text(json.dumps(out, indent=1))
    md = ["# Detector benchmark — pooled AUROC [95 % CI] per structure (pretender vs confirmed)", "",
          "Pooled AUROC = per-run AUROC averaged over the three seeds with weights n_pretender x n_confirmed. "
          "CI = 2000-draw percentile bootstrap over target clusters (each target's three seeds resampled together). "
          "36 intervals are screened here; one excluding 0.5 is a selection, not a pointwise confirmation.", "",
          "| detector | A | B | C |", "|---|---|---|---|"]
    for k in out["detectors"]:
        md.append(f"| {k} | " + " | ".join(f"{out['pooled'][s][k]['auroc']:.2f} [{out['pooled'][s][k]['ci_lo']:.2f}, {out['pooled'][s][k]['ci_hi']:.2f}]" for s in "ABC") + " |")
    (RESULTS / "detector_bench_v8.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); main()
