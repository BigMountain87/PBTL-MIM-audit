#!/usr/bin/env python3
"""v8 unified fine-tune — faithful Paper-1 stage-2 recipe for any structure.

Mirrors pbtl_{A,B,C} exactly: filtered+clipped data, rng(42) split,
[wl, params] input order, TMM-stat feature normalization, strict checkpoint
load, AdamW 3e-4 / wd 1e-4 / cosine / 1000 epochs / batch 512,
loss = sum of per-channel MSE (A+R; C: 4 channels), best-val restore.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch

from common import (STRUCTS, DEVICE, RESULTS, PROFILE, UPSTREAM_COMMIT, make_model, seed_sfx,
                    load_filtered, make_split, tmm_phys_stats, build_xgeo,
                    build_phys, set_global_seed, env_info)


def channel_keys(s):
    return ("A_TE", "R_TE", "A_TM", "R_TM") if STRUCTS[s]["dual"] else ("A", "R")


def val_keys(s):
    # upstream val metric: A+R for single-pol, A_TE+A_TM for dual
    return ("A_TE", "A_TM") if STRUCTS[s]["dual"] else ("A", "R")


def main(s, seed=42, init="pretrained", lr=3e-4):
    cfg = STRUCTS[s]
    sx = seed_sfx(seed)
    t_all = time.time()
    print(f"[finetune {s} v8] device={DEVICE} seed={seed} init={init} lr={lr}", flush=True)

    data = load_filtered(s)
    wl = data["wavelengths"]
    L = len(wl)
    train_idx, val_idx, test_idx = make_split(data["n_good"], subset_seed=seed)
    n_train = len(train_idx)
    print(f"  good={data['n_good']}  split: train={n_train} val={len(val_idx)} "
          f"test={len(test_idx)}  wl {wl.min():.0f}-{wl.max():.0f} nm × {L}", flush=True)

    phys_mean, phys_std = tmm_phys_stats(s)
    print(f"  TMM-set phys stats reconstructed (dim={cfg['phys_dim']})", flush=True)

    def tensors_for(idx):
        p = data["params"][idx]
        Xg = torch.tensor(build_xgeo(s, p, wl), device=DEVICE)
        Xp = torch.tensor(build_phys(s, p, wl, phys_mean, phys_std), device=DEVICE)
        ys = {k: torch.tensor(data["chans"][k][idx].reshape(-1), device=DEVICE)
              for k in channel_keys(s)}
        return Xg, Xp, ys

    Xg_tr, Xp_tr, y_tr = tensors_for(train_idx)
    Xg_va, Xp_va, y_va = tensors_for(val_idx)
    Xg_te, Xp_te, y_te = tensors_for(test_idx)

    # TODO(T04): torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8
    # would make CUDA training bitwise reproducible; deliberately NOT enabled so that the
    # archived runs (trained without it) and new runs share one recipe.
    set_global_seed(seed)
    model = make_model(s)
    if init == "pretrained":
        sd = torch.load(cfg["ckpt"], map_location=DEVICE, weights_only=True)
        model.load_state_dict(sd, strict=True)
    else:   # 'scratch' = the M0-style from-scratch control: same architecture, random init
        print("  init=scratch: no checkpoint loaded (from-scratch control)", flush=True)

    def eval_val():
        model.eval()
        with torch.no_grad():
            out = model(Xg_va, p=Xp_va)
            tot = sum(torch.abs(out[k] - y_va[k]).sum() for k in val_keys(s))
            return float(tot.item() / (Xg_va.shape[0] * len(val_keys(s))) * 100)

    def eval_a_only(Xg, Xp, ys):
        model.eval()
        with torch.no_grad():
            out = model(Xg, p=Xp)
            keys = ("A_TE", "A_TM") if cfg["dual"] else ("A",)
            tot = sum(torch.abs(out[k] - ys[k]).sum() for k in keys)
            return float(tot.item() / (Xg.shape[0] * len(keys)) * 100)

    pre_val = eval_val()
    pre_val_a = eval_a_only(Xg_va, Xp_va, y_va)
    print(f"  pre-fine-tune val MAE: {pre_val:.2f}% (upstream metric) / "
          f"{pre_val_a:.2f}% (A-channel)", flush=True)

    epochs, batch = 1000, 512
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_rows = Xg_tr.shape[0]
    best_val, best_state = float("inf"), None
    val_hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_rows, device=DEVICE)
        for st in range(0, n_rows, batch):
            sel = perm[st:st + batch]
            opt.zero_grad()
            out = model(Xg_tr[sel], p=Xp_tr[sel])
            loss = sum(torch.mean((out[k] - y_tr[k][sel]) ** 2)
                       for k in channel_keys(s))
            loss.backward()
            opt.step()
        sch.step()
        if (ep + 1) % 100 == 0:
            vm = eval_val()
            val_hist.append(vm)
            if vm < best_val:
                best_val = vm
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            print(f"  ep {ep+1:4d}/{epochs}  val={vm:5.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mae = eval_a_only(Xg_te, Xp_te, y_te)
    # per-spectrum test MAEs (A channel; C: mean of TE/TM per spectrum)
    model.eval()
    with torch.no_grad():
        out = model(Xg_te, p=Xp_te)
        keys = ("A_TE", "A_TM") if cfg["dual"] else ("A",)
        per_spec = np.mean([np.abs((out[k] - y_te[k]).view(len(test_idx), L)
                                   .cpu().numpy()).mean(axis=1) for k in keys],
                           axis=0) * 100
    print(f"[finetune {s} v8] FINAL test MAE: {test_mae:.2f}% "
          f"(per-spectrum {per_spec.mean():.2f} ± {per_spec.std():.2f}%)", flush=True)

    wpath = RESULTS / f"surrogate_{s}{sx}_v8.pt"
    torch.save(model.state_dict(), wpath)
    np.savez(RESULTS / f"stats_{s}{sx}_v8.npz",
             phys_mean=phys_mean, phys_std=phys_std, wavelengths=wl,
             design_lo=cfg["design_bounds"][0], design_hi=cfg["design_bounds"][1],
             train_lo=cfg["train_bounds"][0], train_hi=cfg["train_bounds"][1])
    log = dict(version="v8", profile=PROFILE, upstream_commit=UPSTREAM_COMMIT,
               hygiene_filter=cfg["filter"], ckpt=str(cfg["ckpt"]) if init == "pretrained" else None,
               init=init, env=env_info(),
               structure=s, seed=int(seed), n_good=data["n_good"],
               n_train=n_train, n_val=len(val_idx), n_test=len(test_idx),
               epochs=epochs, lr=lr, batch=batch, weight_decay=1e-4,
               loss="sum per-channel MSE " + "+".join(channel_keys(s)),
               input_order="[wl_norm, params_norm]",
               phys_stats="TMM-set rng(99) reconstruction",
               wavelengths_min_nm=float(wl.min()), wavelengths_max_nm=float(wl.max()),
               wavelengths_count=L,
               pre_finetune_val_mae_pct=pre_val,
               pre_finetune_val_mae_a_only_pct=pre_val_a,
               best_val_mae_pct=best_val, val_hist_every_100ep=val_hist,
               final_test_mae_pct=test_mae,
               test_per_spectrum_mae_mean_pct=float(per_spec.mean()),
               test_per_spectrum_mae_std_pct=float(per_spec.std()),
               train_idx=train_idx.tolist(), val_idx=val_idx.tolist(),
               test_idx=test_idx.tolist(),
               orig_dataset_idx=data["orig_idx"].tolist(),
               elapsed_s=round(time.time() - t_all, 1))
    (RESULTS / f"finetune_{s}{sx}_v8.json").write_text(json.dumps(log, indent=2))
    print(f"[finetune {s} v8] saved {wpath.name} (+stats, +log)  "
          f"total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True, choices=list(STRUCTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", default="pretrained", choices=["pretrained", "scratch"],
                    help="pretrained = strict-load the Paper-1 TMM checkpoint (default); "
                         "scratch = random init (M0 from-scratch control, T06)")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="fine-tune learning rate (upstream: 3e-4 for TL, 1e-3 for M0)")
    a = ap.parse_args()
    main(a.structure, seed=a.seed, init=a.init, lr=a.lr)
