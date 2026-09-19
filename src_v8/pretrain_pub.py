#!/usr/bin/env python3
"""Regenerate the printed-pipeline TMM-pretrained M_TL+phys checkpoints (T52).

Mirrors Step 1-2 of upstream step0_screen/pbtl_A_redesign.py, pbtl_B_redesign.py and
pbtl_C_v2_redesign.py exactly: rng(99) draws 5000 geometries uniformly over the
training box, the anisotropic/EMA TMM engines of the vendored cb486b5 tree generate
the spectra, physics features are normalized with the TMM-set statistics, and the
MPhys ResNet-256-4 is trained 500 epochs (AdamW lr 1e-3, wd 1e-4, cosine, batch 2048,
loss = per-channel MSE, val L1 every 100 epochs, best-val restore) after
set_global_seed(42); the batch loop is the chunked GPU-resident form.  Upstream never saved the Structure-C checkpoint and overwrote
A/B under legacy filenames, so all three are regenerated here from the recipe.

Requires INVERSETL_PROFILE=pub.  Writes archived_inputs/pub/pretrained_mphys_tmm_{S}_pub.pt
and pretrain_{S}_pub.json.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn

from common import (STRUCTS, DEVICE, PROFILE, UPSTREAM_CODE, UPSTREAM_COMMIT, ARCHIVED_PUB,
                    make_model, tmm_phys_stats, build_xgeo, set_global_seed)

N_TMM, PRETRAIN_EPOCHS, PRETRAIN_LR, BS = 5000, 500, 1e-3, 2048


def tmm_engine(s):
    if s == "A":
        from src.simulation.tmm_struct_a import compute_tmm_batch
    elif s == "B":
        from src.simulation.tmm_struct_b import compute_tmm_batch
    else:
        from src.simulation.tmm_struct_c_aniso import compute_tmm_batch
    return compute_tmm_batch


def main(s):
    assert PROFILE == "pub", "run with INVERSETL_PROFILE=pub"
    cfg = STRUCTS[s]
    wl = cfg["tmm_wavelengths"].astype(np.float32)          # redesign grid (400-1800 x100)
    L = len(wl)
    lo, hi = cfg["train_bounds"]
    t_all = time.time()
    print(f"[pretrain {s} pub] device={DEVICE} code={UPSTREAM_COMMIT} grid {wl.min():.0f}-{wl.max():.0f} x{L}",
          flush=True)

    # ---- Step 1: TMM data (rng(99); the SAME generator later yields the 90/10 split)
    rng = np.random.default_rng(99)
    params = rng.uniform(lo, hi, (N_TMM, cfg["d"])).astype(np.float32)
    t0 = time.time()
    if s == "A":                                              # pbtl_A: float32 params + float32 grid
        tmm = tmm_engine(s)(params, wl, "Cr")
    else:                                                     # pbtl_B / C_v2: float64 grid
        tmm = tmm_engine(s)(params, wl.astype(np.float64), "Cr")
    print(f"  TMM: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)
    if cfg["dual"]:
        chans = {"A_TE": tmm["A_tmm_te"], "R_TE": tmm["R_tmm_te"],
                 "A_TM": tmm["A_tmm_tm"], "R_TM": tmm["R_tmm_tm"]}
    else:
        chans = {"A": tmm["A_tmm"], "R": tmm["R_tmm"]}
    chans = {k: np.clip(v, 0, 1).astype(np.float32).reshape(-1) for k, v in chans.items()}

    # physics features + TMM-set statistics (must equal common.tmm_phys_stats)
    if cfg["tmm_params_dtype"] == np.float64:
        feats = cfg["np_feat"](params.astype(np.float64), wl.astype(np.float64), "Cr")
    else:
        feats = cfg["np_feat"](params, wl, "Cr")
    Xp = feats.reshape(-1, cfg["phys_dim"]).astype(np.float32)
    pm, ps = Xp.mean(0), Xp.std(0) + 1e-8
    pm_ref, ps_ref = tmm_phys_stats(s)
    assert np.abs(pm - pm_ref).max() < 1e-5 and np.abs(ps - ps_ref).max() < 1e-5, "stats mismatch"
    Xp = ((Xp - pm) / ps).astype(np.float32)
    Xg = build_xgeo(s, params.astype(np.float64), wl.astype(np.float64))   # [wl_norm, params_norm]

    n_tr = int(N_TMM * 0.9)
    idx = rng.permutation(N_TMM)
    rows = lambda ii: np.concatenate([np.arange(i * L, (i + 1) * L) for i in ii])
    tr_rows, vl_rows = rows(idx[:n_tr]), rows(idx[n_tr:])
    keys = list(chans)

    def gpu_tensors(r):
        return (torch.tensor(Xg[r]).to(DEVICE), {k: torch.tensor(chans[k][r]).to(DEVICE) for k in keys},
                torch.tensor(Xp[r]).to(DEVICE))

    # ---- Step 2: pre-train MPhys.  GPU-resident tensors iterated in same-size chunks
    # with a per-epoch torch.randperm reshuffle — the loop upstream pbtl_C_v2_redesign.py
    # documents as mathematically equivalent to its DataLoader(shuffle=True) form
    # (same batch size, full-pass eval, per-epoch shuffle); DataLoader over GPU
    # TensorDatasets indexes item-by-item and is ~50x slower.
    set_global_seed(42)
    model = make_model(s)
    Xg_tr, y_tr, Xp_tr = gpu_tensors(tr_rows)
    Xg_vl, y_vl, Xp_vl = gpu_tensors(vl_rows)
    n_tr_rows = Xg_tr.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PRETRAIN_EPOCHS)
    crit = nn.MSELoss()
    val_keys = ("A_TE", "A_TM") if cfg["dual"] else ("A", "R")
    best_vl, best_st, hist = float("inf"), None, []
    t0 = time.time()
    for ep in range(PRETRAIN_EPOCHS):
        model.train()
        perm = torch.randperm(n_tr_rows, device=DEVICE)
        for i in range(0, n_tr_rows, BS):
            sel = perm[i:i + BS]
            out = model(Xg_tr[sel], p=Xp_tr[sel])
            loss = sum(crit(out[k], y_tr[k][sel]) for k in keys)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = vn = 0
                for i in range(0, Xg_vl.shape[0], BS):
                    out = model(Xg_vl[i:i + BS], p=Xp_vl[i:i + BS])
                    vl += sum(nn.functional.l1_loss(out[k], y_vl[k][i:i + BS], reduction="sum").item()
                              for k in val_keys)
                    vn += (min(i + BS, Xg_vl.shape[0]) - i) * len(val_keys)
                vm = vl / vn
            hist.append(vm)
            print(f"  ep {ep+1:3d}/{PRETRAIN_EPOCHS} val L1={vm*100:.3f}% ({time.time()-t0:.0f}s)", flush=True)
            if vm < best_vl:
                best_vl, best_st = vm, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_st)

    ARCHIVED_PUB.mkdir(parents=True, exist_ok=True)
    out_pt = cfg["ckpt"]
    torch.save(model.state_dict(), out_pt)
    import hashlib
    sha = hashlib.sha256(out_pt.read_bytes()).hexdigest()
    log = dict(structure=s, profile=PROFILE, upstream_code=str(UPSTREAM_CODE), upstream_commit=UPSTREAM_COMMIT,
               recipe="pbtl_{A,B}_redesign.py / pbtl_C_v2_redesign.py Step 1-2", n_tmm=N_TMM, epochs=PRETRAIN_EPOCHS,
               lr=PRETRAIN_LR, batch=BS, weight_decay=1e-4, seed=42, phys_dim=cfg["phys_dim"],
               train_lo=lo.tolist(), train_hi=hi.tolist(), grid=[float(wl.min()), float(wl.max()), L],
               best_val_l1_pct=best_vl * 100, val_hist_pct=[h * 100 for h in hist],
               ckpt=str(out_pt), ckpt_sha256=sha, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               elapsed_s=round(time.time() - t_all, 1))
    (ARCHIVED_PUB / f"pretrain_{s}_pub.json").write_text(json.dumps(log, indent=2))
    print(f"[pretrain {s} pub] best val L1 {best_vl*100:.3f}%  -> {out_pt.name} sha256 {sha[:12]}  "
          f"({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True, choices=list(STRUCTS))
    main(ap.parse_args().structure)
