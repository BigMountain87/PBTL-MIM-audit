#!/usr/bin/env python3
"""v8 pre-flight gate. Run BEFORE any pipeline stage. Verifies:

1. dataset hygiene filter reproduces the expected good-sample counts
2. numpy and torch physics-feature implementations agree (all structures)
3. pretrained checkpoints load strict into the v8 model classes
4. faithful pre-fine-tune val MAE (regression check: B ≈ 20.5 % A-channel)
5. the inverse optimizer actually moves (anti-frozen-optimizer smoke test)
"""
from __future__ import annotations
import numpy as np
import torch

from common import (STRUCTS, DEVICE, make_model, load_filtered, make_split,
                    tmm_phys_stats, build_xgeo, build_phys)

print(f"[selfcheck] device={DEVICE}")
ok = True

# ---- 1+2: data counts + feature parity --------------------------------
for s, cfg in STRUCTS.items():
    data = load_filtered(s)   # raises on count mismatch
    print(f"[1] {s}: good={data['n_good']} (expected {cfg['expected_good']}) OK")

    rng = np.random.default_rng(0)
    lo, hi = cfg["design_bounds"]
    geos = rng.uniform(lo, hi, (8, cfg["d"]))
    wl = data["wavelengths"]
    feats_np = cfg["np_feat"](geos.astype(np.float64), wl.astype(np.float64),
                              "Cr")                      # (8, L, F)
    geos_t = torch.tensor(geos, dtype=torch.float32, device=DEVICE)
    worst = 0.0
    for li in rng.choice(len(wl), 5, replace=False):
        ft = cfg["torch_feat"](geos_t, float(wl[li]), metal="Cr").cpu().numpy()
        worst = max(worst, float(np.abs(ft - feats_np[:, li, :]).max()))
    status = "OK" if worst < 5e-3 else "FAIL"
    if worst >= 5e-3:
        ok = False
    print(f"[2] {s}: torch-vs-numpy feature max|diff|={worst:.2e} {status}")

# ---- 3+4: checkpoint load + faithful pre-FT val MAE -------------------
for s, cfg in STRUCTS.items():
    model = make_model(s)
    sd = torch.load(cfg["ckpt"], map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[3] {s}: checkpoint loads strict OK")

    data = load_filtered(s)
    wl = data["wavelengths"]
    _, val_idx, _ = make_split(data["n_good"])
    pm, ps = tmm_phys_stats(s)
    p = data["params"][val_idx]
    Xg = torch.tensor(build_xgeo(s, p, wl), device=DEVICE)
    Xp = torch.tensor(build_phys(s, p, wl, pm, ps), device=DEVICE)
    keys = ("A_TE", "A_TM") if cfg["dual"] else ("A",)
    with torch.no_grad():
        out = model(Xg, p=Xp)
        mae = float(np.mean([torch.abs(out[k] - torch.tensor(
            data["chans"][k][val_idx].reshape(-1), device=DEVICE)).mean().item()
            for k in keys]) * 100)
    print(f"[4] {s}: pre-fine-tune val MAE (A-channel) = {mae:.2f}%")

# ---- 5: anti-frozen-optimizer smoke test ------------------------------
s = "B"
cfg = STRUCTS[s]
data = load_filtered(s)
wl = data["wavelengths"]
pm, ps = tmm_phys_stats(s)
model = make_model(s)
model.load_state_dict(torch.load(cfg["ckpt"], map_location=DEVICE,
                                 weights_only=True))
model.eval()
lo = torch.tensor(cfg["design_bounds"][0], dtype=torch.float32, device=DEVICE)
hi = torch.tensor(cfg["design_bounds"][1], dtype=torch.float32, device=DEVICE)
tlo = torch.tensor(cfg["train_bounds"][0], dtype=torch.float32, device=DEVICE)
thi = torch.tensor(cfg["train_bounds"][1], dtype=torch.float32, device=DEVICE)
pm_t = torch.tensor(pm, device=DEVICE)
ps_t = torch.tensor(ps, device=DEVICE)
L = len(wl)
wln = torch.tensor((wl - wl.min()) / (wl.max() - wl.min()),
                   dtype=torch.float32, device=DEVICE)
tgt = torch.tensor(data["chans"]["A"][0], dtype=torch.float32, device=DEVICE)
u = torch.full((3, cfg["d"]), 0.5, device=DEVICE, requires_grad=True)
u0 = u.detach().clone()
opt = torch.optim.Adam([u], lr=0.05)
for _ in range(40):
    x = lo + u * (hi - lo)
    feats = torch.stack([cfg["torch_feat"](x, lam, metal="Cr")
                         for lam in wl.tolist()], dim=0)
    feats = (feats - pm_t) / ps_t
    pn = (x - tlo) / (thi - tlo)
    xg = torch.cat([wln.view(L, 1, 1).expand(L, 3, 1),
                    pn.unsqueeze(0).expand(L, 3, cfg["d"])], dim=-1)
    out = model(xg.reshape(L * 3, -1), p=feats.reshape(L * 3, -1))
    loss = torch.mean((out["A"].view(L, 3) - tgt.unsqueeze(1)) ** 2)
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        u.clamp_(0, 1)
move = float((u.detach() - u0).abs().max())
status = "OK" if move > 0.05 else "FAIL — optimizer frozen"
if move <= 0.05:
    ok = False
print(f"[5] smoke inverse (40 iters): max|Δu| = {move:.3f} {status}")

print("\n[selfcheck] " + ("ALL CHECKS PASSED" if ok else "FAILURES — DO NOT RUN"))
raise SystemExit(0 if ok else 1)
