#!/usr/bin/env python3
"""T12(1) — lookup null: how well does a nearest-training-neighbour (or the mean training
spectrum) already match each inverse-design target, with no optimizer at all?
numpy only.  Usage: python3 scripts/lookup_null_v8.py [--results results_v8] [--profile legacy|pub]
Writes <results>/lookup_null_v8.json.  Expected seed-42 NN counts (legacy): 15/14/10.
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import os
RAW = Path(os.environ.get("INVERSETL_UPSTREAM_ROOT", Path.home() / "mim_novel")) / "data/raw"   # Paper-1 inputs root
DATASETS = {"legacy": {"A": "struct_A_vis_500.npz", "B": "struct_B_500.npz", "C": "struct_C_500.npz"},
            "pub": {"A": "struct_A_500_redesign.npz", "B": "struct_B_500_redesign.npz", "C": "struct_C_500_redesign.npz"}}
SEEDS = {"": 42, "_s123": 123, "_s777": 777}
TAU = 5.0


def main(results, profile):
    R = ROOT / results
    out = dict(results_dir=results, profile=profile, tau_pct=TAU, runs={})
    for s in "ABC":
        raw = np.load(RAW / DATASETS[profile][s], allow_pickle=True)
        keys = ("A_TE", "A_TM") if s == "C" else ("A",)
        spec = {k: np.clip(raw[k].astype(np.float64), 0, 1) for k in keys}
        for sx, seed in SEEDS.items():
            ft = json.load(open(R / f"finetune_{s}{sx}_v8.json"))
            inv = np.load(R / f"inverse_{s}{sx}_v8.npz", allow_pickle=True)
            train_rows = np.asarray(ft["orig_dataset_idx"])[np.asarray(ft["train_idx"])]
            tgt_rows = inv["orig_indices"]
            # sanity: the archived targets must equal the dataset rows (after clipping)
            for k in keys:
                assert np.allclose(inv[f"A_target_{k}"], spec[k][tgt_rows], atol=1e-6), (s, sx, k)
            nn_mae, nn_idx, mean_mae = [], [], []
            for t in tgt_rows:
                d = np.mean([np.mean(np.abs(spec[k][train_rows] - spec[k][t]), axis=1) for k in keys], axis=0)
                j = int(np.argmin(d)); nn_mae.append(float(d[j] * 100)); nn_idx.append(int(train_rows[j]))
                mean_mae.append(float(np.mean([np.mean(np.abs(spec[k][train_rows].mean(0) - spec[k][t])) for k in keys]) * 100))
            nn_mae, mean_mae = np.array(nn_mae), np.array(mean_mae)
            surr = inv["mae_surrogate"].astype(float)
            out["runs"][f"{s}{sx}"] = dict(
                structure=s, seed=seed, n_targets=int(len(tgt_rows)), n_train=int(len(train_rows)),
                nn_mae_pct=nn_mae.tolist(), nn_train_orig_idx=nn_idx, mean_spectrum_mae_pct=mean_mae.tolist(),
                nn_le_tau=int((nn_mae <= TAU).sum()), mean_le_tau=int((mean_mae <= TAU).sum()),
                surrogate_claimed_le_tau=int((surr <= TAU).sum()),
                nn_mae_median_pct=float(np.median(nn_mae)), mean_spectrum_mae_median_pct=float(np.median(mean_mae)),
                surr_claimed_mae_median_pct=float(np.median(surr)),
                n_targets_where_nn_beats_claimed=int((nn_mae < surr).sum()))
    (R / "lookup_null_v8.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {R/'lookup_null_v8.json'}")
    for k, v in out["runs"].items():
        print(f"{k:8s} NN<=tau {v['nn_le_tau']:2d}/{v['n_targets']}  mean-spec<=tau {v['mean_le_tau']:2d}  "
              f"NN median {v['nn_mae_median_pct']:.2f}%  claimed median {v['surr_claimed_mae_median_pct']:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results_v8"); ap.add_argument("--profile", default="legacy", choices=["legacy", "pub"])
    a = ap.parse_args(); main(a.results, a.profile)
