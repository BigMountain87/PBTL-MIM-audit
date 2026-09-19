"""Feature-size census of the companion's Structure B dataset (Paper 1, [1]).

Paper 1 justifies its 64 x 64 real-space raster as "established practice for MIM absorbers
with feature sizes >= 50 nm" (its Section 2, RCWA methods).  Structure B's design space
allows ring widths and disk radii down to 10 nm, so this script measures, on the manifested
dataset behind Paper 1's printed tables (`struct_B_500_redesign.npz`, sha256 in
results_pub/INPUTS_MANIFEST.txt), how many samples actually satisfy that premise.

Four in-plane features of the ring-disk unit cell are taken from the eight parameters
(P, R_out, R_in, R_disk, t_Cr, d_SiO2, theta, phi): ring width R_out - R_in, ring-disk gap
R_in - R_disk, edge gap P/2 - R_out, and disk diameter 2 R_disk.  The smallest of the four
is the sample's minimum feature.  The raster pixel is P/64.

    python3 scripts/paper1_b_feature_census_v11.py --npz <path> [--results results_pub]

Writes <results>/paper1_b_feature_census_v11.{json,md}.  Read-only on the dataset.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREMISE_NM = 50.0
GRID = 64


def main(npz, results):
    z = np.load(npz, allow_pickle=True)
    X = np.asarray(z["params"], dtype=np.float64)
    names = [str(n) for n in z["param_names"]]
    assert names[:4] == ["P", "R_out", "R_in", "R_disk"], names
    P, Ro, Ri, Rd = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    feats = {"ring_width": Ro - Ri, "ring_disk_gap": Ri - Rd,
             "edge_gap": P / 2 - Ro, "disk_diameter": 2 * Rd}
    fmin = np.min(np.stack(list(feats.values()), 1), 1)
    feats["min_feature"] = fmin
    px = P / GRID
    n = len(X)
    rows = {}
    for k, v in feats.items():
        rows[k] = dict(median_nm=float(np.median(v)), min_nm=float(v.min()), max_nm=float(v.max()),
                       n_below_50nm=int((v < PREMISE_NM).sum()), pct_below_50nm=float(100 * np.mean(v < PREMISE_NM)),
                       n_below_20nm=int((v < 20).sum()), pct_below_20nm=float(100 * np.mean(v < 20)))
    out = dict(
        dataset=str(Path(npz).name), sha256=hashlib.sha256(Path(npz).read_bytes()).hexdigest(),
        n_samples=n, premise_nm=PREMISE_NM, grid=GRID,
        design_space_min_nm=dict(R_disk=10, R_in=30, R_out=80),
        pixel_nm=dict(median=float(np.median(px)), min=float(px.min()), max=float(px.max())),
        features=rows,
        min_feature_below_1px=dict(n=int((fmin < px).sum()), pct=float(100 * np.mean(fmin < px))),
        min_feature_below_2px=dict(n=int((fmin < 2 * px).sum()), pct=float(100 * np.mean(fmin < 2 * px))),
        n_reliable=int(np.asarray(z["reliable"]).sum()) if "reliable" in z.files else None,
    )
    R = ROOT / results
    (R / "paper1_b_feature_census_v11.json").write_text(json.dumps(out, indent=2))
    md = ["# Structure B feature-size census (Paper 1 dataset)", "",
          f"`{out['dataset']}` sha256 `{out['sha256'][:12]}…`, n = {n}; premise: features ≥ {PREMISE_NM:.0f} nm; pixel = P/{GRID}",
          "", "| feature | median (nm) | min (nm) | < 50 nm | < 20 nm |", "|---|---:|---:|---:|---:|"]
    for k, r in rows.items():
        md.append(f"| {k} | {r['median_nm']:.1f} | {r['min_nm']:.1f} | {r['n_below_50nm']} ({r['pct_below_50nm']:.1f} %) | {r['n_below_20nm']} ({r['pct_below_20nm']:.1f} %) |")
    md += ["", f"min feature < 1 px: {out['min_feature_below_1px']['n']} ({out['min_feature_below_1px']['pct']:.1f} %); "
               f"< 2 px: {out['min_feature_below_2px']['n']} ({out['min_feature_below_2px']['pct']:.1f} %); "
               f"pixel median {out['pixel_nm']['median']:.1f} nm"]
    (R / "paper1_b_feature_census_v11.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--results", default="results_pub")
    a = ap.parse_args()
    main(a.npz, a.results)
