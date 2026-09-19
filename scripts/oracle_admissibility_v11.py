"""Census of reference-solver spectra outside the companion's admissibility band.

Paper 1 keeps a generated sample only if A in [-0.005, 1.005] at every wavelength and then
clips A to [0, 1] (its `reliable` mask; grazing-order artefact at oblique incidence).  The
audit's oracle (src_v8/oracle.py, simulate_cached) checks finiteness only, so a committed
design's reference spectrum can carry such a point into the MAE.  This script lists every
archived committed-design oracle spectrum with a point outside the band and reports the MAE
(a) as archived, (b) with the spectrum clipped to [0, 1] as the companion does, and (c) with
the offending wavelengths dropped, so the reader can see whether any verdict depends on it.

    python3 scripts/oracle_admissibility_v11.py [--results results_pub]

Writes <results>/oracle_admissibility_v11.{json,md}.
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAU, LO, HI = 5.0, -0.005, 1.005


def main(results):
    R = ROOT / results
    rows, n_designs, n_spectra = [], 0, 0
    files = sorted(glob.glob(str(R / "rcwa_?_v8.npz")) + glob.glob(str(R / "rcwa_?_s*_v8.npz")))
    files = [f for f in files if not any(t in os.path.basename(f) for t in ("_n50", "_m0", "_feas", "repro"))]
    for f in files:
        b = os.path.basename(f); s = b[5]
        seed = b.split("_s")[1].split("_")[0] if "_s" in b else "42"
        z = np.load(f, allow_pickle=True)
        chans = [k[7:] for k in z.files if k.startswith("A_rcwa_")]
        tg = {c: np.asarray(z[f"A_target_{c}"]) for c in chans}
        A = {c: np.asarray(z[f"A_rcwa_{c}"]) for c in chans}
        for i in range(A[chans[0]].shape[0]):
            if not all(np.all(np.isfinite(A[c][i])) for c in chans):
                continue
            n_designs += 1; n_spectra += len(chans)
            out = [(c, A[c][i]) for c in chans if A[c][i].min() < LO or A[c][i].max() > HI]
            if not out:
                continue
            mae = lambda fn: float(100 * np.mean([np.mean(np.abs(fn(c, A[c][i]) - tg[c][i])) for c in chans]))
            raw = mae(lambda c, a: a)
            clipped = mae(lambda c, a: np.clip(a, 0, 1))
            keep = {c: ~((A[c][i] < LO) | (A[c][i] > HI)) for c in chans}
            dropped = float(100 * np.mean([np.mean(np.abs(A[c][i][keep[c]] - tg[c][i][keep[c]])) for c in chans]))
            for c, a in out:
                j = int(np.argmin(a)) if a.min() < LO else int(np.argmax(a))
                rows.append(dict(structure=s, seed=int(seed), index=int(i), channel=c,
                                 extreme=float(a[j]), wavelength_nm=float(z["wavelengths"][j]),
                                 n_points_outside=int(((a < LO) | (a > HI)).sum()),
                                 surrogate_mae_pct=float(z["mae_surrogate"][i]),
                                 mae_archived_pct=raw, mae_clipped_pct=clipped, mae_dropped_pct=dropped,
                                 verdict_archived="pretender" if raw > TAU else "pass",
                                 verdict_clipped="pretender" if clipped > TAU else "pass",
                                 verdict_dropped="pretender" if dropped > TAU else "pass"))
    out = dict(band=[LO, HI], tau_pct=TAU, results_dir=results, n_designs_checked=n_designs,
               n_spectra_checked=n_spectra, n_outside=len(rows),
               n_verdict_changes_clipped=sum(r["verdict_archived"] != r["verdict_clipped"] for r in rows),
               n_verdict_changes_dropped=sum(r["verdict_archived"] != r["verdict_dropped"] for r in rows),
               rows=rows)
    (R / "oracle_admissibility_v11.json").write_text(json.dumps(out, indent=2))
    md = [f"# Oracle spectra outside the companion's admissibility band [{LO}, {HI}]", "",
          f"{n_designs} committed designs ({n_spectra} spectra) in `{results}`; {len(rows)} outside the band; "
          f"verdict changes if clipped to [0, 1]: {out['n_verdict_changes_clipped']}; if the points are dropped: {out['n_verdict_changes_dropped']}.", "",
          "| structure | seed | index | channel | extreme | at (nm) | points | MAE archived | clipped | dropped | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['structure']} | {r['seed']} | {r['index']} | {r['channel']} | {r['extreme']:+.4f} | {r['wavelength_nm']:.0f} | "
                  f"{r['n_points_outside']} | {r['mae_archived_pct']:.2f} % | {r['mae_clipped_pct']:.2f} % | {r['mae_dropped_pct']:.2f} % | "
                  f"{r['verdict_archived']} → {r['verdict_clipped']} / {r['verdict_dropped']} |")
    (R / "oracle_admissibility_v11.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results_pub")
    main(ap.parse_args().results)
