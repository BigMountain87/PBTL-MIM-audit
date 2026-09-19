#!/usr/bin/env python3
"""T03 — compare archived reliability JSONs with regenerated ones.

For each of the nine runs (A/B/C x 42/123/777): ignore meta.seed and meta.env;
integer counts, n, tier1a, taxonomy flags and per_sample mae_surr/mae_rcwa/delta must
be exactly equal; taxonomy.t4_mean_pert_mae_pct (per sample) within 1e-3;
discrimination rho / p within 1e-3.  Prints PASS/FAIL per run, exit 1 on any FAIL.
Usage: python3 scripts/compare_reliability_json.py <archived_dir> <regenerated_dir>
"""
import json, sys
from pathlib import Path
import numpy as np

A, B = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = {"": 42, "_s123": 123, "_s777": 777}
TOL = 1e-3


def eq_exact(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def close(a, b, tol=TOL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False
    m = np.isfinite(a) & np.isfinite(b)
    return bool(np.array_equal(np.isfinite(a), np.isfinite(b)) and np.all(np.abs(a[m] - b[m]) <= tol))


ok_all = True
for s in "ABC":
    for sx, seed in SEEDS.items():
        fa, fb = A / f"reliability_{s}{sx}_v8.json", B / f"reliability_{s}{sx}_v8.json"
        if not fb.exists():
            print(f"{s} {seed}: FAIL (regenerated file missing)"); ok_all = False; continue
        da, db = json.load(open(fa)), json.load(open(fb))
        checks = {}
        ma = {k: v for k, v in da["meta"].items() if k not in ("seed", "env")}
        mb = {k: v for k, v in db["meta"].items() if k not in ("seed", "env")}
        checks["meta (ex seed/env)"] = eq_exact(ma, mb)
        checks["tier1a"] = eq_exact(da["tier1a"], db["tier1a"])
        checks["tier1b counts"] = eq_exact({k: v for k, v in da["tier1b"].items() if isinstance(v, int)},
                                           {k: v for k, v in db["tier1b"].items() if isinstance(v, int)})
        ta, tb = da["taxonomy"], db["taxonomy"]
        checks["taxonomy counts"] = all(ta[k] == tb[k] for k in ("t1", "t2", "t3", "t4", "mode_count"))
        checks["taxonomy per-sample flags"] = eq_exact({k: ta["per_sample"][k] for k in ("t1", "t2", "t3", "t4")},
                                                       {k: tb["per_sample"][k] for k in ("t1", "t2", "t3", "t4")})
        checks["t4 mean pert MAE ±1e-3"] = close(ta["t4_mean_pert_mae_pct"], tb["t4_mean_pert_mae_pct"])
        pa, pb = da["per_sample"], db["per_sample"]
        checks["per_sample orig_indices"] = eq_exact(pa["orig_indices"], pb["orig_indices"])
        for k in ("mae_surr_pct", "mae_rcwa_pct", "delta_pct"):
            checks[f"per_sample {k} exact"] = close(pa[k], pb[k], tol=0.0)
        for k in ("rcwa_vs_surr_PRIMARY", "delta_vs_rcwa_secondary", "delta_vs_surr_secondary_COUPLED"):
            checks[f"discrimination {k} rho/p ±1e-3"] = (close(da["discrimination"][k]["rho"], db["discrimination"][k]["rho"])
                                                          and close(da["discrimination"][k]["p"], db["discrimination"][k]["p"]))
        ok = all(checks.values()); ok_all &= ok
        print(f"{s} {seed}: {'PASS' if ok else 'FAIL'}" + ("" if ok else "  -> " + ", ".join(k for k, v in checks.items() if not v)))
print("ALL PASS" if ok_all else "SOME FAIL")
sys.exit(0 if ok_all else 1)
