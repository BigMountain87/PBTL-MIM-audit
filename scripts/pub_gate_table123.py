#!/usr/bin/env python3
"""T52 gate: does the pub-profile fine-tune (n=350, seed 42, M_TL+phys) reproduce the
printed Paper-1 Tables 1-3?  Reads results_pub/finetune_{A,B,C}_v8.json and
docs/pub_reference_table123.json (seed-42 and 10-seed mean/std from the upstream
pbtl_*_redesign_10seed.npz archives that produced the printed tables).

PASS per structure: |ours - mean10| <= 3*std10 (10-seed spread) — and we also report
the gap to the seed-42 archive value (same seed; differs only by CUDA nondeterminism).
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ref = json.load(open(ROOT / "docs/pub_reference_table123.json"))["values_fraction"]
res_dir = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results_pub")
ok_all, rows = True, []
for s in "ABC":
    f = res_dir / f"finetune_{s}_v8.json"
    if not f.exists():
        rows.append((s, "MISSING", "", "", "", "NOT RUN")); ok_all = False; continue
    d = json.load(open(f))
    ours = d["final_test_mae_pct"]
    m, sd, s42 = (ref[s][k]["M_TL+phys"] * 100 for k in ("mean10", "std10", "seed42"))
    z = (ours - m) / sd
    ok = abs(z) <= 3.0
    ok_all &= ok
    rows.append((s, f"{ours:.3f}", f"{s42:.3f} ({ours - s42:+.3f})", f"{m:.2f} ± {sd:.2f}", f"{z:+.2f}", "PASS" if ok else "FAIL",
                 d.get("pre_finetune_val_mae_pct"), d.get("profile"), d.get("upstream_commit")))
print(f"{'S':<2} {'ours %':>8} {'seed-42 archive (gap)':>24} {'10-seed printed':>16} {'z':>6}  verdict   pre-FT val %  profile")
for r in rows:
    print(f"{r[0]:<2} {r[1]:>8} {r[2]:>24} {r[3]:>16} {r[4]:>6}  {r[5]:<8}  {r[6] if len(r)>6 else ''}  {r[7] if len(r)>7 else ''} {r[8] if len(r)>8 else ''}")
print("\nGATE:", "PASS" if ok_all else "FAIL")
out = {"rows": [dict(structure=r[0], ours_pct=r[1], seed42_archive=r[2], printed_10seed=r[3], z=r[4], verdict=r[5]) for r in rows], "gate": "PASS" if ok_all else "FAIL"}
(res_dir / "pub_gate_table123.json").write_text(json.dumps(out, indent=1)) if res_dir.exists() else None
sys.exit(0 if ok_all else 1)
