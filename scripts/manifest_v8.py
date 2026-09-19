#!/usr/bin/env python3
"""Write <results>/MANIFEST.sha256 (and the identical SHA256SUMS) over every file in the
results directory except the manifests themselves.  Usage: python3 scripts/manifest_v8.py [results_v8]"""
import hashlib, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results_v8")
lines = []
for f in sorted(p for p in R.iterdir() if p.is_file() and p.name not in ("MANIFEST.sha256", "SHA256SUMS")):
    lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
for name in ("MANIFEST.sha256", "SHA256SUMS"):
    (R / name).write_text("\n".join(lines) + "\n")
print(f"{len(lines)} files -> {R/'MANIFEST.sha256'} (+ SHA256SUMS)")
