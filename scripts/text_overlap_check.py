#!/usr/bin/env python3
"""Mechanical text-overlap check between manuscripts (n-word shingles after stripping markup).
    python3 scripts/text_overlap_check.py A.md B.tex [--n 8]
Prints shared shingle counts and the shared phrases. Not a substitute for a result-overlap ledger."""
import argparse, re
from pathlib import Path


def words(t):
    t = t.split("\n## References")[0]
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S); t = re.sub(r"[`$*_|>#\\{}]", " ", t)
    return re.findall(r"[a-z0-9][a-z0-9'’\-\.]*", t.lower())


def shingles(ws, n):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--n", type=int, default=8)
    x = ap.parse_args(); A, B = words(Path(x.a).read_text()), words(Path(x.b).read_text())
    for n in sorted({6, x.n, 10}):
        sa, sb = shingles(A, n), shingles(B, n); c = sa & sb
        print(f"{n}-gram: {len(sa)} vs {len(sb)}, shared {len(c)} ({100 * len(c) / max(len(sa), 1):.2f} % of A)")
    seen = []
    for c in sorted(shingles(A, x.n) & shingles(B, x.n)):
        if not any(c[10:] in s or s[10:] in c for s in seen):
            seen.append(c); print("  -", c)
