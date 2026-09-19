#!/usr/bin/env python3
"""Build an analysis VIEW of the 20 -> 50 target extension.

    python3 scripts/make_n50_view.py [--results results_pub] [--tag _n50] [--view results_pub/n50] [--allow-partial]

The extension writes tagged artifacts (inverse/rcwa/reliability_<S><seed>_n50_v8.*) beside
the protocol's untagged ones.  Every analysis script reads untagged names from a results
dir, so instead of teaching each script about tags this creates a directory of symlinks in
which the tagged artifacts appear under untagged names, and runs unchanged code on it.

What is linked: inverse/rcwa/reliability (tagged -> untagged), finetune/stats/surrogate
(shared with the protocol run, untagged).  What is deliberately NOT linked: random_baseline,
restart0, the M0/feasible controls and the D arm -- those exist for the protocol's 20
targets only, and the scripts skip what is absent rather than compare 50 with 20.

Refuses to build unless all nine tagged reliability files exist, unless --allow-partial.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ("", "_s123", "_s777")


def main(results, tag, view, allow_partial):
    R = (ROOT / results).resolve(); V = (ROOT / view).resolve()
    missing = [f"reliability_{s}{sx}{tag}_v8.json" for s in "ABC" for sx in SEEDS if not (R / f"reliability_{s}{sx}{tag}_v8.json").exists()]
    if missing and not allow_partial:
        sys.exit(f"extension incomplete ({len(missing)} of 9 reliability artifacts missing): {missing[:3]} ...")
    V.mkdir(parents=True, exist_ok=True)
    made, fallback = [], []
    for s in "ABC":
        for sx in SEEDS:
            for kind, ext in (("inverse", ".npz"), ("rcwa", ".npz"), ("reliability", ".json")):
                tagged = R / f"{kind}_{s}{sx}{tag}_v8{ext}"; plain = R / f"{kind}_{s}{sx}_v8{ext}"
                dst = V / f"{kind}_{s}{sx}_v8{ext}"
                src = tagged if tagged.exists() else (plain if allow_partial else None)
                if src is None:
                    continue
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                os.symlink(os.path.relpath(src, V), dst)
                (made if src is tagged else fallback).append(dst.name)
            for kind, ext in (("finetune", ".json"), ("stats", ".npz"), ("surrogate", ".pt")):
                src = R / f"{kind}_{s}{sx}_v8{ext}"; dst = V / src.name
                if src.exists():
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    os.symlink(os.path.relpath(src, V), dst)
    (V / "VIEW_README.txt").write_text(
        f"Analysis view of the {tag} extension built by scripts/make_n50_view.py from {R}.\n"
        f"Tagged artifacts linked under untagged names: {len(made)}\n"
        + (f"PARTIAL VIEW -- protocol (20-target) artifacts used where the extension is missing: {fallback}\n" if fallback else "")
        + "No random_baseline/restart0/M0/feasible/D artifacts on purpose (they exist for the 20 protocol targets only).\n")
    print(f"view {V}: {len(made)} tagged links" + (f", {len(fallback)} protocol fallbacks (PARTIAL)" if fallback else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_pub"); ap.add_argument("--tag", default="_n50")
    ap.add_argument("--view", default="results_pub/n50"); ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args(); main(a.results, a.tag, a.view, a.allow_partial)
