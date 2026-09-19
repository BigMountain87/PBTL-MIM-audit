#!/usr/bin/env python3
"""Assemble (or just check) the release archive behind the Data-availability statement.

    python3 scripts/build_release_archive.py --check            # list what is present/missing, verify hashes
    python3 scripts/build_release_archive.py --stage release/   # copy everything into a staging dir + MANIFEST.sha256 + tarball

Runs on the compute host, where the inputs live.  Contents (manuscript §Data availability):
  code            src_v8/ (incl. vendored upstream_920b1bd + upstream_cb486b5), scripts/
  protocol/record docs/v8_protocol.md, docs/structure_D_prespecification.md, docs/PROGRESS.md,
                  docs/compute_log_v10.md, docs/citation_check_v11.md, docs/review_full_scope_codex_v11/
  paper           paper/manuscript_v11.md, paper/supplementary_v11.md, paper/mlst/{main,supplementary}.pdf, figures_v11/
  artifacts       results_pub/ (all, incl. oracle_cache and n50/), results_v8/ (as-submitted arm)
  inputs          the four pub datasets + five pub checkpoints (INPUTS_MANIFEST.txt in results_pub),
                  the three legacy datasets + three legacy checkpoints (INPUTS_MANIFEST.txt in results_v8)
Every input is verified against the manifest that names it before it is copied; a mismatch aborts.
"""
from __future__ import annotations
import argparse, hashlib, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
CODE = ["src_v8", "scripts"]
DOCS = ["docs/v8_protocol.md", "docs/structure_D_prespecification.md", "docs/PROGRESS.md", "docs/compute_log_v10.md",
        "docs/citation_check_v11.md", "docs/author_review_v11.md", "docs/review_full_scope_codex_v11"]
PAPER = ["paper/manuscript_v11.md", "paper/supplementary_v11.md", "paper/mlst/main.pdf", "paper/mlst/supplementary.pdf",
         "paper/mlst/cover_letter.md", "figures_v11"]
RESULTS = ["results_pub", "results_v8"]
# (manifest file, base dir the manifest's paths are relative to, which arm)
INPUT_MANIFESTS = [(ROOT / "results_pub/INPUTS_MANIFEST.txt", ROOT, "pub"),
                   (ROOT / "results_v8/INPUTS_MANIFEST.txt", ROOT / "upstream_inputs_v8", "legacy")]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inputs():
    """Yield (arm, relpath, abspath, expected_sha) for every manifested input."""
    for mf, base, arm in INPUT_MANIFESTS:
        if not mf.exists():
            print(f"  [inputs] manifest missing: {mf}"); continue
        for line in mf.read_text().splitlines():
            if arm == "legacy" and line.startswith("# pub inputs"):
                break                                   # results_v8's manifest repeats the pub arm; results_pub's is authoritative
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            sha, rel = parts[0], parts[-1]
            yield arm, rel, base / rel, sha


def check():
    ok = True
    for arm, rel, p, sha in inputs():
        if not p.exists():
            print(f"  MISSING  [{arm}] {rel}"); ok = False; continue
        got = sha256(p)
        print(f"  {'ok      ' if got == sha else 'HASH!!  '} [{arm}] {rel}"); ok &= (got == sha)
    for rel in CODE + DOCS + PAPER + RESULTS:
        p = ROOT / rel
        print(f"  {'ok      ' if p.exists() else 'MISSING '} {rel}")
        ok &= p.exists()
    return ok


def stage(dst):
    dst = Path(dst).resolve(); dst.mkdir(parents=True, exist_ok=True)
    for rel in CODE + DOCS + PAPER + RESULTS:
        src = ROOT / rel; tgt = dst / rel
        if src.is_dir():
            shutil.copytree(src, tgt, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "*.tmp.npz"))
        else:
            tgt.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, tgt)
    for arm, rel, p, sha in inputs():
        if sha256(p) != sha:
            sys.exit(f"input hash mismatch: {arm} {rel}")
        tgt = dst / "inputs" / arm / Path(rel).name
        tgt.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, tgt)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (dst / "RELEASE_INFO.txt").write_text(f"InverseTL release staged {time.strftime('%Y-%m-%d %H:%M %Z')}\ngit HEAD {git or 'n/a'}\n"
                                          "Inputs copied into inputs/<arm>/ and verified against results_*/INPUTS_MANIFEST.txt.\n")
    lines = []
    for f in sorted(p for p in dst.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"):
        lines.append(f"{sha256(f)}  {f.relative_to(dst)}")
    (dst / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    tar = dst.with_suffix(".tar.gz")
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(dst, arcname=dst.name)
    print(f"staged {len(lines)} files -> {dst}; tarball {tar} ({tar.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--stage", default=None)
    a = ap.parse_args()
    if a.check or not a.stage:
        print("release check:", "COMPLETE" if check() else "INCOMPLETE")
    else:
        if not check():
            sys.exit("archive incomplete; fix the MISSING/HASH entries first")
        stage(a.stage)
