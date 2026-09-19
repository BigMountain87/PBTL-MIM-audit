#!/usr/bin/env python3
"""T34 — reference-list compliance check.

Asserts: first-citation order is 1..N with no gaps; every list entry is cited at least once;
every entry carries a title and a resolving DOI or arXiv id; the tool citations are present.
Interval brackets ([0,1], [50,400], [5,5]) are excluded by requiring every number to be <= N.
Usage: python3 scripts/check_refs.py paper/manuscript_v10.md [--offline]
"""
import re, sys, urllib.request, json
from pathlib import Path

M = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/manuscript_v10.md")
OFFLINE = "--offline" in sys.argv
txt = M.read_text()
head, sep, refs = txt.partition("\n## References")
assert sep, "no ## References section"
entries = {int(m.group(1)): m.group(2).strip() for m in
           re.finditer(r"^\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", refs, re.S | re.M)}
N = len(entries)
TOKEN = re.compile(r"\[((?:\d+)(?:\s*[–\-,;]\s*\d+)*)\]")
fails, seq, seen = [], [], set()
for m in TOKEN.finditer(head):
    inner = m.group(1)
    if "." in inner:
        continue
    nums = []
    ok = True
    for part in re.split(r"\s*[,;]\s*", inner):
        rng = re.match(r"^(\d+)\s*[–\-]\s*(\d+)$", part)
        if rng:
            a, b = int(rng.group(1)), int(rng.group(2))
            if b <= a or b > N:
                ok = False; break
            nums += list(range(a, b + 1))
        elif part.isdigit():
            if int(part) > N:
                ok = False; break
            nums.append(int(part))
        else:
            ok = False; break
    if not ok:
        continue
    for n in nums:
        if n not in seen:
            seen.add(n); seq.append(n)
if seq != list(range(1, len(seq) + 1)):
    bad = next((i + 1, v) for i, v in enumerate(seq) if v != i + 1)
    fails.append(f"first-citation order breaks at position {bad[0]}: found [{bad[1]}]")
for n in sorted(entries):
    if n not in seen:
        fails.append(f"[{n}] is never cited in the body")
TOOLS = {"torcwa": "TORCWA", "Adam": "Adam:", "AdamW": "Decoupled weight decay", "PyTorch": "PyTorch:",
         "SciPy": "SciPy 1.0", "Wilson": "Probable inference", "Bonett-Wright": "Sample size requirements",
         "Fisher exact": "logic of inductive inference"}
for name, needle in TOOLS.items():
    if needle.lower() not in refs.lower():
        fails.append(f"tool citation missing from the list: {name}")
rows = []
for n in sorted(entries):
    e = entries[n]
    doi = re.search(r"doi:(10\.\S+?)(?:\s|$)", e)
    arx = re.search(r"arXiv:(\d{4}\.\d{4,5})", e)
    title = '"' in e or "“" in e
    if not title:
        fails.append(f"[{n}] has no title")
    if not (doi or arx):
        fails.append(f"[{n}] has neither DOI nor arXiv id")
    status = "skipped"
    if not OFFLINE and (doi or arx):
        url = (f"https://api.crossref.org/works/{doi.group(1).rstrip('.')}" if doi
               else f"https://export.arxiv.org/api/query?id_list={arx.group(1)}")
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                body = r.read().decode(errors="replace")
            status = "OK" if (doi and json.loads(body).get("status") == "ok") or (arx and "<entry>" in body) else "NO-MATCH"
        except Exception as ex:
            status = f"ERR {type(ex).__name__}"
        if status != "OK":
            fails.append(f"[{n}] identifier did not resolve: {status}")
    rows.append((n, status, (doi.group(1) if doi else "arXiv:" + arx.group(1)) if (doi or arx) else "-", e[:58]))
print(f"{N} references, {len(seq)} cited in order")
for n, st, ident, e in rows:
    print(f"  [{n:>2}] {st:<8} {ident:<34} {e}")
print("\n".join(fails) if fails else "\ncheck_refs: PASS")
sys.exit(1 if fails else 0)
