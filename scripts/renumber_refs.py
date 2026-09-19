#!/usr/bin/env python3
"""T34 — renumber the reference list by first citation in the body.

Scans everything before '## References', extracts bracket citations in reading order
(handling '[8–11]', '[8-11]', '[3, 5]', '[NEW-key]' and mixed lists), maps every old key
onto a sequential number, then rewrites both the body citations and the list. Interval
brackets ([0,1], [50,400], anything containing a decimal point) are ignored. Keys that
resolve to an existing numbered entry via a 'see [N].' stub are merged onto that entry.
Usage: python3 scripts/renumber_refs.py paper/manuscript_v10.md [--dry-run]
"""
import re, shutil, sys
from pathlib import Path

M = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/manuscript_v10.md")
DRY = "--dry-run" in sys.argv
txt = M.read_text()
head, sep, refs = txt.partition("\n## References")
assert sep, "no ## References section"

# ---- parse the list: key -> entry text (key is a number or a NEW-* name)
entries, order = {}, []
for m in re.finditer(r"^\[(NEW-[a-z0-9]+|\d+)\]\s*(.*?)(?=\n\[(?:NEW-[a-z0-9]+|\d+)\]|\Z)", refs, re.S | re.M):
    key, body = m.group(1), m.group(2).strip()
    entries[key] = body; order.append(key)
# 'see [N].' stubs merge onto the existing number
alias = {k: re.match(r"see \[(\d+)\]\.?$", v).group(1) for k, v in entries.items() if re.match(r"see \[\d+\]\.?$", v)}
for k in alias:
    del entries[k]

MAXNUM = len(entries)
TOKEN = re.compile(r"\[((?:NEW-[a-z0-9]+|\d+)(?:\s*[–\-,;]\s*(?:NEW-[a-z0-9]+|\d+))*)\]")


MAXNUM = None   # set after the list is parsed: numbers above it are intervals, not citations


def keys_of(inner):
    """Expand one bracket's content into the keys it cites, or None if it is not a citation.
    Rejects interval brackets: anything with a decimal point, and any number larger than the
    reference count (e.g. [0,1], [50,400], [5,5])."""
    if "." in inner:
        return None
    out = []
    for part in re.split(r"\s*[,;]\s*", inner):
        rng = re.match(r"^(\d+)\s*[–\-]\s*(\d+)$", part)
        if rng:
            a, b = int(rng.group(1)), int(rng.group(2))
            if b <= a or b - a > 12 or (MAXNUM is not None and b > MAXNUM):
                return None
            out += [str(i) for i in range(a, b + 1)]
        elif re.match(r"^(NEW-[a-z0-9]+|\d+)$", part):
            if part.isdigit() and MAXNUM is not None and int(part) > MAXNUM:
                return None
            out.append(part)
        else:
            return None
    return [alias.get(k, k) for k in out] or None


# ---- first-citation order over the body
seq, seen = [], set()
for m in TOKEN.finditer(head):
    ks = keys_of(m.group(1))
    if not ks:
        continue
    for k in ks:
        if k in entries and k not in seen:
            seen.add(k); seq.append(k)
missing = [k for k in entries if k not in seen]
new_no = {k: i + 1 for i, k in enumerate(seq)}


def sub(m):
    ks = keys_of(m.group(1))
    if not ks:
        return m.group(0)
    nums = sorted({new_no[k] for k in ks if k in new_no})
    if not nums:
        return m.group(0)
    # collapse contiguous runs of three or more
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return "[" + ", ".join(parts) + "]"


new_head = TOKEN.sub(sub, head)
new_refs = "\n## References\n\n" + "\n\n".join(f"[{new_no[k]}] {entries[k]}" for k in seq) + "\n"
print(f"{len(entries)} entries, {len(seq)} cited, {len(missing)} uncited: {missing}")
if DRY:
    sys.exit(0 if not missing else 1)
shutil.copy(M, M.with_suffix(".md.bak"))
M.write_text(new_head + new_refs)
print(f"renumbered -> {len(seq)} references (backup {M.with_suffix('.md.bak').name})")
sys.exit(1 if missing else 0)
