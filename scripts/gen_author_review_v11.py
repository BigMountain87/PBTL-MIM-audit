#!/usr/bin/env python3
"""Regenerate docs/author_review_v11_r2.md from the current manuscript (T84).

Finds every AUTHOR-SIGNOFF / TODO-AUTHOR / other AUTHOR marker in
paper/manuscript_v11.md with its CURRENT line number and the paragraph (or
list bullet) it sits in, so the review doc doesn't drift from stale line
numbers the way docs/author_review_v11.md (generated 2026-09-13) has since
T74-T81 added ~400 lines. Output is a draft the author_review doc is built
from by hand (section A/E framing still needs human judgment) — this script
only makes B/C/D mechanical and line-number-accurate.
"""
import re
from pathlib import Path

MANUSCRIPT = Path("paper/manuscript_v11.md")
COMMENT_RE = re.compile(r"<!--\s*(AUTHOR-SIGNOFF|AUTHOR-DECISION|AUTHOR):\s*(.*?)-->", re.S)
BULLET_START_RE = re.compile(r"^- \*\*\w")
HEADING_RE = re.compile(r"^#{1,4} ")


def paragraph_span(lines, marker_line_idx):
    """0-indexed. Expand from marker's line to its enclosing paragraph/bullet."""
    start = marker_line_idx
    while start > 0:
        prev = lines[start - 1]
        if prev.strip() == "" or HEADING_RE.match(prev) or BULLET_START_RE.match(lines[start]):
            break
        start -= 1
    end = marker_line_idx
    while end + 1 < len(lines):
        nxt = lines[end + 1]
        if nxt.strip() == "" or HEADING_RE.match(nxt) or BULLET_START_RE.match(nxt):
            break
        end += 1
    return start, end


def main():
    txt = MANUSCRIPT.read_text()
    lines = txt.split("\n")

    print("=" * 20, "AUTHOR-SIGNOFF (B section)", "=" * 20)
    for m in re.finditer(r"<!--\s*AUTHOR-SIGNOFF:\s*(.*?)-->", txt, re.S):
        tag_raw = m.group(1).strip()
        line_no = txt[: m.start()].count("\n") + 1
        idx = line_no - 1
        s, e = paragraph_span(lines, idx)
        para_lines = lines[s : e + 1]
        para = "\n".join(para_lines)
        para_clean = COMMENT_RE.sub("", para).strip()
        para_clean = re.sub(r"[ \t]+", " ", para_clean)
        tag_short = tag_raw.split(" — ")[0].split(" - ")[0].strip()
        note = tag_raw[len(tag_short):].strip(" —-")
        print(f"\n### `{tag_short}` — line {line_no}" + (f"  [HAS EMBEDDED NOTE: {note[:60]}...]" if note else ""))
        print(f"(paragraph/bullet spans source lines {s+1}-{e+1})")
        print("> " + para_clean.replace("\n", "\n> "))

    print("\n" + "=" * 20, "AUTHOR-DECISION (inline, not a keep/edit/drop item)", "=" * 20)
    for m in re.finditer(r"<!--\s*AUTHOR-DECISION:\s*(.*?)-->", txt, re.S):
        line_no = txt[: m.start()].count("\n") + 1
        print(f"line {line_no}: {m.group(1).strip()}")

    print("\n" + "=" * 20, "bare AUTHOR: markers", "=" * 20)
    for m in re.finditer(r"AUTHOR:\s*([^>\n-][^\n]*)", txt):
        line_no = txt[: m.start()].count("\n") + 1
        if "SIGNOFF" in lines[line_no - 1] or "DECISION" in lines[line_no - 1]:
            continue
        print(f"line {line_no}: {lines[line_no-1].strip()}")

    print("\n" + "=" * 20, "TODO-AUTHOR (C section)", "=" * 20)
    for m in re.finditer(r"TODO-AUTHOR", txt):
        line_no = txt[: m.start()].count("\n") + 1
        print(f"line {line_no}: {lines[line_no-1].strip()}")


if __name__ == "__main__":
    main()
