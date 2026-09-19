#!/usr/bin/env python3
"""Per-section word counts for paper/manuscript_v11.md (T83, length-reduction planning).

Counts prose words per heading span (## and ###), excluding HTML comments and
Markdown table rows (lines starting with `|`). Table/figure captions and body
prose are counted; this is a planning estimate, not a submission-gate checker
(cf. check_manuscript_v10.py, which only checks the abstract's word count).

Usage: python3 scripts/wordcount_v11.py [path/to/manuscript.md]
"""
import re
import sys
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.M)
SI_MOVE_CANDIDATES = ["2.7", "3.5", "3.10", "3.11", "3.12"]
END_MATTER_TITLES = {
    "Code, Data, and Compute", "Acknowledgements", "Data availability",
    "Conflict of interest", "Author contributions", "ORCID iDs",
}


def strip_noise(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = [ln for ln in text.split("\n") if not re.match(r"^\s*\|", ln)]
    return "\n".join(lines)


def word_count(text: str) -> int:
    return len(strip_noise(text).split())


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/manuscript_v11.md")
    txt = path.read_text()
    matches = list(HEADING_RE.finditer(txt))
    rows = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        rows.append((level, title, word_count(txt[start:end])))

    # drop the level-1 document title (+ author/affiliation block before Abstract) —
    # not part of the word-limited body under any journal convention
    rows = [r for r in rows if r[0] > 1]

    print(f"{'':<4}{'section':<72}{'words':>8}")
    for level, title, wc in rows:
        indent = "  " * (level - 2)
        print(f"{'#'*level:<4}{indent}{title:<{72-len(indent)}}{wc:>8}")

    ref_idx = next(i for i, (_, t, _) in enumerate(rows) if t == "References")
    body_rows = [r for r in rows[:ref_idx] if r[1] not in END_MATTER_TITLES]
    end_rows = [r for r in rows[:ref_idx] if r[1] in END_MATTER_TITLES]
    refs_wc = rows[ref_idx][2]

    body_total = sum(wc for _, _, wc in body_rows)
    end_total = sum(wc for _, _, wc in end_rows)

    print(f"\n{'TOTAL body (Abstract + Sec.1-5, no end matter/refs)':<80}{body_total:>8}")
    print(f"{'TOTAL end matter (Code/Data..ORCID)':<80}{end_total:>8}")
    print(f"{'TOTAL References':<80}{refs_wc:>8}")
    print(f"{'GRAND TOTAL (all headings, incl. refs)':<80}{body_total+end_total+refs_wc:>8}")

    print("\n--- Option 1 candidates: sections proposed to move to SI ---")
    si_total = 0
    for _, title, wc in body_rows:
        if any(title == n or title.startswith(f"{n}.") for n in SI_MOVE_CANDIDATES):
            print(f"  {title:<70}{wc:>8}")
            si_total += wc
    print(f"  {'subtotal (moves out of body)':<70}{si_total:>8}")
    print(f"  {'body total AFTER move':<70}{body_total - si_total:>8}")


if __name__ == "__main__":
    main()
