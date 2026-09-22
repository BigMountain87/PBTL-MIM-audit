"""Build the arXiv preprint package from the single-source manuscript.

Source of truth stays paper/manuscript_v11.md and paper/supplementary_v11.md; this script
runs scripts/md_to_latex_v10.py to regenerate paper/mlst/{main,supplementary}.tex, then
derives paper/preprint/ from them:
  * a "Preprint" line above the corresponding-author line,
  * the ORCID section removed (journal-only; iDs pending),
  * an assertion that no TODO-AUTHOR / AUTHOR-SIGNOFF text survives into the PDF,
  * latexmk of the merged document, with the supplement appended after the references,
  * arxiv_v1.tar.gz (main.tex + figures/*.pdf), the metadata
    abstract (arXiv caps it at 1920 characters) and a metadata sheet.

    python3 scripts/build_preprint_v11.py [--version 1] [--date 2026-09-19]
"""
import argparse, datetime, os, re, shutil, subprocess, sys, tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "paper/mlst", ROOT / "paper/preprint"
ap = argparse.ArgumentParser()
ap.add_argument("--version", type=int, default=1)
ap.add_argument("--date", default=datetime.date.today().isoformat())
a = ap.parse_args()

subprocess.run([sys.executable, "scripts/make_fig_schematic_v9.py"], cwd=ROOT,
               env={**os.environ, "INVERSETL_PROFILE": "pub",
                    "INVERSETL_FIGURES_DIR": "figures_v11"}, check=True)
r = subprocess.run([sys.executable, "scripts/md_to_latex_v10.py"], cwd=ROOT, capture_output=True, text=True)
print(r.stdout.strip()); assert r.returncode == 0, r.stderr
OUT.mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
for f in (SRC / "figures").glob("*.pdf"):
    shutil.copy(f, OUT / "figures" / f.name)

main = (SRC / "main.tex").read_text()
pre = r"{\small\itshape Preprint v%d, %s. This version has not been peer reviewed.\par}" % (a.version, a.date)
main, n = re.subn(r"(\{\\small Corresponding authors:)", lambda m: pre + "\n" + m.group(1), main); assert n == 1
main, n = re.subn(r"\\section\*\{ORCID iDs\}.*?(?=\\begin\{thebibliography\})", "", main, flags=re.S); assert n == 1
for bad in ("TODO-AUTHOR", "AUTHOR-SIGNOFF", "AUTHOR-DECISION", "PENDING"):
    assert bad not in main, f"{bad} would be printed in the preprint"
supp = (SRC / "supplementary.tex").read_text()
for bad in ("TODO-AUTHOR", "AUTHOR-SIGNOFF"):
    assert bad not in supp, f"{bad} in supplement"
# append the supplement to the main document (arXiv ancillary files are not indexed); its
# citations resolve against the main bibliography, its floats renumber S1.., labels are prefixed
body = supp.split("\\maketitle\n", 1)[1].split("\\begin{thebibliography}", 1)[0]
body = re.sub(r"\\label\{(tab|fig):", r"\\label{s\1:", body); body = re.sub(r"\\ref\{(tab|fig):", r"\\ref{s\1:", body)
# a heading directly followed by a landscape table would sit alone on a page: move it inside
body, n_ls = re.subn(r"(\\(?:sub)*section\*\{(?:(?!\\(?:sub)*section\*).)*?\})\n\n\\begin\{landscape\}", r"\\begin{landscape}\n\1\n", body, flags=re.S)
# S8 tables inherit the S8 heading as caption: use the subsection title instead
def _s8(m):
    title = re.sub(r"^S8\.\d\. ", "", m.group(1)); return re.sub(r"\\captionof\{table\}\{(?:S8\. )?The as-submitted arm of Section 3\.10\}", lambda _: "\\captionof{table}{" + title + "}", m.group(0))
body, n_s8 = re.subn(r"\\subsection\*\{(S8\.\d\. [^}]*)\}(?:(?!\\subsection\*).)*?\\captionof\{table\}\{(?:S8\. )?The as-submitted arm of Section 3\.10\}", _s8, body, flags=re.S)
print(f"supplement: {n_s8} S8 captions renamed")
# section-derived captions ("S6. Per-design table") -> descriptive ("Per-design table")
body, n_cap = re.subn(r"(\\caption(?:of\{table\})?\{)S\d+(?:\.\d+)?\. ", r"\1", body)
print(f"supplement: {n_ls} headings moved into landscape, {n_cap} captions cleaned")
# Keep the supplement title and provenance note with its first landscape table.
intro, first_landscape = body.split("\\begin{landscape}", 1)
body = ("\\begin{landscape}\n\\section*{Supplementary Information}\n"
        + intro + first_landscape)
app = ("\n\\clearpage\n\\setcounter{table}{0}\\setcounter{figure}{0}\n"
       "\\renewcommand{\\thetable}{S\\arabic{table}}\\renewcommand{\\thefigure}{S\\arabic{figure}}\n"
       + body)
assert main.count("\\end{document}") == 1
main = main.replace("\\end{document}", app + "\n\\end{document}")
(OUT / "main.tex").write_text(main)

ok = True
for stem in ("main",):
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", stem + ".tex"], cwd=OUT, capture_output=True, text=True)
    log = (OUT / f"{stem}.log").read_text(errors="replace")
    err = len(re.findall(r"^! ", log, flags=re.M)); und = len([l for l in log.split("\n") if "undefined" in l.lower()])
    pages = re.search(r"Output written on %s\.pdf \((\d+) pages" % stem, log)
    print(f"[{stem}] latexmk exit {r.returncode}; errors {err}; undefined {und}; pages {pages.group(1) if pages else '?'}")
    ok &= r.returncode == 0 and err == 0 and und == 0
assert ok, "LaTeX build failed"

# arXiv metadata: title, abstract (<= 1920 chars), authors, categories
md = (ROOT / "paper/manuscript_v11.md").read_text()
title = md.split("\n", 1)[0].lstrip("# ").strip()
authors = ", ".join(re.sub(r"[¹²³⁴,]+$", "", x.strip()) for x in re.search(r"^\*\*(.+?)\*\*$", md, flags=re.M).group(1).split(", "))
ab = re.sub(r"<!--.*?-->", "", md.split("## Abstract")[1].split("\n## ")[0], flags=re.S)
ab = ab.split("Keywords:")[0]                      # the keyword line is not part of the abstract
ab = re.sub(r"\s+", " ", ab).strip().replace("*", "").rstrip("- ").strip()
note = f"({len(ab)} chars; arXiv limit 1920)"
if len(ab) > 1920:
    # drop the tolerance-sweep sentence for the metadata only; the PDF keeps the full abstract
    ab_short = re.sub(r"The rate depends on the tolerance demanded:.*?of 179\.\s*", "", ab)
    note = f"(PDF abstract {len(ab)} chars > 1920; metadata abstract drops the tolerance-sweep sentence -> {len(ab_short)} chars)"
    ab = ab_short
assert len(ab) <= 1920, len(ab)
ab = ab.replace(" → ", " to ").replace("→", "to").replace("—", "-").replace("–", "-")   # arXiv metadata is ASCII
note = re.sub(r"^\(\d+ chars", f"({len(ab)} chars", note)
(OUT / "arxiv_abstract.txt").write_text(ab + "\n")
npages = re.search(r"Output written on main\.pdf \((\d+) pages", (OUT / "main.log").read_text(errors="replace")).group(1)
(OUT / "arxiv_metadata.md").write_text(f"""# arXiv submission metadata — v{a.version} ({a.date})

- **Title:** {title}
- **Authors:** {authors}
- **Primary category:** cs.LG · **Cross-list:** physics.optics, physics.comp-ph
- **License:** arXiv.org perpetual, non-exclusive license 1.0 (author decision 2026-09-20: journal submission unaffected; any other reuse of the text and figures needs the authors' consent; the code (MIT) and data (CC BY 4.0) licenses of the public repository are unchanged)
- **Comments:** {npages} pages including the supplementary information (Sections S1-S8 appended after the references); 7 + 2 figures, 12 + 10 tables
- **Abstract:** `arxiv_abstract.txt` {note}
- **Files:** `arxiv_v{a.version}.tar.gz` = main.tex + figures/*.pdf (bibliography embedded; pdflatex; single document)
- **Planned for v2:** ORCID iDs, archival DOI, additional raster re-solves
""")
def _anon(ti):                                   # no account names or ids in the archive headers
    ti.uid = ti.gid = 0; ti.uname = ti.gname = ""; return ti
with tarfile.open(OUT / f"arxiv_v{a.version}.tar.gz", "w:gz") as tar:
    tar.add(OUT / "main.tex", arcname="main.tex", filter=_anon)
    for f in sorted((OUT / "figures").glob("*.pdf")):
        tar.add(f, arcname=f"figures/{f.name}", filter=_anon)
(OUT / ".gitignore").write_text("*.aux\n*.fdb_latexmk\n*.fls\n*.log\n*.out\n*.toc\n")
print("preprint package:", OUT / f"arxiv_v{a.version}.tar.gz", note)
