#!/usr/bin/env python3
"""T42 — build the MLST LaTeX submission from the Markdown sources.

    python3 scripts/md_to_latex_v10.py            # writes paper/mlst/{main,supplementary}.tex

Every number reaches the .tex through paper/manuscript_v10.md, which
scripts/check_manuscript_v10.py audits against the JSON artifacts.  Nothing here is
hand-edited afterwards: when the pub-arm numbers land, edit the Markdown and re-run.

MLST takes a format-free first submission (docs/mlst_requirements_v10.md), and IOP
does not publish iopart.cls on CTAN, so the build uses `article` with Times
(newtx).  The fallback is recorded in paper/mlst/SUBMISSION_CHECKLIST.md.
"""
from __future__ import annotations
import re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Sources are explicit inputs.  The converter was left pointing at the v10 manuscript and
# figures_v9 after its figure map moved to the six-figure pub numbering, so the two
# halves disagreed and a build would have raised KeyError on Figure 7 (codex full-scope
# review, 2026-09-12, slice 5).  Defaults are the pub arm; pass --manuscript/--supplement/
# --figures to build anything else, and the checks below derive from FIGFILE.
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--manuscript", default="paper/manuscript_v11.md")
_ap.add_argument("--supplement", default="paper/supplementary_v11.md")
_ap.add_argument("--figures", default="figures_v11")
_ap.add_argument("--build", action="store_true")
ARGS = _ap.parse_args()
MS, SI = ROOT / ARGS.manuscript, ROOT / ARGS.supplement
OUT, FIGSRC = ROOT / "paper/mlst", ROOT / ARGS.figures
BIB = OUT / "refs.bib"

# Caption number -> file. The pub arm has no order-7 figure, so make_figures_v9 emits
# nothing for it and the captions renumber: what the generator calls fig6/fig7 are
# Figures 5 and 6 here. The generator keeps its own names, which encode the legacy
# numbering, and this map is where the two conventions meet.
FIGFILE = {"1": "fig1_protocol_schematic", "2": "fig2_selfreport_vs_oracle",
           "3": "fig3_r_vs_reliability", "4": "fig4_tau_sensitivity",
           "5": "fig6_worst_pretender_spectrum", "6": "fig7_mechanism", "7": "fig8_pretender_summary",
           "S1": "figS1_box_edge_geometry", "S2": "figS2_heldout_vs_rcwa"}
# end matter is unnumbered, and 'References' is built from the Markdown reference list
UNNUMBERED = ["Code, Data, and Compute", "Acknowledgements", "Data availability",
              "Conflict of interest", "Author contributions", "ORCID iDs"]

# characters that appear in the sources, mapped for pdflatex
UNI = {
    "—": r"---", "–": r"--", "−": r"\ensuremath{-}", "…": r"\ldots{}",
    "×": r"\ensuremath{\times}", "±": r"\ensuremath{\pm}", "·": r"\ensuremath{\cdot}",
    "≤": r"\ensuremath{\leq}", "≈": r"\ensuremath{\approx}", "≥": r"\ensuremath{\geq}", "∈": r"\ensuremath{\in}",
    "→": r"\ensuremath{\rightarrow}", "⇒": r"\ensuremath{\Rightarrow}",
    "‖": r"\ensuremath{\|}", "§": r"\S{}", "†": r"\dag{}",
    "τ": r"\ensuremath{\tau}", "Δ": r"\ensuremath{\Delta}", "ρ": r"\ensuremath{\rho}", "θ": r"\ensuremath{\theta}",
    "α": r"\ensuremath{\alpha}",
    "λ": r"\ensuremath{\lambda}", "σ": r"\ensuremath{\sigma}", "ε": r"\ensuremath{\varepsilon}",
    "¹": r"\textsuperscript{1}", "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}",
    "⁴": r"\textsuperscript{4}", "⁻": r"\textsuperscript{\ensuremath{-}}",
    "⁰": r"\textsuperscript{0}", "⁵": r"\textsuperscript{5}", "⁶": r"\textsuperscript{6}",
    "⁷": r"\textsuperscript{7}", "⁸": r"\textsuperscript{8}", "⁹": r"\textsuperscript{9}",
    "₂": r"\textsubscript{2}", "č": r"\v{c}", "ć": r"\'c", "á": r"\'a",
}


def sh(cmd, stdin=None):
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stderr}")
    return r.stdout


def pandoc(md, shift=0):
    """Markdown -> LaTeX.  `shift=-1` maps '##' onto \\section (the sources start at '##')."""
    cmd = ["pandoc", "-f", "markdown+raw_tex-auto_identifiers-smart",
           "-t", "latex", "--wrap=preserve", "--no-highlight"]
    if shift:
        cmd.append("--shift-heading-level-by=%d" % shift)
    return sh(cmd, md).strip()


def strip_comments(t):
    """Drop the author-facing comments without touching the line structure: eating the
    newline after a trailing comment glues the next heading onto the paragraph."""
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = "\n".join(l.rstrip() for l in t.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", t)


# ---------------------------------------------------------------- citations
def cite_map():
    """bibtex key by reference number, from the `note = {ref N}` field T34 wrote."""
    m = {}
    for entry in BIB.read_text().split("\n@")[0:]:
        k = re.search(r"^@?\w+\{([^,]+),", entry)
        n = re.search(r"note\s*=\s*\{ref (\d+)\}", entry)
        if k and n:
            m[int(n.group(1))] = k.group(1)
    return m


def citations(txt, keys, report):
    """[12], [12, 13] and [13-16] become \\cite; intervals and [CHECK] are left alone."""
    def sub(m):
        s = m.group(1)
        if not re.fullmatch(r"\d{1,2}(\s*[,\u2013-]\s*\d{1,2})*", s):
            return m.group(0)
        nums = [int(x) for x in re.split(r"[,\u2013-]", s) if x.strip()]
        if not all(n in keys for n in nums):
            return m.group(0)
        if "\u2013" in s or (len(nums) == 2 and "-" in s):        # a range
            nums = list(range(nums[0], nums[-1] + 1))
        report.append(s)
        return "\\cite{" + ",".join(keys[n] for n in nums) + "}"
    return re.sub(r"\[([^\]\[]{1,40})\]", sub, txt)


# ---------------------------------------------------------------- floats
LINEWIDTH = 455.0          # pt, A4 with 2.5 cm margins
LANDSCAPE = 700.0          # pt, the same page rotated
SIZES = [("", 5.0, 6.0), (r"\small", 4.6, 6.0), (r"\footnotesize", 4.2, 3.0),
         (r"\scriptsize", 3.6, 3.0), (r"\tiny", 3.0, 2.0)]


def fits(width, ncol, maxwidth):
    """The largest type size whose `l` columns fit in maxwidth, or None."""
    for size, cpt, sep in SIZES:
        if sum(w * cpt for w in width) + 2 * ncol * sep <= maxwidth:
            return ("" if sep == 6.0 else "\\setlength{\\tabcolsep}{%.0fpt}" % sep) + size
    return None


def col_spec(rows, ncol, maxwidth=None):
    """`l` columns at the largest type size the table fits in, else wrapped p{} columns
    in proportion to the widest cell.  Guessing wrong here is what produced 1499 overfull
    boxes in the supplement: a 17-column table cannot be laid out at \linewidth/17."""
    width = [max(len(r[i]) for r in rows) for i in range(ncol)]
    pre = fits(width, ncol, maxwidth or LINEWIDTH)
    if pre is not None:
        return "@{}" + "l" * ncol + "@{}", pre, True
    # p{} widths in proportion to the widest cell, but never narrower than the longest
    # single word: a word that does not fit cannot be broken and runs into the next column.
    cpt, sep = 4.8, 3.0
    avail = LINEWIDTH - 2 * ncol * sep
    word = [max((max((len(w) for w in r[i].split()), default=1)) for r in rows) * cpt / avail
            for i in range(ncol)]
    if sum(word) > 1:                 # the floors cannot all be met; widths follow them
        word = [w / sum(word) for w in word]
    frac = [w / sum(width) for w in width]
    for _ in range(4):
        short = [i for i in range(ncol) if frac[i] < word[i]]
        if not short:
            break
        spare = 1 - sum(word[i] for i in short)
        rest = sum(frac[i] for i in range(ncol) if i not in short) or 1.0
        frac = [word[i] if i in short else frac[i] / rest * spare for i in range(ncol)]
    body = "\n".join(r"  >{\RaggedRight\arraybackslash}p{(\linewidth - %d\tabcolsep) * \real{%.4f}}"
                     % (2 * ncol, f) for f in frac)
    return "@{}\n" + body + "@{}", r"\setlength{\tabcolsep}{3pt}\footnotesize", False


def table_tex(md_table, caption_md, number, kind="tab", long=False):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")]
            for l in md_table.strip().split("\n") if not re.fullmatch(r"\|[-:| ]+\|", l.strip())]
    ncol = len(rows[0])
    spec, size, plain = col_spec(rows, ncol)
    # a table too wide for the page but not for a rotated one reads far better rotated
    land = long and not plain and \
        fits([max(len(r[i]) for r in rows) for i in range(ncol)], ncol, LANDSCAPE) is not None
    if land:
        spec, size, plain = col_spec(rows, ncol, LANDSCAPE)
    lt = pandoc(md_table)
    cap = "\\caption{" + pandoc(caption_md) + "}\n\\label{%s:%s}" % (kind, number)
    if long:
        # a float that does not fit on a page is dropped without an error, and the SI has
        # a 182-row table: caption outside, longtable inside, nothing to lose.  pandoc's
        # \def\LTcaptype{none} wrapper stays, so only \captionof steps the counter.
        lt = re.sub(r"\\begin\{longtable\}\[\]\{.*?@\{\}\}",
                    lambda m: "\\begin{longtable}[]{" + spec + "}", lt, count=1, flags=re.S)
        if "l" in spec.replace("@{}", ""):
            lt = re.sub(r"\\begin\{minipage\}\[b\]\{\\linewidth\}\\raggedright\s*", "", lt)
            lt = re.sub(r"\s*\\end\{minipage\}", "", lt)
        out = ("\\begingroup\n" + (size + "\n" if size else "") +
               cap.replace("\\caption{", "\\captionof{table}{") + "\n" + lt + "\n\\endgroup")
        return "\\begin{landscape}\n%s\n\\end{landscape}" % out if land else out
    lt = re.sub(r"^\{\\def\\LTcaptype[^\n]*\n", "", lt).rstrip().rstrip("}").rstrip()
    lt = re.sub(r"\\begin\{longtable\}\[\]\{.*?@\{\}\}\n", "", lt, flags=re.S)
    if plain:
        lt = re.sub(r"\\begin\{minipage\}\[b\]\{\\linewidth\}\\raggedright\s*", "", lt)
        lt = re.sub(r"\s*\\end\{minipage\}", "", lt)
    lt = lt.replace("\\endhead\n", "").replace("\\endlastfoot\n", "")
    lt = re.sub(r"\\bottomrule\\noalign\{\}\n", "", lt)
    lt = lt.replace("\\end{longtable}", "").rstrip()
    return ("\\begin{table}[htbp]\n\\centering\n" + (size + "\n" if size else "") +
            cap + "\n\\begin{tabular}{" + spec + "}\n" + lt +
            "\n\\bottomrule\\noalign{}\n\\end{tabular}\n\\end{table}")


def figure_tex(number, caption_md):
    return ("\\begin{figure}[htbp]\n\\centering\n"
            "\\includegraphics{figures/%s.pdf}\n" % FIGFILE[number] +
            "\\caption{" + pandoc(caption_md) + "}\n\\label{fig:%s}\n\\end{figure}" % number)


def extract_floats(md, kind_prefix=""):
    """Pull figure blockquotes and captioned tables out, leaving @@F1@@ / @@T1@@ markers."""
    lines, out, floats = md.split("\n"), [], {}
    i = 0
    while i < len(lines):
        l = lines[i]
        m = re.match(r"^> \*\*Figure (S?\d+)\.\*\* ?(.*)", l)
        if m:                                     # blockquote figure caption
            num, cap = m.group(1), [m.group(2)]
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                cap.append(lines[i].lstrip("> ").rstrip())
                i += 1
            floats["F" + num] = figure_tex(num, " ".join(c for c in cap if c))
            out.append("@@F%s@@" % num)
            continue
        m = re.match(r"^\*\*Table (S?\d+)\. (.*)", l)
        if m:                                     # caption paragraph, then the table
            num, cap = m.group(1), [m.group(2)]
            i += 1
            while i < len(lines) and lines[i].strip():
                cap.append(lines[i].rstrip())
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            assert lines[i].startswith("|"), f"table {num}: no table after the caption"
            tbl = []
            while i < len(lines) and lines[i].startswith("|"):
                tbl.append(lines[i]); i += 1
            cap_md = " ".join(cap).replace("**", "", 1)
            floats["T" + num] = table_tex("\n".join(tbl), cap_md, num)
            out.append("@@T%s@@" % num)
            continue
        out.append(l); i += 1
    return "\n".join(out), floats


def uncaptioned_tables(md):
    """SI tables carry no caption line: number them in order and caption each with the
    supplementary section it sits in (the main text cites Tables S1-S3, which are the
    first three)."""
    lines, out, floats, n, sec, sub = md.split("\n"), [], {}, 0, "", ""
    i = 0
    while i < len(lines):
        h = re.match(r"^## (S\d+\. .*)$", lines[i])
        if h:
            sec, sub = h.group(1), ""
        # a numbered heading inside a section (the verbatim protocol of S4) names the
        # table that follows it, so two tables in one section get distinct captions
        h2 = re.match(r"^#{2,4} (\d+(?:\.\d+)?\.? .*)$", lines[i])
        if h2 and not h:
            sub = h2.group(1)
        if lines[i].startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].startswith("|"):
                tbl.append(lines[i]); i += 1
            n += 1
            floats["T%d" % n] = table_tex("\n".join(tbl), sec + (" — protocol §" + sub if sub else ""), str(n), long=True)
            out.append("@@T%d@@" % n)
            continue
        out.append(lines[i]); i += 1
    return "\n".join(out), floats


# ---------------------------------------------------------------- assembly
def unicode_pass(tex):
    for k, v in UNI.items():
        tex = tex.replace(k, v)
    return tex


def clean(tex, floats):
    for k, v in floats.items():
        tex = tex.replace("@@%s@@" % k, v)
    for h in UNNUMBERED:
        tex = tex.replace("\\section{%s}" % h, "\\section*{%s}" % h)
    tex = tex.replace("^\\*", "^{*}")                      # \* is not a math star
    tex = re.sub(r"\\hypertarget\{[^}]*\}\{%\n", "", tex)
    tex = re.sub(r"\\textsuperscript\{(\d)\},\\textsuperscript\{(\d)\}",
                 r"\\textsuperscript{\1,\2}", tex)
    tex = re.sub(r"\\texttt\{([0-9a-f]{24,})\}", r"\\texttt{\\seqsplit{\1}}", tex)

    def breakable(m):                       # long file names are otherwise unbreakable
        s = m.group(1)
        if len(s) < 18 or " " in s or "seqsplit" in s:
            return m.group(0)
        return "\\texttt{%s}" % re.sub(r"(?<=[_,/.\-])(?!$)", "\\\\allowbreak{}", s)
    tex = re.sub(r"\\texttt\{((?:[^{}]|\{[^{}]*\})*)\}", breakable, tex)
    tex = tex.replace("\\label{sec:", "\\label{sec:")
    return unicode_pass(tex)


def bibliography(refs_md, keys):
    items = re.findall(r"^\[(\d+)\] (.*?)(?=\n\[\d+\] |\Z)", refs_md.strip(), re.S | re.M)
    assert len(items) == len(keys), f"{len(items)} reference entries vs {len(keys)} bib keys"
    body = []
    for num, text in items:
        body.append("\\bibitem{%s}\n%s" % (keys[int(num)], pandoc(" ".join(text.split()))))
    return ("\\begin{thebibliography}{%d}\n" % len(items) +
            "\n\n".join(body) + "\n\\end{thebibliography}")


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
% MLST accepts a format-free first submission and IOP does not distribute
% iopart.cls on CTAN, so this is the article-class fallback of docs/mlst_requirements_v10.md.
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}   % after amsmath; newtxmath supplies the AMS symbols
\usepackage{graphicx}
\usepackage{booktabs,longtable,array,calc}
\usepackage{pdflscape}
\usepackage{ragged2e}   % \RaggedRight still hyphenates, unlike \raggedright
\usepackage{caption}
\usepackage[margin=2.5cm]{geometry}
\usepackage{microtype}
\usepackage{seqsplit}   % SHA-256 strings are otherwise unbreakable
\usepackage{newunicodechar}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false}
\setlength{\parskip}{0pt}
\newcounter{none}   % pandoc's \def\LTcaptype{none} for an uncaptioned longtable
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\renewcommand{\arraystretch}{1.15}
"""


def main():
    keys = cite_map()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for stem in FIGFILE.values():
        shutil.copy(FIGSRC / (stem + ".pdf"), OUT / "figures" / (stem + ".pdf"))

    # ---------------- main text
    t = strip_comments(MS.read_text())
    title = re.match(r"# (.*)", t).group(1)
    head, rest = t.split("## Abstract", 1)
    authors = [l for l in head.split("\n")[1:] if l.strip()]
    abstract_md, body_md = rest.split("---", 1)
    abstract_md, keywords = abstract_md.split("**Keywords:**")
    body_md, refs_md = body_md.split("## References", 1)

    body_md = re.sub(r"^(#{2,3}) [\d.]+\.? (.*)$",
                     lambda m: "%s %s {#sec:%s}" % (m.group(1), m.group(2),
                                                    re.match(r"^#{2,3} ([\d.]+)", m.group(0)).group(1).rstrip(".")),
                     body_md, flags=re.M)
    body_md, floats = extract_floats(body_md)
    report = []
    body_md = citations(body_md, keys, report)
    abstract_md = citations(abstract_md, keys, report)
    body = clean(pandoc(body_md, shift=-1), floats)

    tex = [PREAMBLE]
    for ch, rep in UNI.items():
        tex.append("\\newunicodechar{%s}{%s}" % (ch, rep))
    # \maketitle centres the author block in unbreakable lines; the affiliations are
    # longer than \textwidth, so the front matter is set by hand.
    names, affil = authors[0], [a for a in authors[1:] if a.strip() != "---"]
    front = ["\\begin{center}",
             "{\\LARGE\\bfseries " + unicode_pass(pandoc(title)) + "\\par}",
             "\\vspace{1.2em}",
             "{\\large " + unicode_pass(pandoc(names)).replace("\\textbf{", "{") + "\\par}",
             "\\vspace{0.8em}", "{\\small\\itshape"] + \
            [unicode_pass(pandoc(a)) + "\\par" for a in affil[:-1]] + \
            ["}", "\\vspace{0.6em}",
             "{\\small " + unicode_pass(pandoc(affil[-1])) + "\\par}",
             "\\end{center}"]
    tex += [r"\begin{document}"] + front + [
            r"\begin{abstract}", unicode_pass(pandoc(abstract_md.strip())), r"\end{abstract}",
            r"\noindent\textbf{Keywords:} " + unicode_pass(pandoc(keywords.strip())).replace("\n", " "),
            body, r"\section*{References}",
            unicode_pass(bibliography(refs_md, keys)), r"\end{document}"]
    (OUT / "main.tex").write_text("\n\n".join(tex) + "\n")

    # ---------------- supplementary
    s = strip_comments(SI.read_text())
    s_title = re.match(r"# (.*)", s).group(1)
    s_body = s.split("\n", 1)[1]
    # S4 reproduces the frozen protocol inside a ```markdown fence.  Verbatim in the PDF
    # would overflow every line, so it is typeset, set apart by \small.
    fences = [l for l in s_body.split("\n") if l.startswith("```")]
    assert fences == ["```markdown", "```"], f"unexpected fenced blocks: {fences}"
    # Protect the frozen protocol from reference-number conversion. Its Fourier
    # orders [5,5] and [7,7] are numeric arrays, not bibliography citations.
    protocol_head, protocol_tail = s_body.split("```markdown", 1)
    protocol, protocol_after = protocol_tail.split("```", 1)
    # HTML entities decode in prose, but would print literally inside code spans.
    protocol = "".join(part if i % 2 else re.sub(r"\[([^\]\n]+)\]", r"&#91;\1&#93;", part)
                       for i, part in enumerate(re.split(r"(`[^`\n]*`)", protocol)))
    s_body = protocol_head + "```markdown" + protocol + "```" + protocol_after
    s_body = re.sub(r"^```markdown$", lambda m: "\n\\begingroup\\small\n", s_body, flags=re.M)
    s_body = re.sub(r"^```$", lambda m: "\n\\endgroup\n", s_body, flags=re.M)
    s_body = re.sub(r"^# (.*)$", r"### \1", s_body, flags=re.M)      # protocol headings (S4)
    s_body = re.sub(r"^## (S\d+)\. (.*)$", r"## \1. \2", s_body, flags=re.M)
    s_body, s_floats = extract_floats(s_body)
    s_body2, s_tabs = uncaptioned_tables(s_body)
    s_floats.update(s_tabs)
    s_body2 = citations(s_body2, keys, report)
    s_tex = clean(pandoc(s_body2, shift=-1), s_floats)
    s_tex = s_tex.replace("\\^{}", "\\textasciicircum{}")   # a literal caret ([0,1]^d), not an accent
    s_tex = re.sub(r"\\(sub)*section\{", lambda m: "\\%ssection*{" % (m.group(1) or ""), s_tex)
    stex = [PREAMBLE]
    for ch, rep in UNI.items():
        stex.append("\\newunicodechar{%s}{%s}" % (ch, rep))
    stex += [r"\renewcommand{\thetable}{S\arabic{table}}",
             r"\renewcommand{\thefigure}{S\arabic{figure}}",
             r"\begin{document}", r"\title{%s}" % unicode_pass(pandoc(s_title)),
             r"\author{}", r"\date{}", r"\maketitle", s_tex,
             unicode_pass(bibliography(refs_md, keys)), r"\end{document}"]
    (OUT / "supplementary.tex").write_text("\n\n".join(stex) + "\n")

    print(f"main.tex        {len(floats)} floats, {len(keys)} references, "
          f"{len(report)} citations converted")
    print(f"supplementary   {len(s_floats)} floats")
    if ARGS.build:
        build()


def build():
    """latexmk both documents, then check that LaTeX numbered every section, table and
    figure exactly as the Markdown does -- a heading lost in conversion shows up here."""
    ok = True
    for stem in ("main", "supplementary"):
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", stem + ".tex"],
                           cwd=OUT, capture_output=True, text=True)
        log = (OUT / (stem + ".log")).read_text(errors="replace")
        und = [l for l in log.split("\n") if "undefined" in l.lower()]
        print(f"[{stem}] latexmk exit {r.returncode}; {len(und)} undefined; "
              f"{len(re.findall(chr(92)+chr(92)+'includegraphics', (OUT/(stem+'.tex')).read_text()))} figures")
        ok &= r.returncode == 0 and not und
        for l in und[:5]:
            print("   ", l.strip())
        bad = [(k, v) for k, v in re.findall(r"\\newlabel\{(\w+:[\w.]+)\}\{\{([^}]*)\}",
                                           (OUT / (stem + ".aux")).read_text())
               if k.split(":")[1].lstrip("S") != v.lstrip("S")]
        if bad:
            ok = False
            print(f"   [{stem}] numbering differs from the Markdown: {bad}")
    txt = sh(["pdftotext", str(OUT / "main.pdf"), "-"])
    checks = {
        "keywords": "Keywords:" in txt,
        f"{sum(1 for k in FIGFILE if k.isdigit())} figures": all("Figure %d" % i in txt for i in range(1, sum(1 for k in FIGFILE if k.isdigit()) + 1)),
        "12 tables": all("Table %d" % i in txt for i in range(1, 13)),
        "numbered references": "[57]" in txt or "57." in txt.split("References")[-1],
        "end matter in order": [h for h in UNNUMBERED if h in txt] == UNNUMBERED,
        "no TeX escapes left": "\\textbackslash" not in txt and "@@" not in txt,
    }
    for k, v in checks.items():
        print(f"   main.pdf {k}: {'ok' if v else 'MISSING'}")
    ok &= all(checks.values())
    print("build:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
