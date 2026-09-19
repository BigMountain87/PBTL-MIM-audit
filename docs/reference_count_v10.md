# Reference count — author decision (T34)

`scripts/check_refs.py` passes: **57 references**, first-citation order strictly 1…57, every
entry cited at least once, every entry carrying a title and a DOI or arXiv id that resolves
(all 57 checked live against Crossref/arXiv on 2026-09-06).

**The plan's target was 30–40.** The count exceeds it because T32 and T33 added the four
literatures the reviewer lenses asked for (surrogate management / multi-fidelity, offline MBO
and Goodhart, nanophotonic reviews and photonic transfer learning, UQ) plus the tool citations.
Note that the plan's own literature-scout lens recommended "roughly 45–50 references", so the
30–40 figure in T34 is inconsistent with the rest of the plan. MLST sets no reference limit
(`docs/mlst_requirements_v10.md`).

## If a cut is wanted, this is the order I would cut in

| # | reference | why it is the most cuttable |
|---|---|---|
| 27 | Kim et al., *Offline model-based optimization: comprehensive review* (2025 preprint) | un-refereed; the field is already covered by refs 8–11 |
| 31 | Surana et al., *Overconfident oracles* (2025 preprint) | un-refereed; supports one illustrative sentence |
| — | one of the five nanophotonic reviews (So 2020 / Khaireh-Walieh 2023) | three reviews already establish the point |
| — | one of Forrester 2009 / Jin 2011 | the two surveys support the same clause |
| — | one of Kennedy 2000 / Peherstorfer 2018 | both cited for "multi-fidelity formalises the same structure" |
| — | Guo 2017 (calibration) | least load-bearing of the five UQ citations |
| — | one of Brookes 2019 / Tripp 2020 | both are items in the same remedy list |

Cutting the first two brings the list to 55; cutting all seven brings it to 50. Reaching 40
would require removing citations that carry their own sentence, which I have not done.

**Author decision:** keep 57, cut to ~50, or name a different target.
