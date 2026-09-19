# MLST / IOP author requirements — fetched 2026-09-06 (T51)

Retrieval date for every row: **2026-09-06**. Quotes are verbatim (≤ 25 words) from the page at the URL.
`UNVERIFIED — author to check` marks pages that could not be fetched (bot protection).

| # | requirement | verbatim quote | URL |
|---|---|---|---|
| 1 | Scope | "bridges the application of machine learning across the sciences with advances in machine learning methods and theory as motivated by physical insights" | https://publishingsupport.iopscience.iop.org/journals/machine-learning-science-and-technology/about-machine-learning-science-technology/ |
| 2 | Article type = Paper, length limit | "Reports of original research work; normally not more than 8500 words." | same as #1 |
| 3 | Letters (for contrast) | "not normally be more than eight journal pages in length (5000 words)" | same as #1 |
| 4 | Abstract word limit | "Should not normally be more than 300 words" | https://publishingsupport.iopscience.iop.org/journals/machine-learning-science-and-technology/ |
| 5 | Format-free first submission | "We only require you to upload a single PDF file (and any relevant supplementary data)"; "You can format your paper in the way that you choose!" | https://publishingsupport.iopscience.iop.org/questions/article-format/ |
| 6 | LaTeX template | "iopjournal class file" … "it is not essential to use this class file or to format your article in the same style. Any common variant of TeX is acceptable." | https://publishingsupport.iopscience.iop.org/questions/latex-template/ |
| 7 | Peer-review model | "Single-anonymous, double-anonymous (author choice)" | same as #1 |
| 8 | Double-anonymous: optional, what to strip | "If you choose to submit single-anonymous then it is at your own discretion." — remove "all author names, affiliations", "direct reference to your previous publications", "institutions or funding information" | https://publishingsupport.iopscience.iop.org/double-anonymous-faqs/ ; checklist: https://publishingsupport.iopscience.iop.org/questions/checklist-for-anonymising-your-manuscript/ |
| 9 | Figure formats / size / fonts | "Vector EPS … or PDF" preferred; "typically 8.5cm for a small/single-column figure and 15cm for a large/double-column figure"; "text sizes of 8 to 12 pt"; fonts "Times, Helvetica, Courier or Symbol"; "colours are distinguishable if the figure is converted to greyscale" (no dpi stated) | https://publishingsupport.iopscience.iop.org/questions/figures-journal-articles/ |
| 10 | Reference style | "You may use either of these two systems for your references" (Vancouver numerical / Harvard); "it is not necessary to format your references in the ways shown in the guidelines" | https://publishingsupport.iopscience.iop.org/questions/references/ |
| 11 | Data availability statement | "The journal requires authors to include a data availability statement in their article."; template "The data that support the findings of this study are openly available at [URL/DOI]"; dataset citations "must include a persistent identifier such as a DOI" | https://publishingsupport.iopscience.iop.org/iop-publishing-data-availability-policy/ ; templates: https://publishingsupport.iopscience.iop.org/iop-publishing-standard-data-policy/ |
| 12 | AI-use disclosure | "they must disclose this usage in the Acknowledgements section of their manuscript. This disclosure should list the model and version of the generative AI tool and how it was used"; "AI tools cannot meet the requirements for authorship" | https://publishingsupport.iopscience.iop.org/questions/generative-ai-tools/ |
| 13 | APC amount | Price list 2026: Machine Learning: Science and Technology (2632-2153) **GBP 2500 / EUR 3000 / USD 3125**, "Fully open"; about page: "£2500 / €3000 / $3125 (excluding VAT). IOP members receive 25% discount." | https://publishingsupport.iopscience.iop.org/wp-content/uploads/2026/01/2026_IOPP_APC-Pricing_External.xlsx ; same as #1 |
| 14 | APC timing / discounts | "Where APCs apply, they are only charged once an article has been accepted for publication."; reviewer reward "10% discount on the cost of publishing" | https://publishingsupport.iopscience.iop.org/questions/article-publication-charge-pricing-and-the-costs-of-open-access-publishing/ ; https://publishingsupport.iopscience.iop.org/reviewer-discounts-on-article-publication-charges/ |
| 15 | Supplementary material limits | "up to a maximum of 50 MB each, as long as the combined file size for all files including the main article is no more than 150 MB"; "Titles must not exceed 30 characters, and descriptions must not exceed 30 words"; "Files will not be included in peer review by default" | https://publishingsupport.iopscience.iop.org/questions/supplementary-material/ |
| 16 | ORCID | Checklist asks "Do you have an ORCID iD?" and "ORCID IDs and affiliations of all co-authors" — recommended; no mandatory statement found | https://publishingsupport.iopscience.iop.org/submission-checklist/ |
| 17 | Author approval | "All named authors have consented to submission to the journal, approved the submitted version of the article, and all further revisions."; CRediT: "Authors may wish to use a taxonomy such as CRediT" | https://publishingsupport.iopscience.iop.org/questions/ethics-of-authorship/ |
| 18 | Line numbers | "There is no need for you to include line numbers in your manuscript as these will automatically be added on submission." | https://publishingsupport.iopscience.iop.org/questions/article-format/ |
| 19 | Submission-options page (APC/OA on iopscience.iop.org) | UNVERIFIED — author to check (bot-protection redirect; values in #13 come from the publishing-support site instead) | https://iopscience.iop.org/journal/2632-2153/page/submission-options |

## Mismatches with plan assumptions (PLAN_v10 / amendment)

| plan assumption | fetched requirement | action |
|---|---|---|
| Abstract ≤ 300 words (T30) | "not normally more than 300 words" | none — matches |
| APC GBP 2500, 25 % member discount (T43) | GBP 2500 / EUR 3000 / USD 3125; 25 % IOP-member discount; 10 % reviewer discount also exists | none — matches; add the reviewer-discount option to the checklist |
| `iopjournal` class required, tectonic build (T42) | format-free first submission: single PDF; template optional | **T42 step is optional** — the `article`-class fallback is acceptable at first submission; keep iopjournal only if it compiles cleanly |
| Double-anonymous chosen via `\documentclass` option (T42/T43) | author's choice; if double-anonymous, strip names/affiliations/self-citations/funding from the PDF | none — author decision stands; note the strip list for the anonymised PDF |
| GenAI disclosure in a separate "Declaration of Generative AI Use" section (v9) / one sentence in Acknowledgements (T23/T40) | must be in **Acknowledgements**, listing **model and version** and how used | **T23's replacement sentence must name model + version** (Claude Opus 5 / Fable 5.1 / Codex version TODO-AUTHOR) and live in Acknowledgements — the separate Declaration section is removed |
| Figures 6.7 in (17.0 cm) double-column, Arial/Helvetica/DejaVu (T17) | double-column "typically 15cm"; fonts "Times, Helvetica, Courier or Symbol" | **T17: set double-column width to 15 cm (5.9 in) and single to 8.5 cm (3.35 in); font-family Helvetica** |
| Paper ≤ 8500 words (editor lens) | "normally not more than 8500 words" | none — v9 body 6384 words |
| Supplementary titles ≤ 30 chars (T43) | ≤ 30 characters, description ≤ 30 words, 50 MB/file, 150 MB total | none — matches; add size limits to the checklist |
| Data availability sentence with DOI (T40) | mandatory statement; template sentence above; dataset DOI required | none — matches |
| ORCID required (T40/T43) | recommended, asked for all co-authors; not stated as mandatory | keep placeholders; not a blocker |
