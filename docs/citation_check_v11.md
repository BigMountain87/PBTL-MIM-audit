# Citation attribution check (T71) — 2026-09-13

Method: (1) every DOI resolved through Crossref and every arXiv id through the arXiv API / Semantic Scholar / arxiv.org (Claude, online); (2) codex (gpt-6-astra, offline, no prior reviews) judged each attribution from its own knowledge — report `review_full_scope_codex_v11/7_citations.md`; (3) disagreements settled online.

## Bibliographic records

| ref | identifier | resolves to | verdict |
|---|---|---|---|
| [13] | arXiv 13 | (arxiv.org page checked by WebFetch) —  () | OK |
| [14] | arXiv 14 | (arxiv.org page checked by WebFetch) —  () | OK |
| [15] | arXiv 15 | (arxiv.org page checked by WebFetch) —  () | OK |
| [16] | arXiv 16 | (arxiv.org page checked by WebFetch) —  () | OK |
| [17] | doi | A trust-region framework for managing the use of approximation models  — Structural Optimization 15, 16-23 (1998) | OK |
| [18] | doi | A rigorous framework for optimization of expensive functions by surrog — Structural Optimization 17, 1-13 (1999) | OK |
| [19] | doi | Recent advances in surrogate-based optimization — Progress in Aerospace Sciences 45, 50-79 (2009) | OK |
| [20] | doi | Surrogate-assisted evolutionary computation: Recent advances and futur — Swarm and Evolutionary Computation 1, 61-70 (2011) | OK |
| [21] | doi | Predicting the output from a complex computer code when fast approxima — Biometrika 87, 1-13 (2000) | OK |
| [22] | doi | Survey of Multifidelity Methods in Uncertainty Propagation, Inference, — SIAM Review 60, 550-591 (2018) | OK |
| [23] | doi | A composite neural network that learns from multi-fidelity data: Appli — Journal of Computational Physics 401, 109020 (2020) | OK |
| [24] | arXiv 24 | (arxiv.org page checked by WebFetch) —  () | OK |
| [25] | arXiv 25 | (arxiv.org page checked by WebFetch) —  () | OK |
| [26] | arXiv 26 | (arxiv.org page checked by WebFetch) —  () | OK |
| [27] | arXiv 27 | (arxiv.org page checked by WebFetch) —  () | OK |
| [28] | arXiv 28 | (arxiv.org page checked by WebFetch) —  () | OK |
| [29] | arXiv 29 | (arxiv.org page checked by WebFetch) —  () | OK |
| [30] | arXiv 30 | (arxiv.org page checked by WebFetch) —  () | OK |
| [31] | arXiv 31 | (arxiv.org page checked by WebFetch) —  () | OK |
| [32] | doi | Active learning of deep surrogates for PDEs: application to metasurfac — npj Computational Materials 6, None (2020) | OK |
| [33] | doi | Global Optimization of Dielectric Metasurfaces Using a Physics-Driven  — Nano Letters 19, 5366-5372 (2019) | OK |
| [34] | doi | The preregistration revolution — Proceedings of the National Academy of Sciences 115, 2600-2606 (2018) | OK |
| [35] | arXiv 35 | (arxiv.org page checked by WebFetch) —  () | OK |
| [36] | arXiv 36 | (arxiv.org page checked by WebFetch) —  () | OK |
| [37] | doi | Leakage and the reproducibility crisis in machine-learning-based scien — Patterns 4, 100804 (2023) | OK |
| [38] | doi | Weak baselines and reporting biases lead to overoptimism in machine le — Nature Machine Intelligence 6, 1256-1269 (2024) | OK |
| [47] | arXiv 47 | (arxiv.org page checked by WebFetch) —  () | OK |
| [48] | arXiv 48 | (arxiv.org page checked by WebFetch) —  () | OK |
| [49] | arXiv 49 | (arxiv.org page checked by WebFetch) —  () | OK |
| [50] | arXiv 50 | (arxiv.org page checked by WebFetch) —  () | OK |
| [51] | doi | A review of uncertainty quantification in deep learning: Techniques, a — Information Fusion 76, 243-297 (2021) | OK |
| [56] | arXiv 56 | (arxiv.org page checked by WebFetch) —  () | OK |

All 32 identifiers resolve to the intended works. Corrections applied: [36] full title and pages 1–20; [27] venue TMLR 2025; [31] venue ICML 2024 AI4Science workshop, fifth author added; Table 12 label Kumar & Levine 2020. Codex's claim that [15] Design-Bench is ICLR 2022 was checked online and is wrong (ICML 2022, PMLR 162, pp. 21658–21676) — the entry stands.

## Attribution changes (manuscript §1.3, §4.4, §4.6)

- [17, 18]: no longer described as establishing 'periodic re-certification' or the protocol as trust-region management 'in minimal form'; the protocol keeps one expensive-model consultation and gives up the iterative guarantees.
- [21–23]: the multi-fidelity reading of TMM→RCWA transfer is stated as ours.
- [13, 24, 25, 16, 26]: remedies described individually (proposal/generative adaptation, direct inversion, oracle re-weighting) instead of a blanket 'constrain or repair'.
- [28–30]: best-of-n claim cited to Gao et al. [30] directly, with 'can raise the proxy score while the gold score falls'; Goodhart specified as regressional/extremal.
- [31]: characterised by its abstract (oracles disagree by architecture and seed; poor OOD generalisation).
- [17] removed from the 'certification-only' claim; that is this paper's protocol. 'Better long-run investment' marked as our judgement.
- [37, 38]: [38]'s scope restricted to fluid-related PDEs.
- [47–51]: 'designed for exactly this situation' and 'all of them degrade' replaced by what each work shows; 'by construction' OOD claim replaced by the measured infeasible fractions; conformal coverage caveat added.
- [56]: T2 (within 0.05 of a face) is not the boundary penalty (leaving the box); '9–18 would have been pushed back' is now an upper bound; top-k re-simulation vs Δ audit distinguished by role.

All 21 `[CHECK]` tags removed; the §4.6 mapping paragraph carries AUTHOR-SIGNOFF V11-4.6-MAPPINGS.

## Addendum 2026-09-16 — reference [58] added

- [58] Setinek, Galletti, Gross, Schnürer, Brandstetter, Zellinger, "SIMSHIFT: A benchmark for adapting neural surrogates to distribution shifts," arXiv:2506.12007. Verified via the arXiv API (title, six authors, published 2025-06-13). Cited in Table 12 and one §4.6 paragraph as the closest recent forward-prediction treatment of surrogate degradation under distribution shift; the paragraph states the two differences (experimenter-specified vs optimizer-induced shift; field error vs design decision) and calls the works complementary. No result of [58] is restated numerically.

## Addendum 2026-09-19 — references [59] and [60] added (codex preprint review, should-fix)

- [59] Fannjiang, Bates, Angelopoulos, Listgarten, Jordan, "Conformal prediction under feedback covariate shift for biomolecular design," PNAS 119(43) e2204569119 (2022), doi:10.1073/pnas.2204569119. Verified via the arXiv API (2202.03613v5: title, five authors, journal_ref and DOI as above). Cited in §4.4 beside the conformal-wrapper remark: the paper corrects conformal intervals for the covariate shift that a design procedure itself induces, the shift this audit measures downstream.
- [60] Kandasamy, Dasarathy, Schneider, Póczos, "Multi-fidelity Bayesian optimisation with continuous approximations," ICML 2017, PMLR 70. Verified via the arXiv API (1703.06240: title, four authors, 2017-03-18); venue from the PMLR volume for ICML 2017. Cited in §1.3 to separate sequential fidelity selection from static multi-fidelity prediction.
