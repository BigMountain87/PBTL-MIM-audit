# Codex review iteration 2 — manuscript_v1 → v2

**Date**: 2026-05-11
**Raw log**: [docs/review_iter2_raw.log](review_iter2_raw.log)
**Verdict**: Major revision (still).

## Decision matrix — review items applied in v2

### Must-fix

| # | Issue | Decision | v2 implementation |
|---|---|---|---|
| M1 | Empirical base still too small for "framework" | **APPLY (frame)** — strengthen pilot framing further. Cannot run B/C in this iteration; explicitly title the manuscript a "proposed framework + pilot illustration" rather than "framework + pilot evaluation". | Title clarified; abstract leads with "proposed". Section 1.4 already separates IS/IS-NOT but we sharpen. |
| **M2** | **Δ not shown to add info beyond RCWA MAE** — sharpest finding. On N=6 the rankings are nearly identical. | **APPLY** — Add a dedicated comparison subsection (§3.5 in v2) that reports Spearman rank correlation between Δ, RCWA MAE, surrogate MAE, and T2/T3 flags. **Honestly admit** that on N=6 Δ adds little and motivate the v1 batch (large N) as the test of incremental value. | New §3.5 "Discrimination ablation" with rank correlations + honest admission. |
| M3 | T4 unmeasured | **APPLY (measure)** — Run a surrogate-based 1% Gaussian perturbation sweep on each recovered geometry. Report perturbation-induced spectrum MAE distribution. | New `tier_d_perturbation_sweep` in script; T4 binary flag = mean perturbation MAE > 5%. New §3.4 with results. |
| **M4** | **T2 baseline analysis** — 1 − 0.9¹⁰ ≈ 65% → 5/6 not extraordinary | **APPLY** — Compute baseline T2 rate on 10,000 uniform-random box samples; report enrichment. | New `t2_baseline_enrichment` in script; new line in §3.4 with sober observation. |
| M5 | Reproducibility (matplotlib cache, private paths in JSON) | **APPLY** — Add MPLCONFIGDIR fallback in script, `--no-plot` CLI mode, sanitise JSON paths to upstream-relative, add `requirements.txt`. | All applied in v2 script. |

### Should-fix

| # | Issue | Decision | v2 implementation |
|---|---|---|---|
| S1 | "k=2 most common conservative multiple" — no cite | APPLY — drop the "common in literature" claim; describe as "arbitrary, sensitivity-tested". | Edit Methods §2.3 wording. |
| S2 | "rises sharply" overstates N=6 | APPLY — use counts. | Edit §3.1 wording. |
| S3 | "Routinely walks to edges" too broad | APPLY — restrict to "in this pilot, five of six recovered geometries lie near at least one bound; the 10-dimensional uniform-random baseline rate is ≈ 65%, so this is not yet a surprise". | Edit §3.1 / §4.2. |
| S4 | Tandem methodological detail | APPLY — add line stating same surrogate weights, identical 6 targets, identical 500+300 Adam, identical box clamp; tandem invnet from Paper 1 §S13 release. | Edit §3.3. |
| S5 | Paper 1 dependence | APPLY — add a short standalone-readability note in §4.4 explaining what Paper 1 contributes (the surrogate weights and the 6 targets) and that the framework specification stands alone. | Edit §4.4. |

### Minor / nitpicks

1. Prompt typo `manuscript_vv1.md` → fix in `docs/review_prompt.md` and prompt-template substitution. APPLY.
2. Reference numbering `[1]` then `[3]` — add `[2]` Choi-Paper-1 or renumber. APPLY (renumber inline references).
3. Cheng `J. Singh` → `Prashant Singh` per metadata. APPLY.
4. Spelling normalize/normalise inconsistency (manuscript says "normalised" / `normalized` JSON). APPLY (US English everywhere).
5. "v1 batch" inside manuscript v1 confusing → rename "follow-up batch". APPLY.
6. Remove process notes from manuscript body. APPLY.

### Things to defend (v2 must keep)

- Pilot framing
- Per-sample tables
- Surrogate-claimed vs RCWA-oracle distinction
- Slogan as motivation
- τ / k sensitivity sweeps

### Knockout question (still open)

> "Can the authors demonstrate, on a pre-declared multi-structure, multi-seed target set with confidence intervals, that Δ and the fake-optimum taxonomy detect inverse-design failures better than simply ranking recovered designs by RCWA MAE or surrogate MAE?"

Cannot fully answer in v2 (no new structures); but **partially** addressed by adding §3.5 discrimination ablation that *honestly demonstrates* the open question.

## Summary

We accept 16/16 review items again. v2 changes are concentrated in:
1. New T4 perturbation sweep (no longer a placeholder).
2. T2 baseline enrichment (sober interpretation of 5/6).
3. Δ vs surrogate-MAE / RCWA-MAE discrimination ablation (§3.5).
4. Reproducibility hardening (no-plot mode, MPLCONFIGDIR, requirements.txt, sanitised JSON paths).
5. Citation/typography cleanup, language softening.
