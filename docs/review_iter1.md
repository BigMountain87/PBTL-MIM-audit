# Codex review iteration 1 — manuscript_v0 → v1

**Date**: 2026-05-11
**Reviewer**: codex-cli 0.130.0 (`codex exec` non-interactive)
**Prompt template**: [docs/review_prompt.md](review_prompt.md)
**Raw log**: [docs/review_iter1_raw.log](review_iter1_raw.log)

## Verdict
Major revision.

## Decision matrix — which review items we apply (and why)

### Must-fix — all 5 ACCEPT

| # | Issue (codex) | Decision | Implementation in v1 |
|---|---|---|---|
| M1 | Empirical base too small for headline claims (N=6, single Structure A, single seed) | **APPLY** — fundamentally correct. Cannot run full v1 multi-Structure / multi-seed in this iteration; reframe v0 as **pilot / case study**, remove rate-like quantitative claims from abstract / contributions / conclusion. v1 plan (Structure B/C porting, multi-seed, N=50 bootstrap) remains future work. | Abstract & conclusion rewrite to "pilot observations on Structure A." Contributions §1.3 marks 4 items as "framework + pilot evidence." |
| M2 | Fake-optimum taxonomy not correctly operationalized — T2 z-score is computed against the 6 recovered designs themselves, not the design-space distribution; T3 uses raw Euclidean distance | **APPLY** — real bug. Fix script to use **box bounds** from Paper 1 §2.1.1 for T2 normalization, normalize geometry to [0,1] before T3 pairwise distance. Recompute. | Script update: load box bounds explicitly; T2 = recovered geometry near box boundary (norm < 0.05 or > 0.95); T3 = nearest-neighbour normalized distance < 0.5 × median pair. |
| M3 | "Fake optimum" terminology conflicts with Tier 1A — Sample 6 surrogate MAE 8.68% is **above** τ=5%, so it is not a "surrogate pretender" | **APPLY** — semantic conflict real. Two-step fix: (a) **redefine fake-optimum criterion** to be Δ-only (gap-based) regardless of surrogate threshold; (b) rewrite §3.2 to describe Sample 6 honestly — surrogate already self-reported failing (8.68%) but RCWA shows the failure is **4× worse**, which is the inverse-design risk we want to flag. | Rename to "Δ-flagged failure" where appropriate; keep "fake optimum" but with corrected definition. §3.2 narrative rewrite. |
| M4 | Tandem conclusion over-interpreted — "Δ predicts" is wrong (Δ requires post-RCWA, not a pre-decision predictor); also script note typo says 3/6 worse but real count is 4/6 | **APPLY** — both fixed. Replace "predicts" → "diagnoses post hoc". Fix script note. | Script note typo fix. Conclusion §5 rewording. |
| M5 | Not standalone / reproducible — private paths in script, "inherits references from Paper 1" | **APPLY (partial)** — private path replaced with env-var override `INVERSETL_UPSTREAM`. Full reference list embedded inline. Note that data is from Paper 1 (under review) is honest disclosure. | Script reads `os.environ.get("INVERSETL_UPSTREAM", "...")`; references expanded; private path comment removed from manuscript main. |

### Should-fix — all 5 ACCEPT

| # | Issue | Decision | Implementation |
|---|---|---|---|
| S1 | 11 vs 10 dimensionality inconsistency | APPLY | §2.1: "geometry input is 10 design parameters concatenated with one normalized wavelength = 11 input units; inverse optimisation is over the 10 design parameters with the wavelength channel swept." |
| S2 | Justify thresholds with sensitivity (τ, k) | APPLY | New script function `sensitivity_sweep` writes JSON; Methods §2.3 cites the sensitivity panel. |
| S3 | Δ "cheap" qualification | APPLY | "computed at one extra full-wave RCWA call per recovered geometry" (not surrogate-only). |
| S4 | Convergence stability single-seed warning | APPLY | Methods §2.3.C explicitly labels as "single-trajectory descriptor; multi-seed variance is future work." |
| S5 | Prior-work table "None" too dismissive | APPLY | Table column rewording: "Reliability framework treatment" → values describe each prior's specific treatment (e.g., "implicit via tandem" for Liu 2018). |

### Minor / nitpicks — all 6 ACCEPT

1. Remove draft-status notes from manuscript body — APPLY
2. "spectacular" / "canonical" → neutral language — APPLY ("large", "representative")
3. Spelling consistency — APPLY (US English: "optimization", "modeling")
4. Avoid "reviewer-friendly" — APPLY
5. Tandem 3/6 → 4/6 typo — APPLY (script + manuscript)
6. Full reference list inline — APPLY

### Things to defend (v1 must preserve)

- Slogan "Forward MAE ≠ downstream usability"
- RCWA revalidation as the empirical anchor
- Per-sample table transparency
- §4.3 limitations candor (extend, not reduce)
- Tandem comparison framing (with narrowed claims)

### Knockout question (to address in v1 / v2)

> "Can the authors demonstrate, on a pre-declared target set across Structures A/B/C with multiple optimization seeds and confidence intervals, that the proposed Δ-based reliability framework detects inverse-design failures better than ordinary surrogate MAE or RCWA MAE alone?"

**Plan**: v1 manuscript will state this **explicitly** as the central open question and acknowledge that v0 provides a pilot anchor only. The proper answer is the v1 batch of experiments (Structure B/C + multi-seed + N=50). Rather than hide the gap, we promote the question to the title of §4.3 and frame manuscript v1 as case-study + roadmap.

---

## Summary

We accept 16/16 review items. v1 changes are concentrated in:
1. Script bug fixes (T2/T3 box-bound normalization, tandem typo)
2. Sensitivity sweep over τ and k (new figure)
3. Manuscript-wide reframe as pilot / case study (abstract, contributions, conclusion)
4. Terminology disambiguation (fake optimum vs Δ-flagged failure)
5. Reproducibility (env vars, inline references)

No items rejected, but several softened (e.g., M1 implementation = reframing rather than running new experiments — running new experiments would be deferred to a later iteration in any case).
