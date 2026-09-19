# Codex review iteration 3 — manuscript_v2 → v3

**Date**: 2026-05-11
**Raw log**: [docs/review_iter3_raw.log](review_iter3_raw.log)
**Verdict**: Major revision (still — and this is the structural ceiling we hit on six-sample data).

## Decision matrix — review items applied in v3

The codex iter3 critique is now structural rather than fixable by editing.
Items M1, M2, M3, M4, M5, M6 each ultimately require either new
cross-structure experiments or a different scope. Since the user-specified
scope of this session is "iterate the manuscript with codex review" rather
than "run the cross-structure follow-up batch", we accept the reviewer's
implicit alternative: **reframe the manuscript explicitly as a short
technical note / pre-specified protocol + pilot illustration**, rather
than a full research article. With that reframe, the remaining edits
are tractable.

### Must-fix

| # | Issue | Decision | v3 implementation |
|---|---|---|---|
| M1 | Δ not shown to add information | **APPLY (reframe)** — repositioned manuscript as protocol + pilot. Title, abstract, §1.4 all explicitly state this. The discrimination ablation (§3.5) becomes the *raison d'être* of the follow-up batch rather than a deficiency. | Title becomes "A Pre-Specified Protocol …", §1.4 "Status of this manuscript" subsection. |
| M2 | N = 6 too weak for journal | **APPLY (reframe)** — same. The manuscript is now positioned as the kind of artifact one publishes *before* the cross-structure batch (e.g., as a journal "Methods" or "Pre-registration Note"). | Abstract opens "we present a pre-specified protocol …". §5 conclusion explicitly says "the present manuscript is the protocol; the follow-up batch is the test." |
| M3 | Thresholds arbitrary | **APPLY** — add §2.5 "Pre-specified thresholds and rationales" giving application/fabrication grounding for each (τ, k, T1 factor, T2 ε, T3 factor, T4 threshold). | New §2.5. |
| M4 | T4 surrogate-only | **APPLY** — rename T4 to "**surrogate-local smoothness** (proxy for non-robustness)". | §2.4 + §3.4. |
| M5 | RCWA convergence | **APPLY** — cite Paper 1 §2.2.1 convergence test (N = 5 vs N = 7 mean spectral MAE 2.2 %) as sufficient for the same RCWA settings; explicitly note that Sample 6's 37.57 % is not within the convergence error bar. | §2.2 + §4.5. |
| M6 | Reproducibility depends on under-review Paper 1 | **APPLY (partial)** — explicitly archive `mphys_350.pt`, `phys_stats.npz`, `constrained_rcwa_validation.npz`, `tandem_results.npz` in this repo (with hash) and emit a `pinning.md` note. Cannot solve fully until Paper 1 publishes — disclose. | New `docs/pinning.md`; manifest of file hashes. |

### Should-fix

| # | Issue | Decision | v3 implementation |
|---|---|---|---|
| S1 | Tandem comparison underpowered | APPLY — narrow wording further: "this implementation on this pilot". | §3.3. |
| S2 | "pre-registered" too strong | APPLY — replace with "pre-specified". | abstract + §1. |
| S3 | "established" Paper 1 r > 0.3 | APPLY — replace with "reported". | §1.1. |
| S4 | T2 baseline against test geometries | DEFER — Paper 1 test geometries available but the v3 script change adds noise to the v3 narrative; we keep the uniform-random baseline and explicitly state in §3.4 that a training-geometry baseline is part of the follow-up batch. |
| S5 | Binomial CIs for counts | APPLY — Wilson score 95 % intervals computed in script and reported in §3.1. | New helper in script + table notes. |

### Minor / nitpicks

1. Spelling — APPLY (full US English pass).
2. "fake-optimum" → keep as our coined term but parenthetically gloss as "surrogate-induced failure mode" once. PARTIAL APPLY.
3. Slogan repetition — APPLY (keep in abstract and §1.2 only; remove from §3.2 / §5 redundancies).
4. Clarify MAE = absorptance percentage points — APPLY.
5. Remove "codex iteration" from main body, move to revision-history supplement — APPLY.
6. Path consistency — APPLY.

### Things to defend (v3 must keep)

- Honest scope (now strengthened by explicit "protocol + pilot" framing)
- Per-sample tables
- Surrogate vs RCWA distinction
- T2 uniform-random baseline correction
- §3.5 honest admission

### Knockout question

> "Does Δ identify inverse-design failures better than simply running
> RCWA once and ranking by RCWA MAE, on a pre-declared multi-structure,
> multi-seed target set with uncertainty estimates?"

Cannot be answered in v3. The manuscript's revised conclusion makes this
question the *target* of the follow-up batch, not the *deliverable* of the
present manuscript.

## Closing note for this iteration

After three rounds (v0 → v1 → v2 → v3), the codex critique converges on a
structural rather than presentational issue: a six-sample post-hoc
accounting cannot be promoted to a journal article. v3 is therefore the
final version we can produce **without running new experiments**. Any
further codex iteration would re-state the same structural critique. The
appropriate next step is the cross-structure follow-up batch on
`compute-host`, after which a v4 manuscript can address the M1 / M2
critique with data rather than reframing.

We stop the codex iteration loop here for this session.
