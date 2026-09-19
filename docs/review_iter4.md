# Codex review iteration 4 — manuscript_v5 → v6

**Date**: 2026-05-11
**Raw log**: [docs/review_iter4_raw.log](review_iter4_raw.log)
**Verdict**: Major revision.

## Decision matrix

### Must-fix

| # | Issue | Decision | v6 implementation |
|---|---|---|---|
| M1 | Evidence file mismatch — codex was shown only `reliability_metrics.json` (Structure A) | **APPLY** — update `docs/review_prompt.md` to enumerate all evidence files (B, C, ROC) so next codex sees the full numerical basis. Not a manuscript bug per se, but a review-setup bug. | `docs/review_prompt.md` lists all evidence files explicitly. |
| M2 | "Monotone progression" overclaim with N = 3 anchors | **APPLY** — replace "monotone progression" throughout with "observed three-point trend" / "three-anchor ordering" / similar. Remove "robust signature", "operational meaning". | Abstract, §1.3, §3.2, §3.5, §5. |
| M3 | **ROC label leakage** — Δ = RCWA MAE − Surr MAE, failure label = RCWA MAE > τ, so Δ partially contains the label. Δ is not a pre-deployment predictor (requires RCWA). | **APPLY (critical methodological fix)** — (a) add an explicit Section 3.6.1 "On Δ vs RCWA MAE circularity" disclaiming the partial overlap and arguing the *relative* comparison Δ vs surrogate MAE remains valid because both arms see the same failure labels. (b) Remove "pre-deployment confidence check" phrasing in §4.2. | New §3.6.1 + §4.2 rewrite. |
| M4 | RCWA convergence not verified on B/C top-Δ cases | **APPLY (partial)** — explicit note in §4.5 that per-sample N = 7 convergence on B-11, B-16, A-6 is part of the follow-up batch; cite Paper 1's structure-A test as the only currently available basis. Don't claim convergence on B/C without that data. | §4.5 rewrite. |
| M5 | "Journal-tier discrimination value" overclaim | **APPLY** — replace with "preliminary three-anchor evidence". Move benchmark-level claims to future work. | Abstract, §3.6, §5. |

### Should-fix

| # | Issue | Decision | v6 implementation |
|---|---|---|---|
| S1 | T3 lacks uniform-random baseline (only T2 does) | **APPLY** — add T3 uniform-random shuffle baseline to script + report in §3.4/§3.5. | Update `reliability_analysis_C.py` and `reliability_analysis_B.py`; recompute. |
| S2 | T2 wording inconsistency on Structure C | **APPLY** — state "C: 25 % vs baseline 52 % — below the null point estimate, but uncertainty wide at N = 20". | §3.5. |
| S3 | "Memorised RCWA samples" overclaim | **APPLY** — replace with "fits held-out RCWA moderately (test MAE 5.17 %)". | §4.1. |

### Minor

1. Title — remove "ROC Discrimination of the Δ Diagnostic" or qualify. APPLY (qualify).
2. "Pretender" — define once, use sparingly. APPLY (define in §3.3).
3. Remove "codex iteration" from main body. APPLY (move to docs/revision_history.md).
4. Use "to our knowledge, in cross-fidelity transferred-surrogate inverse design" instead of bare "first". APPLY.
5. AI declaration length — keep but note it's appropriate for transparency. KEEP as is.

### Things to defend

- Slogan, taxonomy, T2 baselines, follow-up section, three-anchor data: all defensible if we soften interpretive language.

## Knockout question (still open after v6)

> "Can the authors demonstrate, with fully supplied reproducible B/C/ROC
> evidence and uncertainty estimates, that Δ adds information beyond
> surrogate-only predictors without using the same RCWA MAE both to
> construct the score and define the failure label?"

This requires either (a) a different score that doesn't include RCWA (e.g., surrogate-only ensemble disagreement, surrogate gradient norm), or (b) bootstrap CIs on AUC differences with multi-seed runs. Both are explicit follow-up items in §4.4.

## Closing note

Four rounds (v0 → v1 → v2 → v3 → v4 → v5 → v6) of independent codex review applied. v6 is the final manuscript this session produces. After v6, the remaining gaps (label-leakage-free predictor, multi-seed bootstrap CIs, N = 50, T4 sweep) are all listed in §4.4 and are bounded by ~1 week of additional GPU work plus ~1–2 weeks of writing — the path to v7 as a fully-validated journal submission.
