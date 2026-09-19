# Revision history — `paper/manuscript_v*.md`

| Version | Date | Trigger | Headline change |
|---|---|---|---|
| v0 | 2026-05-10 | initial draft | full reliability framework draft on Structure A pilot |
| v1 | 2026-05-11 | codex review iter 1 | reframe as pilot; T2/T3 use box bounds; sensitivity sweep added |
| v2 | 2026-05-11 | codex review iter 2 | T4 actually measured (surrogate-side); T2 uniform-random baseline; discrimination ablation; reproducibility hardening |
| v3 | 2026-05-11 | codex review iter 3 | reposition as protocol + pilot; pre-specified threshold rationales (§2.5); Wilson 95 % CIs on counts; archived inputs with SHA-256 |

Per-iteration decision matrices: `docs/review_iter{1,2,3}.md`.
Raw codex logs:                `docs/review_iter{1,2,3}_raw.log`.
Hashed input bundle:            `archived_inputs/`, hashes in `docs/pinning.txt`.

After three iterations the codex critique converged on a structural
rather than presentational issue: a six-sample, single-structure,
single-seed post-hoc accounting cannot, by editing alone, be promoted
to a journal article. v3 therefore reframes the manuscript as a
**pre-specified protocol + pilot illustration**, which makes the
pilot's scope match what the data actually supports. The next
substantive revision (v4) requires the cross-structure follow-up batch
described in `manuscript_v3.md` §4.3, which is the missing core
evidence for the knockout question.

| v4 | 2026-05-11 | cross-structure pilot (A+B) | two-anchor synthesis; pretenders B-11/B-16; confidence inversion (**later retracted**) |
| v5–v7 | 2026-05-11 | codex iter 3–10 | three-anchor r-progression, ROC; iter 10 = ACCEPT on the v7 evidence |
| — | 2026-06-10 | `review_iter4_full_audit.md` | v4a/v7 evidence invalidated (wavelength-grid mismatch, frozen optimizer, Δ-coupling); protocol `v8_protocol.md` frozen |
| v8 | 2026-06-10 → 06-12 | v8 protocol run (3 structures × 3 seeds, order-7 full revalidation, mechanism controls) + codex round-1 (`review_v8_round1.md`, Major Revision) | 179/179 self-certified, 95 pretenders; corrected-r monotonicity reported honestly; `mechanism_v8.json`, `order7_revalidation_v8.json`, `pooled_v8.json` added; venue → MLST |
| v9 | 2026-09-04 | final consolidation (Paper 1 published with corrected r) | preliminary/published r terminology; pooled-CI non-independence caveat (C1-stat); order-7 per-design instability on B/C reported; Figures 1–5 generated (`figures_v9/`, `scripts/make_figures_v9.py`); Paper 1 citation updated |
