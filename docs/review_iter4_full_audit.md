# Review iteration 4 — Full audit (code + numbers + statistics), 2026-06-10

Three parallel review tracks (Claude Fable 5): (1) v7 pipeline code review,
(2) manuscript-vs-results fact check, (3) referee-style statistical review.
CRITICAL-1 was independently re-verified from `inverse_design_{B,C}_v7.npz`.

**Verdict: manuscript_v4's headline results are invalid. Both result
generations (v4a and v7) require a v8 re-run before any v5 manuscript.**

---

## A. Why the v4a headline collapsed (already fixed in v7)

v4a ran Structure B (and C) on a 380–780 nm wavelength axis while the
datasets (`struct_B_500.npz`, `struct_C_500.npz`) are sampled at
400–1800 nm — the same trap as Paper 1's wavelength discrepancy. The v7
re-run (2026-05-11) pinned the dataset grid into `phys_stats_*_v7.npz`
and swapped B to the pretraining 13-feature set. Under the corrected
grid, every headline number changes:

| Metric (Structure B) | v4a (= manuscript v4) | v7 |
|---|---|---|
| Forward test MAE | 5.17 % | 4.73 % |
| Δ-flag threshold (k=2) | 10.34 % | 9.47 % |
| Δ-flagged | **8/20 (40 %)** | **0/20** |
| Max Δ | 26.23 % | 8.79 % |
| Max amplification | 7.76× | 2.21× |
| Pretenders | 2/20 (B-11, B-16) | 1/20 (B-16, marginal 4.90→5.79) |
| ρ(Δ, Surr MAE) | **−0.36 "confidence inversion"** | **+0.21 (gone)** |
| ρ(RCWA, Surr MAE) | +0.14 | +0.89 (= Structure A) |
| RCWA success (τ=5 %) | 0/20 | 0/20 (survives) |
| T1/T2/T3/T4 | 4/1/18/0 | 0/4/19/0 |

Structure C v7 (not yet in the manuscript): test MAE 4.46 %, 0/20 flags,
max Δ 3.28 %, 0 pretenders, ρ(RCWA,Surr)=+0.89, T3=17/20. The monotonic
r→reliability story breaks: max Δ is A (28.9 %) > B (8.8 %) > C (3.3 %),
i.e. the high-r structure looks worst. Pooled ROC
(`roc_discrimination_v7.json`): plain surrogate MAE ranks failures as
well as or better than Δ (AUC 0.901 vs 0.868 at the 10 % label).

## B. CRITICAL bugs that also invalidate the v7 numbers

**C1 — Frozen optimizer (re-verified directly).**
`src/inverse_design_B.py:114-118` / `_C.py:111-116` run Adam on raw
physical parameters (P ≈ 300–800 nm) with lr 3e-2/1e-2, so total
displacement ≈ 18–22 nm. Measured from `inverse_design_*_v7.npz`: all 20
recovered geometries lie within ±21 nm (B) / ±23 nm (C) of the mid-box
init across every parameter — the P box is 500 nm wide. All 20 "inverse
designs" are one point near the initialization. Consequences: T3≈19/20
is the bug's signature, not mode collapse; 0/20 success, Δ, Spearman, ROC
are all artifacts. Fix: optimize normalized [0,1] variables mapped to
physical units, multi-start (current init is deterministic;
`torch.manual_seed(idx)` is a no-op), per-target seeds.

**C2 — Unfiltered divergent RCWA samples.**
`struct_B_500.npz`: 39/500 spectra with A < −0.01 (min −2.27);
`struct_C_500.npz`: 100/500 (min −9.4). Paper 1 filters
(`good = all(A ≥ −0.01)`) and clips to [0,1]; `finetune_B.py:86` /
`finetune_C.py:84-92` do neither. Corrupted rows sit in the training set
(31/350 B, 71/350 C), the test split that defines the Δ-flag threshold,
and the inverse targets themselves (B idx 183 & 86, C idx 62).

## C. MAJOR

**M3 — Pretrained-checkpoint mismatch.** (a) Pretrain geo input is
`[wl, params]`; v7 feeds `[params, wl]` — same dim, `load_state_dict`
succeeds with silently permuted columns. (b) Pretrain reused TMM-set
phys-stats; v7 recomputes on RCWA data → pre-fine-tune val MAE 31.0 %
vs 20.5 % faithful (reproduced to the digit). (c) C only: pretrain
Wx/Wy bounds [50,400] vs v7 [50,720]; ~34 % of C data outside the
pretraining range. Fine-tuning recovers, but "Paper 1 Stage 2 recipe"
is not what was run.

**M4 — T4 never measured.** `reliability_analysis_B.py:115` /
`_C.py:111`: `t4_flag = np.zeros(n)` placeholder, yet JSON reports
`t4_count_all: 0` as if tested.

**M5 — Statistical coupling artifact.** Δ = RCWA − Surr contains −Surr,
so ρ(Δ, Surr) is negatively biased by construction. With Table 3.2's
empirical SDs and Pearson ρ(R,S)=0.255, the coupling formula predicts
corr(Δ,S) = −0.353; observed = −0.353. The v4a "confidence inversion"
carried zero signal beyond the subtraction. The defensible statistic is
ρ(RCWA, Surr): +0.89 (A) vs +0.14 (B, v4a) — "confidence becomes
uninformative", not "anti-informative". (Moot for v7 numbers, but the
lesson must shape v5's stats.)

**M6 — Inference gaps.** ρ(Δ,Surr)=−0.36 at N=20: p≈0.12, CI
[−0.70,+0.11]; A-side +0.71 at N=6: p≈0.11. Fisher exact 1/6 vs 8/20:
p=0.38; 0/6 vs 2/20: p=1.0. None of the headlined contrasts is
significant. T3 clustering additionally breaks the independence
assumption behind Wilson CIs / Spearman (effective N « 20).

## D. Manuscript defects (manuscript_v4.md, fixable by editing)

- §2.2 "100 channels (380–780 nm)" — false for B/C (400–1800 nm).
- §2.3-A cites a τ-sensitivity "Section 3.5" that no longer exists.
- §3.4 "T3 = 1 for both" — false for B-11 (`per_sample_t3[10]=false`).
- §3.2 line below the table swaps 20.65/30.16 between samples 11 and 16.
- Abstract/Conclusion "disagrees by 30.16 %/20.65 %" conflates RCWA MAE
  with Δ (26.23/17.99).
- Table 3.3 labels A's 4.77 % forward MAE "this work" (it is Paper 1's).
- Tandem baseline (4/6 worse, retained through iter2) silently dropped;
  revision_history.md has no v4 row.
- "Pre-specified" holds only for B; A is protocol-development data —
  state the v3 freeze (date + pinning hashes) explicitly.
- §4.3 "RCWA MAE could be performed at any random sample" is a false
  premise — here it *is* evaluated at the optimizer-chosen x*. Reframe Δ
  as a surrogate-overconfidence diagnostic, not a better ranker.
- §4.5: A's 2.2 pp convergence floor cannot be generalized to B's Fano
  resonances; run an N=7 spot check on ~5 B geometries (~minutes).

## E. What survives the audit

- Train/val/test split integrity (350/50/100, seed 0; targets ⊂ held-out
  test, seed 42; no leakage — programmatically verified).
- Physics-feature consistency across finetune/inverse/validate (numpy vs
  torch formula-identical; stats pinned, not recomputed at inverse time).
- The v7 grid fix itself, and the protocol design (hierarchy + taxonomy
  + nulls + Wilson CIs) as a specification.
- Structure A's pilot (native 380–780 nm, unaffected by the grid bug) —
  though N=6 and curated targets remain pilot-grade.

## F. v8 checklist (ordered)

1. Rewrite inverse design: normalized variables + multi-start + real
   seeds (C1).
2. Filter/clip datasets per Paper 1 before fine-tune/targets (C2);
   also mask the 12/500 bad samples in `gen_A_500_broadband` output
   (A down to −33.5, saved unfiltered; its energy check is circular).
3. Restore faithful pretrain config: column order, TMM phys-stats,
   C bounds (M3).
4. Implement T4 (M4).
5. RCWA N=7 convergence spot check on B.
6. Re-run finetune → inverse → RCWA → analysis for B and C; A optional
   re-run under the unified broadband grid.
7. Manuscript v5 from v8 numbers only; recenter stats on ρ(RCWA, Surr)
   with p/CI everywhere; fix every item in section D.
8. Minor: `inverse_design_C.py:162` missing else-branch (NameError path);
   phys-stats computed on all 500 incl. test; C taxonomy T1 uses
   TE/TM-averaged spectra while Tier1A/B average MAEs; Spearman lacks
   tie correction; ROC τ=5 % cell degenerate (45/46 positives).
