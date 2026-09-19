# v8 Protocol — Reliability of Physics-Transferred Surrogates in Inverse Metasurface Design

**Status.** Pre-specified protocol, frozen 2026-06-10, before any v8 computation.
Supersedes all v4a/v7 numerical results (invalidated by the wavelength-grid bug,
the frozen-optimizer bug, and the Δ-coupling statistical artifact; see
`docs/review_iter4_full_audit.md`). The protocol *specification* (metric
hierarchy, taxonomy, thresholds) carries over from manuscript v4 §2.3–2.5 with
the corrections listed in §6 below.

---

## 1. Research question

Paper 1 (Choi et al. 2026, under review) introduced a pilot-set-computable
transferability diagnostic — the TMM–RCWA Pearson correlation r — and showed
it predicts *forward* prediction benefit of physics-based transfer learning
(PBTL). The present study asks:

> **Does r predict the trustworthiness of a physics-transferred surrogate when
> it is used as the forward block of a gradient-based inverse-design loop?**

Hypothesis H1 (from Paper 1 §4.2 #4, §4.6 #3): downstream reliability degrades
as r falls. Pre-specified orderings to test, across the three anchors
A (median r = +0.72), C (+0.34), B (−0.07):

- H1a: RCWA-oracle success rate is non-increasing as r falls (A ≥ C ≥ B).
- H1b: Δ-flag rate is non-decreasing as r falls (A ≤ C ≤ B).
- H1c: ρ(RCWA MAE, Surr MAE) — "is the surrogate's self-report informative
  about true quality?" — is non-increasing as r falls.

Any outcome is reportable: confirmation, partial confirmation, or refutation
(the last would itself be a finding: forward-transferability diagnostics do
not transfer to the inverse setting).

## 2. Design

Three structures from the Paper 1 release, each run through the **identical**
pipeline (single parameterized codebase, `src_v8/`):

| | Structure A | Structure B | Structure C |
|---|---|---|---|
| Physics | asym. dual-dielectric dual-cavity | ring-disk Fano | anisotropic rect (TE/TM) |
| Median pilot r (Paper 1) | +0.72 | −0.07 | +0.34 |
| Design params d | 10 | 8 | 7 |
| Physics features | 17 | 13 | 13 |
| Wavelength grid (native) | 380–780 nm × 100 | 400–1800 nm × 100 | 400–1800 nm × 100 |
| Dataset | struct_A_vis_500 | struct_B_500 | struct_C_500 |
| Good samples (filter §3.1) | 479/500 | 461/500 | 400/500 |
| Output channels | A,R | A,R | A_TE,R_TE,A_TM,R_TM |
| Surrogate | MPhys (1 head) | MPhys (1 head) | MPhys_dual (2 heads) |

Wavelength grids are **per-structure native grids** (the grids the datasets
and the TMM-pretrained checkpoints were built on). Cross-structure comparisons
are of dimensionless quantities (rates, correlations, ratios), not raw spectra.
This is disclosed; the 380–780 nm claim of manuscript v4 §2.2 was wrong for
B/C and is retired.

## 3. Pipeline (per structure)

### 3.1 Data hygiene (mirrors Paper 1 exactly)

`good = all-wavelength A ≥ −0.01` (C: A_TE **and** A_TM), then clip A,R to
[0,1]. Divergent torcwa rows (39 B / 100 C / 21 A) never enter training,
validation, testing, or the target pool.

### 3.2 Split (mirrors Paper 1 stage 2 exactly)

`rng(42).permutation(N_good)` → test = last 50, val = previous 50,
remaining = rest. Fine-tune train set = first 350 of `rng(42)`-permuted
remaining (A: 350/379, B: 350/361, **C: full 300/300** — the upstream code
trains on the full pool when it is smaller than 350 and we disclose
n_train(C) = 300).

### 3.3 Surrogate fine-tune (faithful to the pretrained checkpoints)

Start from Paper 1's TMM-pretrained checkpoints
(`pretrained_mphys_tmm{,_B,_C}.pt`), loaded **strict**. Faithfulness
requirements (violations of these are what invalidated v7's "pretrained"
claim):

1. Geometry input is `[wl_norm, params_norm]` (wavelength **first**).
2. `params_norm` uses the **training bounds** of the checkpoint:
   A = get_bounds("A"), B = BOUNDS_B, **C = BOUNDS_C with Wx,Wy ∈ [50,400]**
   (the C dataset is sampled wider, up to ~700 nm; the upstream fine-tune fed
   normalized values > 1 for those samples and we mirror that, disclosed).
3. Physics features are normalized with the **TMM-set statistics**, recomputed
   deterministically (features depend only on `rng(99)` TMM params and the
   wavelength grid — no simulation needed).
4. Loss = MSE(A) + MSE(R) (C: all four channels), AdamW lr 3e-4, weight decay
   1e-4, cosine annealing, 1000 epochs, batch 512, best-val checkpoint
   (val L1 every 100 epochs), seed 42 — the upstream `train_model` recipe.

Forward test MAE (A channel; C: mean of TE/TM) on the 50-sample test split
defines the Δ-flag threshold (§4, k = 2).

### 3.4 Inverse design (the v8 core fix)

For each target spectrum A*(λ) (C: TE and TM jointly):

- **Variable**: u ∈ [0,1]^d (normalized), mapped to physical units on the
  **design box** = dataset sampling box (get_bounds). Clamped to [0,1] after
  every step. This replaces v4a/v7's Adam-on-raw-nanometers, which could move
  at most ~22 nm in a 500 nm box (the frozen-optimizer bug).
- **Optimizer**: Adam on u, lr 0.05 × 500 iters, then 0.02 × 300 iters.
- **Multi-start**: R = 8 restarts per target — 1 mid-box + 7 uniform draws
  from `rng(1000 + target_idx)` — run as one batched tensor; the committed
  geometry is the restart with the lowest final surrogate loss. All restart
  endpoints are archived (they feed the Tier-1C dispersion diagnostic).
  *(Amendment, 2026-06-11, flagged by independent codex review: the as-run
  implementation seeds restarts with `rng(1000 + original-dataset index of
  the target)` — a fixed, pre-data-determined quantity, but not literally
  the wording above. All v8 results use the as-run definition; recorded
  here for exact reproducibility.)*
- **Objective**: mean squared error over the 100 wavelength channels between
  surrogate prediction and target (C: mean of TE and TM MSE).
- Physics features are recomputed differentiably from the current geometry at
  every iteration; feature normalization uses the §3.3 TMM stats.

**Targets**: N = 20 per structure, drawn `rng(42).choice(test_split, 20,
replace=False)` — random, from held-out data, identical mechanism for all
three structures (this removes the curated-6-target confound of Structure A
in manuscript v4).

### 3.5 Oracle revalidation

Every committed geometry is simulated with full-wave RCWA (torcwa) at the
dataset-generation settings (64 × 64 grid, Fourier order [5,5]).
**Convergence spot check** (pre-specified): the 3 highest-Δ geometries per
structure are re-simulated at order [7,7]; the per-structure convergence floor
is reported next to the Δ values it must not be confused with.

## 4. Metrics (pre-specified, unchanged thresholds)

- **Tier 1A — success rate**, τ = 5 %: surrogate-claimed (Surr MAE ≤ τ),
  oracle (RCWA MAE ≤ τ). **Pretender** := Surr MAE ≤ τ AND RCWA MAE > τ
  (formal definition, fixed before any v8 run). Wilson 95 % CIs on all counts
  including zeros.
- **Tier 1B — oracle gap**: Δ = RCWA MAE − Surr MAE at the committed geometry;
  flag when Δ > 2 × forward-test-MAE. Report max Δ; amplification
  (RCWA/Surr) only with the small-denominator caveat.
- **Tier 1C — convergence**: multi-start dispersion (std of the 8 final losses;
  max pairwise distance among the 8 endpoints in u-space) + loss-tail RSD of
  the committed run.
- **Taxonomy** (per committed geometry): T1 oracle divergence
  (RCWA MAE > 3 × Surr MAE); T2 box-edge (any u within 0.05 of 0/1), against
  the d-dimensional uniform null; T3 mode collapse (NN distance < 0.5 × median
  pair distance), against the N = 20 uniform null (mean + 95 % band);
  **T4 surrogate-local smoothness — actually measured in v8** (K = 100
  Gaussian perturbations, σ = 1 % of box width, flag if mean perturbation
  MAE > 5 %).

## 5. Statistical analysis plan (corrects the v4 errors)

1. **Primary discrimination statistic: ρ(RCWA MAE, Surr MAE)** (Spearman,
   asymptotic p from scipy, approximate Bonett–Wright Fisher-z 95 % CI —
   no exact/permutation inference is claimed). This is the clean "does
   surrogate confidence track oracle truth" quantity.
2. ρ(Δ, Surr MAE) is reported **only as secondary with the coupling caveat**:
   Δ contains −Surr by construction, so its negative bias is partly
   tautological (in v4a data the coupling formula reproduced the observed
   −0.353 exactly — zero residual signal). No "confidence inversion" language
   unless the effect survives in ρ(RCWA, Surr) itself.
3. Cross-structure count comparisons (success, flags, pretenders): Fisher
   exact tests; we state outright when differences at N = 20 are not
   significant.
4. N = 20 with T3 clustering ⇒ effective sample size may be smaller; the
   cluster structure (number of distinct modes, within-cluster spread) is
   reported alongside every count-based CI.
5. All seeds fixed and logged: split 42, train-subset 42, targets 42,
   restarts rng(1000 + original-dataset index of the target), training 42
   (multi-seed extension: train-subset/training seeds 123 and 777).

## 6. Corrections relative to manuscript v4 (why v8 numbers will differ)

| # | v4/v7 defect | v8 fix |
|---|---|---|
| 1 | B/C run on 380–780 nm vs 400–1800 nm data (v4a) | native grids from dataset, pinned in artifacts |
| 2 | Adam on raw nm — optimizer frozen within ±22 nm | normalized variables + multi-start (§3.4) |
| 3 | divergent RCWA rows in train/test/targets | Paper 1 filter (§3.1) |
| 4 | pretrain input column order permuted; phys stats recomputed on RCWA; C bounds mismatch | faithful loading (§3.3) |
| 5 | T4 reported but never computed | implemented (§4) |
| 6 | ρ(Δ, Surr) headline (coupling artifact) | primary stat = ρ(RCWA, Surr) (§5) |
| 7 | A: 6 curated targets vs B: 20 random | identical N = 20 random targets for all structures |
| 8 | single deterministic init; `manual_seed` no-op | R = 8 multi-start, archived endpoints |

## 7. Success criteria for "this is a paper"

The paper stands on whichever of these the v8 data delivers:

- **H1 confirmed (monotone degradation)**: the r diagnostic extends to
  inverse design → direct sequel to Paper 1.
- **H1 refuted (no relation / non-monotone)**: forward transferability does
  not certify inverse usability *and* the popular r-style diagnostic is shown
  insufficient — the protocol itself becomes the contribution, with the
  refutation as the empirical core.
- **Mixed**: the taxonomy localizes *which* failure mode tracks r and which
  does not — a structure-resolved reliability map.

What the paper can no longer claim regardless of outcome: the v4a confidence
inversion, the 7.76× amplification, pretenders B-11/B-16, and any number
produced by the frozen optimizer.
