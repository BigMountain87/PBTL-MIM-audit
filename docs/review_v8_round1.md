### 1. Verdict

**Major revision.** The core seed-42 and pooled pretender arithmetic mostly matches the JSON, but the manuscript contains one non-negotiable interpretive/numerical error about corrected-*r* monotonicity, and several quantitative claims in §3.4–§3.5 are not verifiable from the declared JSON ground truth.

### 2. Numerical-fidelity audit

| Manuscript claim | JSON / evidence value | Status |
|---|---:|---|
| Abstract: 179 valid designs, one degenerate C RCWA failure | C s777 `meta.n_valid = 19`; all others 20 → 179 total | MATCH |
| Abstract/§3.4: surrogate-claimed success 179/179 | Sum `tier1a.surrogate_pass`: A 60, C 59, B 60 | MATCH |
| Abstract/§3.4: oracle confirms 84/179 | A 33, C 21, B 30 | MATCH |
| Abstract/§3.4: pretenders 95/179 = 53.1%, Wilson [45.8, 60.2]% | A 27/60, C 38/59, B 30/60; Wilson = 45.77–60.24% | MATCH |
| §3.2 seed-42 table: A/C/B n_train 350/300/350 | `synthesis_v8.json` A/C/B `n_train` | MATCH |
| §3.2 forward MAE 4.84/3.86/3.19% | `forward_test_mae` 4.8431/3.8641/3.1856 | MATCH |
| §3.2 surrogate pass 20/20 each | `surrogate_pass` 20/20 each | MATCH |
| §3.2 oracle success 11/7/11 | `oracle_success` A/C/B = 11/7/11 | MATCH |
| §3.2 pretenders 9/13/9 | `pretender` A/C/B = 9/13/9 | MATCH |
| §3.2 Δ-flagged 0/3/6 | `flagged` A/C/B = 0/3/6 | MATCH |
| §3.2 thresholds 9.69/7.73/6.37% | `flag_thr` 9.686/7.728/6.371 | MATCH |
| §3.2 max Δ 7.38/13.67/31.03% | `max_delta` 7.3849/13.6724/31.0328 | MATCH |
| §3.2 max amplification 54.70/27.45/51.73× | `max_amp` 54.6966/27.4485/51.7297 | MATCH |
| §3.2 ρ and p: +0.18 p=.44, −0.04 p=.88, −0.03 p=.89 | `rho_rcwa_surr`, `rho_p` A/C/B = .18496/.43499, −.03609/.87993, −.03308/.88988 | MATCH |
| §3.2 T1/T2/T3/T4: A 19/15/0/0, C 11/11/7/0, B 16/14/5/0 | `synthesis_v8.json` fields `t1`–`t4` | MATCH |
| §3.3 H1a values 11/7/11, not monotone | `hypotheses.H1a...monotone=false` | MATCH for submitted-*r* |
| §3.3 H1b values 0/3/6, monotone; A-vs-B p=.020 | `hypotheses.H1b...monotone=true`; Fisher `flagged.A_vs_B=0.020196` | MATCH |
| §3.3 H1c values +.185/−.036/−.033, not monotone | `hypotheses.H1c...monotone=false` | MATCH for submitted-*r* |
| Abstract/§1.4/W1: corrected *r* also does not yield monotone H1a/H1c | Corrected order is A(+.72), B(+.64), C(+.44). Seed-42 H1a values become 11/11/7, monotone non-increasing; H1c becomes .18496/−.03308/−.03609, also monotone non-increasing. Pooled H1a rates also monotone: 33/60, 30/60, 21/59 | **MISMATCH / BLOCKER** |
| §3.4 ρ range −0.107…+.529 | Reliability JSON min/max = B s123 −0.10677, C s123 +0.52932 | MATCH |
| §3.4 nominal ρ significant: C s123 p=.016, A s777 p=.048 | `reliability_C_s123`: ρ=.5293 p=.01639; `reliability_A_s777`: ρ=.4481 p=.04753 | MATCH |
| §3.4 “neither survives 9-test correction” | Bonferroni threshold .05/9=.00556; both p-values exceed | MATCH |
| §3.4 worst-case order-5 true error 33.6% B s777 | `reliability_B_s777.per_sample.mae_rcwa_pct` max = 33.5834% | MATCH |
| §3.4 largest amplification 157.6× A s123 | `reliability_A_s123.tier1b.max_amplification=157.5812` | MATCH |
| §3.4 order-7 B 9→9, C 13→16, oracle pass 7→4 | Not present in `synthesis_v8.json/md` or any listed reliability JSON | **UNVERIFIABLE / BLOCKER** |
| §3.5 mechanism table random/best-8/single-start counts and Fisher p-values | No `random_baseline_*`, `restart0_*`, or mechanism JSON present in declared evidence | **UNVERIFIABLE / BLOCKER** |
| §3.6 T4 0/60 and perturbation MAE 1.6–2.9% | Seed-42 T4 flags 0/60 match; max T4 means A/C/B ≈1.55/1.89/2.87% | MATCH if scoped to seed 42 |
| §3.6 T2 11–16/20 per run | Reliability JSON T2 true counts across nine runs are 11–16 | MATCH |
| §3.6 pooled T2 p=.022, order-5/order-7 timing 18–36s/73–147s | Not in declared JSON evidence | UNVERIFIABLE |

### 3. Must-fix (blockers)

1. **Corrected-*r* monotonicity is reported incorrectly** ([manuscript_v8.md:47](paper/manuscript_v8.md:47), [139](paper/manuscript_v8.md:139), [530](paper/manuscript_v8.md:530)).  
   Under corrected *r*, the order is A, B, C. Seed-42 H1a becomes 11/11/7 and H1c becomes +0.185/−0.033/−0.036, both descriptively monotone. Pooled oracle success is also monotone: 33/60, 30/60, 21/59. This does not prove the diagnostic “certifies” usability, but it invalidates the manuscript’s claim that H1a/H1c are not supported under corrected *r*.  
   **Fix:** Separate submitted-*r* confirmatory H1 from corrected-*r* post-hoc reordering. State that corrected-*r* restores descriptive monotonicity for H1a, and for seed-42 H1c, while H1b/severity break and count contrasts remain statistically weak.

2. **Order-7 full revalidation counts are unverifiable from the declared ground truth** ([manuscript_v8.md:399](paper/manuscript_v8.md:399)).  
   The manuscript claims B 9→9 and C 13→16, but the listed JSON evidence contains no order-7 fields.  
   **Fix:** Add machine-readable order-7 reliability summaries to `results_v8/`, include denominators and per-sample order-5/order-7 MAEs, and label full B/C order-7 revalidation as post-protocol expansion beyond the pre-registered “3 highest-Δ” spot check.

3. **The mechanism table is not backed by the declared evidence package** ([manuscript_v8.md:411](paper/manuscript_v8.md:411)).  
   Random-search and single-start counts, mean oracle MAE, and Fisher p-values are absent from the specified JSON files.  
   **Fix:** Provide `mechanism_v8.json` or equivalent with the three commitment rules, counts, denominators, Fisher tests, and mean oracle MAEs.

### 4. Should-fix (strong, non-blocking)

1. **“Refutation” language is too strong given corrected-*r* behavior and N** ([manuscript_v8.md:139](paper/manuscript_v8.md:139)).  
   The submitted-*r* H1 is not supported; corrected-*r* partially restores descriptive ordering.  
   **Fix:** Use “does not establish certification” rather than blanket “refutation.”

2. **§3.6 mixes seed-42 and nine-run statements** ([manuscript_v8.md:431](paper/manuscript_v8.md:431)).  
   “T3 appears on C and B but not A” is seed-42 only; A has T3 flags in extra seeds.  
   **Fix:** Explicitly scope that sentence to seed 42.

3. **“First” novelty claim needs narrowing** ([manuscript_v8.md:524](paper/manuscript_v8.md:524)).  
   The offline MBO literature already covers surrogate gaming.  
   **Fix:** Claim novelty in pre-registered photonics cross-fidelity audit with full-wave revalidation, not in the argmin-of-imperfect-surrogate mechanism.

### 5. Minor / nitpicks

- The abstract says “highest-Δ geometries are re-validated at order 7” while §3.4 says “full revalidation” for B/C. Harmonize.
- Add the pooled Wilson CI to a JSON file, not only prose.
- “Every committed geometry” should be “every valid committed geometry” wherever C s777 is included.

### 6. Faithfulness to the pre-registration

Mostly faithful on τ=5%, Δ threshold = 2× forward MAE, Fisher tests, Spearman primary statistic, T4 measurement, and retirement of confidence inversion / 7.76× / B-11/B-16. However, full order-7 B/C revalidation is beyond the frozen protocol’s pre-specified “3 highest-Δ geometries” spot check and must be labeled as an exploratory robustness expansion. The corrected-*r* analysis is also post-hoc and currently misreported.

### 7. Statistical-rigor check

The manuscript is appropriately cautious about N=20/structure/seed, asymptotic Spearman p-values, nine-test multiplicity, wide Wilson intervals, and the Δ-vs-Surr coupling caveat. The main statistical overreach is the corrected-*r* statement: “not supported under corrected *r*” is not defensible for H1a and seed-42 H1c. “Not supported” is generally better than “refuted” here, especially because corrected-*r* restores some descriptive monotonicity without resolving uncertainty.

### 8. Comparison-to-prior-work check

The prior-work positioning is mostly fair, especially the acknowledgment that the mechanism belongs beside offline MBO / surrogate-gaming work. The manuscript should not imply originality for “selecting the argmin of an imperfect surrogate causes over-optimistic designs”; that is close to Kumar & Levine, Trabucco, and Fannjiang & Listgarten. The defensible novelty is the pre-registered, multi-structure computational-photonics reliability audit with RCWA oracle revalidation.

### 9. Reviewer's knockout question

After reordering by Paper 1’s corrected *r*, why should readers accept the headline “*r* does not order inverse reliability” when H1a becomes descriptively monotone at seed 42 and pooled over seeds, and H1c is also monotone at seed 42?