## 1. Verdict

**Not ready for submission. The headline count reproduces; the manuscript surrounding it does not consistently describe that experiment.**

I recomputed surrogate and reference MAEs from the saved spectra for all nine runs in both arms. They reproduce **16/179 pub pretenders** and **95/179 legacy pretenders**. The maximum discrepancy against archived MAEs was below \(8\times10^{-15}\) percentage points.

The major problems are:

- Legacy methods, results, and interpretations remain embedded in the pub manuscript.
- **107 of Table 10’s 108 numerical entries disagree with the corresponding pub artifact fields.**
- The supplement supplies legacy results where the manuscript promises pub evidence.
- The controls establish that pretenders persist under several alternatives; they do **not** isolate one causal explanation.
- The checker passes this contradictory manuscript.

**Title and abstract:** “Certifies Nothing” overstates what was established. The data demonstrate failures of the 5% self-check on these committed designs, not a general impossibility theorem. All-pass behavior is an observed result of this experiment, not guaranteed “by construction” by the fixed-iteration optimizer. The abstract’s “three controls leave one explanation standing” is substantially stronger than the evidence.

A defensible title would be:

> **A Reference-Solver Audit of Surrogate-Certified Inverse Metasurface Designs**

I read no earlier review documents. Findings below come from the four assigned files and fresh inspection or computation of their supporting artifacts.

## 2. Must-Fix

**1. The Methods describe the wrong release.**

Sources: `paper/manuscript_v11.md:213`, `:223`, `:248`, `:252`, `:493`, `:503`.

The main tables concern the pub arm, but the Methods describe legacy datasets, normalization, feature counts, checkpoints, solver precision, and forward errors.

| Quantity | Manuscript’s legacy description | Current pub evidence |
|---|---|---|
| Reliable samples, A/B/C | 479/461/400 | **499/498/487** |
| C training size | 300 | **350** |
| A wavelength band | 380–780 nm | **400–1800 nm** |
| C physics features | 13 | **18** |
| C width normalization upper bound | 400 nm | **720 nm** |
| Upstream commit | `920b1bd` | **`cb486b5`** |
| Solver precision | complex128 | **complex64** |
| Solver order | Fixed 5 | **Adaptive**, stored `order=-1` |
| Seed-42 forward MAE, B/A/C | 3.19/4.84/3.86% | **1.7221/1.7574/2.1071%** |

Sources are `results_pub/finetune_{A,B,C}_v8.json`, keys `n_good`, `n_train`, `wavelengths_*`, `final_test_mae_pct`, `upstream_commit`; `stats_C_v8.npz:phys_mean,train_hi`; and `rcwa_*_v8.npz:adaptive,sim_dtype,upstream_commit`.

I also hashed the actual pub pretrained checkpoints. Their SHA-256 prefixes are **A `66aa7db1`, B `892aabda`, C `84fe0a59`**, not Table 2’s prefixes.

This prevents readers from reproducing the headline experiment from the Methods. The later limitations correctly describe several pub settings, making the contradiction internal as well as artifact-based.

**2. The supplement does not contain the pub evidence that the manuscript cites.**

Sources: `paper/supplementary_v10.md:3`, `:18`, `:24`, `:333`, `:518`; `paper/manuscript_v11.md:620`, `:708`, `:720`.

- Supplementary S1 totals **95 pretenders**, not 16.
- S2 contains only legacy commitment-rule results, despite the manuscript saying it provides both arms.
- S6 contains the legacy targets and **95 pretenders**.
- Supplementary S8, cited at manuscript line 598, does not exist.
- S3’s legacy order-7 table is appropriate when explicitly cited as legacy evidence.

I checked all **180 S6 rows’ surrogate MAEs, reference MAEs, and gaps**, and all **60 S3 rows’ solver MAEs** against the legacy artifacts. Those numbers reproduce. Their use as support for pub results is the failure.

**3. The manuscript reverses the current discrimination result.**

Sources: `paper/manuscript_v11.md:637`, `:653`, `:768`, `:799`, `:812`.

The assertion that the continuous self-report carries “no usable rank information” is legacy text.

Fresh pooled calculations from pub arrays give:

| Structure | Pooled Spearman ρ | Ordinary pooled AUROC |
|---|---:|---:|
| B | −0.012503 | 0.381264 |
| A | +0.330158 | 0.280702 |
| C | **+0.760878** | **0.985455** |

These reproduce `stats_supplement_v9.json:pooled_rho`.

The three C correlations are **0.750376, 0.828571, 0.718797**. All three survive the artifact’s nine-test Holm correction: adjusted p-values **0.001107, 0.000058, 0.002491**. The manuscript instead says two runs are nominally significant and neither survives correction.

Table 10 is also numerically wrong throughout. Its corresponding pub artifact reports, for example:

- C ensemble disagreement: **0.945**, not 0.70.
- C held-out ensemble MAE: **0.978**, not 0.72.
- C surrogate MAE: **0.978**, not 0.68.
- C distance to dataset pool: **0.495**, not 0.78.
- A continuous T4 value: **0.919**, not 0.55.

Thus the claimed *distance-based exceptions* are not the current exceptions. The conclusion’s near-perfect C discrimination statement is supported; the Results contradict it.

Additionally, Table 10 labels its tenth diagnostic **“multi-start loss spread”**, while the artifact’s corresponding key is **`endpoint_spread`**. Geometric endpoint spread and loss spread are different measurements.

**4. The controls do not isolate a single cause or establish identical behavior.**

Sources: `paper/manuscript_v11.md:28`, `:179`, `:745`, `:845`, `:854`, `:1156`; `paper/mlst/cover_letter.md:19`.

The numerical paired-control results reproduce. Their causal interpretation does not follow.

- M0’s pooled median Δ shift is **+0.1352 pp**, but Structure A shifts **+0.7481 pp**, p = **0.00169**.
- A’s forward error increases from **1.7574% to 2.5860%**, approximately **47%**. Calling this “the same accuracy” is too strong.
- The feasibility control’s pooled median is **−0.00112 pp**, but its mean shift is **−0.79308 pp**. Structure B’s mean shift is **−1.71113 pp**.
- Feasibility-constrained pretenders fall **6→3**. That is not evidence that infeasibility contributes nothing.
- Random search is another optimizer of the same surrogate. Its failures show gradients are **not necessary** for pretenders; they do not demonstrate that gradients have no contribution.

The correct distinction is: **none of these interventions eliminates pretenders**. “Remove … as explanations,” “behaves identically,” and “leave one explanation standing” exceed that result.

The text appropriately distinguishes the feasibility test’s significant p-value from its tiny median. But it then treats that median as a bound on the entire behavior, overlooking heterogeneous and tail improvements.

**5. The feasibility and recoverability sections use obsolete numbers and conclusions.**

Sources: `paper/manuscript_v11.md:829`, `:833`, `:863`, `:868`.

I recomputed infeasibility directly from committed geometries using the stated generator constraints:

| Structure | Infeasible / valid | Pretenders among infeasible | Pretenders among feasible | Fisher p |
|---|---:|---:|---:|---:|
| B | **33/60** | **9/33** | **0/27** | **0.002926** |
| A | **33/59** | **2/33** | **0/26** | **0.498539** |
| C | **13/60** | **0/13** | **5/47** | **0.575439** |

These reproduce `feasibility_v8.json:pooled`. The manuscript instead gives B/A/C infeasibility counts **36/60, 32/60, 34/59**, and B pretender rates **69% versus 21%**.

The lookup/recoverability corrections are:

| Seed-42 quantity, B/A/C | Manuscript | Recomputed from pub artifact arrays |
|---|---|---|
| Lookup successes | 14/15/10 | **20/19/11** |
| Median lookup MAE | 3.3/3.3/5.0% | **2.526/3.224/4.779%** |
| Median surrogate MAE at truth | 3.1/4.8/3.0% | **1.618/1.689/1.761%** |
| Committed prediction better than prediction at truth | 100/100/90% | **100/100/100%** |
| Median normalized geometry distance | 0.62/1.13/0.49 | **0.645/0.779/0.308** |

Sources: `lookup_null_v8.json:runs.*.nn_mae_pct`; `recoverability_v8.json:runs.*.{surr_mae_at_truth_pct,u_dist_best_true,surr_mae_at_best_pct}`.

On C, the optimizer achieves **18 reference successes versus 11 lookup successes**. A blanket claim that the loop does not clearly beat lookup must be replaced with structure-specific results and an appropriate paired comparison.

Also, scoring a committed geometry better than the generating geometry is not itself a failure: the inverse problem is nonunique. Failure requires reference disagreement.

**6. Discussion §4.2 states the opposite ordering from the current results.**

Sources: `paper/manuscript_v11.md:948`, `:952`.

It says worst-case severity and Δ-flags order with preliminary r. Current seed-42 A/C/B values are:

- Successes: **19/18/17** — monotone.
- Flags: **2/1/7** — not monotone.
- Maximum Δ: **9.9007/4.6607/12.1092** — not monotone.

Furthermore, “the contrast vanishes under a common threshold” is not generally true. `stats_supplement_v9.json:common_threshold_h1b` retains **A/B/C = 2/7/1** at either A’s or B’s common threshold, including B–C Fisher p = **0.043596**.

The heading of §4.2 reflects the current result; its paragraph does not.

**7. A legacy reproducibility test is presented as validation of the pub solver.**

Source: `paper/manuscript_v11.md:479`.

The supporting artifact is `results_v8/repro_check_v8.json`, explicitly identifying:

- `results_dir = results_v8`
- `upstream_commit = 920b1bd`
- `sim_dtype = torch.complex128`
- fixed order 5.

I compared its 12 saved recheck spectra against the legacy spectra: **zero deviation reproduces**. No corresponding pub reproducibility artifact was present in `results_pub`.

That validates those legacy checks, not the main experiment’s adaptive complex64 solver. Label the claim by arm or supply the pub check.

**8. Several captions, references, and mathematical expressions are broken.**

Sources and concrete failures:

- `manuscript_v11.md:233`, `:270`, `:275`: clipping and optimization bounds are printed as **`[2]`** rather than `[0,1]`. The optimization domain is mathematically wrong.
- `:552`: Figure 2 says “roughly half” the points are above 5%; the pub proportion is **16/179 = 8.94%**.
- `:653`: Figure 3 identifies two nominally significant runs; current artifacts have **four**, including all three C runs.
- `:663`: Figure 4 says the claimed denominator falls to **159**; pub minimum in the stated sweep is **171**. **159 belongs to legacy.**
- `:673`: **18,000** wavelength solves describes all nine runs, not the three seed-42 runs in §3.2.
- `:785`: T3 counts are transposed: **B=7, C=5**, not C=7, B=5.
- `:531`: unresolved **“Section 3.x.”**
- `:685`, `:907`, `:1134`: **§3.11 does not exist**.
- `:598`: **Supplementary S8 does not exist**.
- `:785`, `:1005`: feasibility and detector cross-references point to §3.6 instead of §3.8 and §3.7.
- `:255`: citation **[40]** is attached to Fourier order, but reference [40] is the listed mixture-density inverse-design paper.

**9. The pre-specification narrative is internally inconsistent.**

Sources: `paper/manuscript_v11.md:391`, `:398`, `:462`; `paper/supplementary_v10.md:102`; `paper/mlst/cover_letter.md:12`.

The supplement says the multi-seed extension was appended **after seed-42 runs**. The manuscript calls it a **pre-specified extension** and describes the next-day modification as a single restart-seeding amendment.

I recomputed both identifiers from the protocol embedded in S4:

- SHA-256: matches.
- Git blob ID: matches.

Those identifiers establish the embedded text’s identity, not its existence before execution. The manuscript itself says no contemporaneous public deposit or commit exists.

Reconcile the extension’s timing and qualify the abstract/letter’s “before computing anything” language. The September control decisions are explicitly informed by earlier results and should remain distinguished from the original plan.

**10. The checker gives false assurance and has a separate runtime bug.**

Sources: `scripts/check_manuscript_v10.py:44`, `:99`, `:115`, `:159`, `:224`.

The default Python could not import NumPy. I therefore executed the unchanged checker through an in-memory standard-library adapter implementing only its used NumPy operations: NPZ loading, array conversion, and `nanmax`.

Result for the pub manuscript:

> **37/37 numbers matched; one skipped; PASS.**

Concrete failures in coverage:

- It loads `detector_bench_v8`, `feasibility_v8`, `lookup_null_v8`, and `recoverability_v8` but never checks their claims.
- It does not check the control-analysis artifact.
- It matches strings anywhere, without binding them to their statistic, structure, or arm.
- Its T4 maximum check passes because **“3.0 %” occurs elsewhere**, while the actual T4 paragraph says **3.1%**.
- Its legacy-wording scan catches **53.1%**, but misses the stale **53%** at manuscript line 995.
- Missing optional artifacts silently remove checks.

Separately, running the assigned manuscript with `--results results_v8` reaches:

> `UnboundLocalError: s310`

` s310` is initialized only in the pub-specific branch at line 165 but referenced unconditionally at line 224.

**Complete quantitative claim ledger**

Repeated claims are grouped below; locations identify their occurrences. “Matches” distinguishes artifact agreement from independent recomputation. Historical and external claims that cannot be established from the permitted evidence are explicitly marked unverified.

| Manuscript locations / quantitative claim | Required artifact key or evidence | Result |
|---|---|---|
| Abstract; §§1.5, 3.4, 4.1, 5: 179 valid, 179 claimed, 163 confirmed, 16 pretenders, 8.9% | `pooled_v8.totals`; recomputed spectra | **Matches** |
| Same locations: Wilson [5.6,14.0]% | `pooled_v8.totals.pretender_wilson95_pct` | **Recomputed; matches** |
| Abstract; §§1.5, 3.4, 5: τ=2.5%, 53/171, 31% | `pooled_v8.tau_sweep.by_tau["2.5"].pooled` | **Recomputed; matches** |
| Abstract; §3.4: τ=7.5%, 6/179; τ=10%, 3/179 | Corresponding `tau_sweep.by_tau` entries | **Recomputed; matches** |
| Abstract; §§3.6, 4.7: M0 +0.14 pp [−0.07,+0.44] | `control_analysis_v10.conditions.m0.primary` | Median recomputed; interval matches JSON |
| Abstract; §3.8: feasibility −0.00 pp [−0.21,−0.00] | `control_analysis_v10.conditions.feas.primary` | Median recomputed; interval matches JSON; rounding obscures nonzero endpoints |
| Abstract; §§1.5, 3.10, 5: 95/179, 53.1% | Legacy `pooled_v8.totals` | **Recomputed; matches** |
| §1.2: “3% forward MAE” | Illustrative example | Not an empirical result; identify as illustration |
| §1.3: 8 restarts, 6,400 draws | `inverse_*.n_restarts,iters`; `mechanism_v8.per_structure.*.random_search.budget` | Settings match; claimed degradation from stronger selection is unsupported |
| §§1.4, 2.7: freeze date, subsequent dates/timestamps | Embedded S4 record; dated artifact provenance | Dates are asserted; temporal priority **not independently established** |
| §§1.4, 2.1, 3.3, 4.7: preliminary r=.72/.34/−.07; printed r=.96/.83/.65 | `stats_supplement_v9.ca_trend.conventions.*.scores` | Matches analysis inputs; underlying pilot correlations not recomputed |
| §§3.3, 4.7: amplitude scores 7.9/8.9/16.9% | `ca_trend.conventions.tmm_mae.scores` | Matches absolute analysis inputs; upstream forward result unverified |
| Table 1: dimensions B/A/C=8/10/7 | `inverse_*.param_names` | **Matches** |
| Table 1: features 13/17/13 | `stats_*.phys_mean` lengths | **C wrong:18** |
| Table 1: wavelength grids and 100 samples | `inverse_*.wavelengths` | 100 matches; **A band wrong** |
| Table 1: good counts 461/479/400 | `finetune_*.n_good` | **Wrong:498/499/487** |
| Table 1: output-channel and head counts | Training `loss` metadata; prediction channels | Channel metadata agrees; architecture/head implementation not independently inspected |
| §2.1: ResNet-256-4, Sigmoid head | Checkpoint architecture specification | **Not independently verified within assigned code** |
| §2.1: C bounds [50,400] | `stats_C.train_lo,train_hi` | **Wrong for pub:[50,720]** |
| §2.1: lr=3e−4, decay=1e−4, 1,000 epochs, batch512 | `finetune_*.lr,weight_decay,epochs,batch` | **Matches metadata** |
| §2.1: train350/350/300; remainders379/361/300 | `finetune_*.n_train,n_good,n_val,n_test` | **Wrong for pub**; train350 each, remainders399/398/387 |
| §2.1: validation/test50 each; seeds42/123/777 | `finetune_*.n_val,n_test,seed` | **Matches** |
| Table 2: six input hash prefixes | Actual dataset/checkpoint hashes | **Pub checkpoint prefixes wrong**; dataset files unavailable for rehashing |
| §2.1: March2026 checkpoints, commit920b1bd, grid64², complex128 | `rcwa_*.upstream_commit,rcwa_settings,sim_dtype` | Grid matches; **commit and precision are legacy**; historical date unverified |
| §§2.1, 2.2: clip/domain `[2]` | S4 domain; `inverse_*.design_lo,design_hi` | **Corrupted notation** |
| §2.2; Fig1; Box1: 500+300 iterations, .05/.02 learning rates, 8 restarts | `inverse_*.iters,lrs,n_restarts,endpoints_u` | **Matches** |
| §2.2:20 held-out targets, fixed target/restart seed formulas | `inverse_*.orig_indices`; `finetune_*.test_idx`; S4 | Counts/held-out membership supported; restart seed formula lacks an archived seed field |
| §§2.3–2.5; Table3; Fig1; Box1: τ5, k2, T1×3, ε.05, T3×.5, K100, σ1%, T4 threshold5 | S4; reliability thresholds/taxonomy | Matches specification; K/σ execution not independently rerun |
| §2.5:95% intervals; nine-run extension | Reliability CI fields; nine artifacts | Supported; “all counts” intervals not actually displayed throughout main tables |
| §2.6; Table4:eight historical defects, ±22nm, six versus20 targets | Embedded historical descriptions | **Historical claims unverified**; no pilot/review documents used |
| §2.7: SHA-256 and git blob ID | Embedded S4 bytes | **Both recomputed and match** |
| §2.7:6/60 main-arm seed42 pretenders | Seed42 spectra | **Recomputed; matches** |
| §§2.7, 3.6:80% power at control rates23–35% | `control_analysis_v10.conditions.*.power` | Matches JSON; not a hard detection boundary |
| §3 introduction:12 checks, four/structure, zero deviation | Legacy `repro_check_v8`; recheck spectra | **Recomputed for legacy only** |
| §3.1:forward3.19/4.84/3.86; thresholds9.69/7.73/6.37 | Pub `finetune_*.final_test_mae_pct`; `tier1b.flag_threshold_pct` | **Wrong arm** |
| §3.1:pilot5.17% | Pilot evidence | **Unverified** |
| §3.2:torcwa0.1.4.2, grid64², order5 | `rcwa_*.torcwa_version,rcwa_settings,adaptive` | Version/grid match; **fixed order wrong** |
| Table6:n_train350 each; forward1.72/1.76/2.11 | Reliability `meta` | **Matches** |
| Table6:claimed20 each; confirmed17/19/18; pretenders3/1/2 and Wilson CIs | `tier1a`; spectra | **Recomputed; matches** |
| Table6:flags7/2/1; thresholds3.44/3.51/4.21 | `tier1b` | **Matches; flags recomputed** |
| Table6:maxΔ12.11/9.90/4.66; amplification35.45/43.62/3.25 | `tier1b.max_delta_pct,max_amplification` | **Matches** |
| Table6:ρ .12/.34/.75, p .62/.14/<.001 | `discrimination.rcwa_vs_surr_PRIMARY` | ρ recomputed; p-values match JSON |
| Table6:T1–T4 counts; modes16/20/17 | `taxonomy.{t1,t2,t3,t4,mode_count}` | **Matches** |
| §3.2:54 confirmed,6 rejected,85–95% agreement | Seed42 counts | **Recomputed; matches** |
| §3.2:B flags7 versus6; one-third as many pretenders; thresholds3.44 versus6.37 | Pub/legacy B reliability artifacts | **Matches** |
| Fig2:roughly half rejected | Pub pooled spectra | **Wrong:8.94%** |
| Table7:all counts, correlations, maxima, monotonicity, Fisher/Holm values | Reliability; `pooled_v8.multiplicity` | **Matches**, apart from surrounding plural “orderings … hold” implication |
| §3.3:trend .026/.030, .269/.221, .981; seed42 .292 | `stats_supplement_v9.ca_trend` | **Recomputed from saved scores/counts; matches** |
| §3.3:chance ordering probability1/6; three conventions | No empirical artifact |1/6 requires exchangeability/no ties; artifact actually contains four conventions, including `intermediate` |
| Table8:all pooled counts, intervals, flags and per-seed correlations | `pooled_v8`; reliability artifacts | **Matches** |
| §3.4:one failed A_s777 geometry;179/180 | `rcwa_A_s777.failed,mae_rcwa` | **Matches** |
| §3.4:ICC .79/.10/−.02; DEFF2.58/1.21/.96; effectiveN23/50/61 | `stats_supplement_v9.clustering` | **ICC recomputed; stated arithmetic matches** |
| §3.4:pooledDEFF1.33, effectiveN134; CI[5.2,15.0]; bootstrap[4.0,14.5] | `clustering.pooled` | Matches artifact |
| §3.4:per-seed B3/3/3,A1/0/1,C2/1/2 | Per-run spectra | **Recomputed; matches**, but A777 denominator is19 |
| §3.4:mild10,moderate3,severe3; boundaries7.5/10 | `tau_sweep.severity_bins`; spectra | **Recomputed; matches** |
| §3.4:ρ range−.107…+.529; two significant; Holm minimum.15 | `pooled_v8.multiplicity`; per-runρ | **Wrong arm** |
| §3.4:pooledρ .02/.34/.28; p .92/.011/.082; AUROC .52/.66/.70 | `stats_supplement_v9.pooled_rho` | **Wrong arm** |
| Fig4:claimed denominator159 | Pub `tau_sweep` | **Wrong:171** |
| §3.5:orders9/13/17;18,000;59%/7%/35%; A98%,C75%,B83% | `adaptive_orders_v10.{all,A,B,C}` | Counts/rounded shares match; **scope is nine runs** |
| §3.5:A58min,B3min | `rcwa_*.elapsed` | **Recomputed; matches rounded medians** |
| §§3.5, 4.7:order5 error+5 to+15pp | External convergence study | **Unverified within permitted evidence** |
| Fig5:target19, .25%/15.9%, radii253/208nm; truth114/104nm, ratio.34 | `rcwa_B_s123`; `inverse_B_s123` | **Recomputed; matches** |
| Fig5:best target15, .24%/.96% | Same artifacts | **Recomputed; matches**, including best-reference index |
| Table9:all nine rule counts and confirmed counts | `random_baseline_*`, `restart0_*`, base RCWA arrays | **Recomputed; matches** |
| §3.6:Fisher .45/.34/1.00; budget6,400 | `mechanism_v8.per_structure` | **Matches** |
| §3.6:M0 lr1e−3; forward1.89/2.59/1.96 | `finetune_*_m0` | **Matches metadata** |
| §3.6:60 pairs; median,p=.176;6→4;discordance4/2;p=.688; per-structure shifts/p | `control_analysis_v10.conditions.m0`; paired arrays | **Pairing, shifts, Wilcoxon, McNemar recomputed; match** |
| §3.7; §4.3:T4 zero/179; mean1.0–1.4%; max3.1% | `pooled_v8.t4`; per-run T4 arrays | Zero flags matches; **means actually.786–1.124%, max2.9685%** |
| §3.7:endpoint spread1.6–2.0 | `tier1c.endpoint_spread_u_mean` | **Wrong:1.356–1.882** |
| §3.7:T2 9–18/run;OR2.18,p.246;sensitivity.75,specificity.37,PPV.10 | `stats_supplement_v9.t2_pretender` | **Matches** |
| §3.7:T3 C7/B5 | Reliability taxonomy | **Transposed** |
| Table10:36 AUROCs plus72 interval endpoints | `detector_bench_v8.pooled.*` | **107/108 numerical entries mismatch**; full replacement below |
| §3.7; §§5,Compute:103–4595s,median1655s; B158,C1662,A3485 | `rcwa_*.elapsed`; `pooled_v8.timing.order5` | **Recomputed; matches**; JSON’s `order5` label is misleading for pub |
| §3.8:generator .9P/.45P/10nm rules | `feasibility_v8.constraints`; geometry arrays | Used in recomputation; consistent |
| §3.8:infeasibility counts,rates,Fisher/CMH p-values | `feasibility_v8.pooled` | **Wrong arm** |
| §3.8:uniform infeasibility.77/.56/.52 | `feasibility_v8.monte_carlo` | Matches saved Monte Carlo output; not rerun |
| §3.8:60 pairs;−.00,p.0071;6→3;discordance4/1,p.375; per-structure shifts/p | `control_analysis_v10.conditions.feas`; paired arrays | **Recomputed; matches** |
| §3.9:lookup and recoverability quantities | `lookup_null_v8.runs`; `recoverability_v8.runs` | **Wrong; corrected above** |
| §3.10:forward ranges3.19–4.84 versus1.58–2.11 | All18 `finetune` results | **Recomputed extrema; matches** |
| §3.10:84/163 confirmed; per-structure legacy30/27/38 and pub9/2/5 | Both arms’ spectra | **Recomputed; matches** |
| §3.10:44.1pp,Newcombe[35.2,52.1],Fisher2.3e−20; per-structure p-values | `pub_vs_legacy_v10` | **Independently recomputed; matches rounding** |
| §3.10:clustered interval[33.3,54.5] | Should have a stored bootstrap key in `pub_vs_legacy_v10` | **No such key found**; independent bootstrap gave[33.3,54.7], broadly consistent, not exact reproduction |
| §§3.10,5:sixfold difference |95/16 | **5.9375; reasonable rounding** |
| §4.1:Cρ+.75…+.83 across seeds | Per-runρ | **Wrong lower endpoint:.719** |
| §4.4:OR2.8,p.003,specificity.39;53% failure | Pub T2/pooled artifacts | **Wrong arm** |
| Box1:six steps;τ5,k2,R8,N20,seeds42/123/777 | Embedded protocol and artifact settings | Settings match; temporal qualification remains |
| §4.5:legacy A9 pretenders/no flags;pub B7/3,A2/1; thresholds3.4–4.2 | Reliability artifacts | **Matches** |
| Table11:3 structures×3seeds,4 taxonomy types | Run inventory; taxonomy | **Matches** |
| §4.6:11–16 T2 flags/run | `taxonomy.t2` | **Wrong for pub:9–18** |
| §4.7:P/λmin≈1.6–2.0 | `inverse_*.design_hi,wavelengths` | **A=1.5; B=C=2.0** |
| §4.7:surrogate MAE.05–3.9% | All pub spectra | **Wrong: .0599–4.6332%** |
| §4.7:legacy C300,400nm versus~720nm;pub487/350/18features | Arm-specific metadata/stats | Numeric settings supported; “one-third outside” not independently checked |
| §5:“same … targets” across arms | `inverse_*.orig_indices,A_target_*` | **False for all three structures** |
| §5:“hours of optimization” | Optimization timing artifact | **Not substantiated by the inspected evidence** |
| Compute:GPU4070TiSUPER;torcwa version;commitcb486b5 | Reliability `meta.env`; RCWA metadata | Matches recorded environment; hardware not independently verified |
| Compute:legacy16–39s | Legacy timing artifact | Arm-specific timing claim; does not describe pub |
| Acknowledgements:specified AI versions,five review iterations | Provenance record | **Not verified**; earlier reviews deliberately not inspected |

Bibliographic years, volumes, pages, identifiers, and author affiliations are not treated as experimental measurements; their external accuracy cannot be established under the requested source restriction.

**Table 10 replacement values from the current artifact**

Each cell below is `auroc [ci_lo,ci_hi]` from `results_pub/detector_bench_v8.json:pooled.{structure}.{key}`. I independently recomputed all **36 ordinary pooled AUROCs** from `runs.*._values` and `_labels`; all match the artifact’s `auroc_naive` fields. The displayed `auroc` fields differ from those ordinary estimates, so the manuscript must also explain that estimator.

| Artifact key | B | A | C |
|---|---|---|---|
| `ens_std` | .804 [.636,.943] | .730 [.421,1.000] | .945 [.846,1.000] |
| `heldout_mae` | .745 [.555,.915] | .514 [.211,.813] | .978 [.896,1.000] |
| `ensmean_mae` | .725 [.524,.897] | .378 [.167,.611] | .978 [.913,1.000] |
| `knn_train` | .797 [.651,.922] | .730 [.526,.914] | .418 [.139,.736] |
| `knn_pool` | .791 [.637,.925] | .730 [.474,.944] | .495 [.211,.776] |
| `tmm_mae_vs_surr` | .425 [.233,.632] | .622 [.353,.875] | .231 [.087,.400] |
| `tmm_mae_vs_target` | .431 [.229,.637] | .622 [.368,.871] | .220 [.057,.405] |
| `t4_mean_pert` | .477 [.216,.745] | .919 [.778,1.000] | .659 [.486,.817] |
| `t2_flag` | .621 [.486,.730] | .662 [.559,.765] | .462 [.214,.724] |
| `endpoint_spread` | .497 [.299,.698] | .270 [.000,.667] | .736 [.579,.875] |
| `mae_surr` | .399 [.182,.634] | .270 [.105,.471] | .978 [.895,1.000] |
| `infeasible_flag` | .765 [.684,.845] | .743 [.639,.861] | .374 [.288,.445] |

## 3. Should-Fix

- **Scope the title’s “nothing.”** `manuscript_v11.md:25`, `:528`, `:930`: an all-pass indicator cannot discriminate *within this selected sample*. That does not prove every thresholded surrogate check is universally uninformative.

- **Do not make tolerance and surrogate quality alternatives.** `:26`, `:633`: the rate depends on both tolerance and the pipeline. The two-arm comparison at fixed τ directly demonstrates the latter.

- **Separate significance from flatness.** `:580`, `:596`: p=.981 is evidence of little detected linear trend under those scores, not an estimated zero effect or proof of no relationship.

- **Avoid treating non-significance as equivalence.** `:724`, `:958`: the commitment rules cannot be ranked reliably here. That does not establish equal performance.

- **Do not infer a regularizer’s ineffectiveness from one diagnostic.** `:968`: zero T4 threshold flags under one perturbation scale does not show that training with the cited smoothness regularizer would leave the same optima unchanged.

- **Document the cluster-adjusted trend calculation.** `stats_supplement_v9.ca_trend.*.deff_corrected` replaces both counts and denominators with fractional effective counts. Explain this approximation; do not imply it is a direct target-cluster permutation test.

- **Clarify bootstrap reproducibility.** Independent standard-library bootstraps supported the qualitative control conclusions but did not reproduce the archived endpoints exactly: M0 approximately **[−.070,.400]**, feasibility **[−.229,−.000056]**. Report the exact resampling method, seed, repetitions, stratification, and percentile convention.

- **Remove submission placeholders.** DOI, correspondence details, funding, authorship, ORCIDs, `[CHECK]`, and author-decision markers remain. The cover letter’s “no results, figures or text overlap” at line33 is also unverified without comparing the companion paper.

## 4. Checked and OK

The following substantive checks succeeded:

- **Both headline rates**, recomputed from spectra rather than aggregate counts.
- **All pub per-run pretender counts**, Δ-flags, and Spearman correlations.
- **The full τ sweep** and severity-bin counts.
- **Wilson intervals** for the headline rates.
- **All nine commitment-rule cells**, recomputed from NPZ arrays.
- **Control pairing, median shifts, Wilcoxon tests, discordant counts, and exact McNemar tests.**
- **The preliminary/published/amplitude trend calculations**, using the artifact’s specified counts and scores.
- **Per-structure ICCs**, independently reconstructed.
- **All 36 ordinary pooled detector AUROCs**, reconstructed from saved detector values and labels.
- **The Figure 5 example**, including the reversed ring radii and best-reference design index.
- **Cross-arm Fisher tests and Newcombe interval**, independently recomputed.
- **Legacy order-7 flips:** A1, B6, C3.
- **Legacy reproducibility spectra:** exact agreement on the 12 recorded checks.
- **Embedded protocol hashes:** exact agreement.
- **Supplement S3 and S6 numerical fidelity to the legacy artifacts.**

I did **not** rerun GPU training or RCWA simulations. “Recomputed” here means recalculation from archived spectra, geometries, per-design arrays, or explicitly identified statistical inputs—not a fresh physical solve.

## 5. Files read

| Assigned file | Lines | Complete read |
|---|---:|---|
| [manuscript_v11.md](paper/manuscript_v11.md) | 1,351 | Yes |
| [supplementary_v10.md](paper/supplementary_v10.md) | 538 | Yes |
| [cover_letter.md](paper/mlst/cover_letter.md) | 45 | Yes |
| [check_manuscript_v10.py](scripts/check_manuscript_v10.py) | 260 | Yes |

Supporting artifacts inspected were the corresponding pub/legacy inverse, RCWA, reliability, fine-tune, statistics, commitment-rule, detector, feasibility, recoverability, lookup, control-analysis, adaptive-order, cross-arm, and legacy revalidation artifacts identified above. Earlier review documents were neither opened nor used.
