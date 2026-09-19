## 1. Verdict

**Major revision. The headline counts reproduce; several interpretations and secondary quantitative statements do not survive inspection.**

I independently recovered **16/179 pub-arm pretenders, 95/179 legacy-arm pretenders, 53/171 at τ = 2.5%, D’s 4/50, and the cross-solver change from 6/60 to 13/60**. The main evidence is real. The manuscript nevertheless contains a directly reversed conclusion about feasibility, unsupported solver-certification language, misleading control interpretations, and inconsistent detector summaries.

The noun-phrase title is appropriately scoped as an audit. The **abstract and conclusion still overclaim**: these observations support failure of this pipeline’s threshold check, existence of pretenders under the tested controls, and differences between these two releases. They do not establish a universal optimization mechanism, comparative equivalence of commitment rules, or physical certification.

No earlier review document was opened. Findings below come from the assigned text and computations performed during this pass.

Execution qualification: the installed Python environments lacked NumPy. I decoded the numeric NPZ arrays using Python’s standard library and independently implemented the relevant arithmetic, rankings, exact count tests, and signed-rank calculations. I also executed the checker’s unchanged logic with an in-memory adapter for its three NumPy operations. No repository files were changed.

## 2. Must-Fix

1. **The conclusion’s universal feasibility claim is false—and C reverses its direction.**

   [Manuscript:1362](paper/manuscript_v11.md:1362) says the regularity holds on every testable structure: pretenders are committed outside the generator’s feasible region.

   Recomputing constraints from `inverse_*_v8.npz.best_params`, then classifying the spectra, gives:

   | Structure | Pretenders / infeasible designs | Pretenders / feasible designs |
   |---|---:|---:|
   | A | 2/33 | 0/26 |
   | B | 9/33 | 0/27 |
   | C | **0/13** | **5/47** |
   | D | 4/20 | 0/30 |

   These reproduce `feasibility_v8.pooled.*.table` and `structure_d_v10.feasibility_D`. All five C pretenders are feasible. Section 3.8 correctly says this at lines 911–912; the conclusion contradicts it. Replace the universal claim with the A/B/D-specific observation and explicitly retain C as the counterexample.

2. **The cross-solver probe is being promoted into convergence certification and a decomposition of the legacy rate.**

   [Manuscript:1030](paper/manuscript_v11.md:1030) calls the adaptive solver “converged,” calls order-5 rejections manufactured, and says the published pipeline removes that part of the legacy failure rate. [W8:1322](paper/manuscript_v11.md:1322) repeats the attribution.

   The paired arrays reproduce seven adaptive-pass/order-5-fail flips and zero reverse flips. They establish **truncation sensitivity on these 60 pub-arm geometries**. They do not establish which answer is physically correct, nor how many legacy-arm failures would disappear on its different geometries. The manuscript itself acknowledges unconverged adaptive results at lines 743–744 and 1320–1321.

   Also revise “163 that are [good]” at [1115](paper/manuscript_v11.md:1115) and “known to be” at [1368](paper/manuscript_v11.md:1368) to **meeting the specified reference-solver tolerance**.

3. **The constrained-control interpretation mistakes significance for attribution and understates the observed effect on A.**

   [Manuscript:928](paper/manuscript_v11.md:928) says “the whole effect sits on B.” Recomputed mean paired Δ shifts are:

   - B: **−1.7111 pp**
   - A: **−0.7063 pp**
   - C: **+0.0382 pp**

   With equal sample sizes, A contributes approximately **29.7% of the pooled net mean improvement**. A’s nonsignificant signed-rank result does not establish zero contribution.

   [Line 941](paper/manuscript_v11.md:941) additionally says the feasible rerun “halves the B count.” It changes **3 → 1**, a two-thirds reduction. **6 → 3 is the pooled count.** Both are directly recoverable from the paired arrays and `control_analysis_v10.conditions.feas.per_structure`.

   Finally, W7’s claim that transfer’s contribution is bounded to “about a percentage point” is too broad: A’s reported Hodges–Lehmann interval extends to approximately **+1.675 pp**, and a pooled location-shift interval is not a bound on individual or structure-specific effects.

4. **The abstract uses a Δ interval to support a statement about pretender rates.**

   [Abstract:27](paper/manuscript_v11.md:27) describes an “indistinguishable rate,” immediately supported by the median **Δ shift +0.14 pp [−0.07, +0.44]**.

   That interval concerns a continuous paired gap, not the difference in binary pretender rates. The relevant binary evidence is **6 → 4, discordances 4/2, exact McNemar p = 0.6875**. It does not demonstrate equivalence. Section 3.6 largely makes the distinction correctly; the abstract should preserve it.

   Similarly, “random search is no better” in the abstract and [conclusion:1348](paper/manuscript_v11.md:1348) implies a comparison the study says it cannot resolve at lines 790–791. Use “also produces pretenders; the comparison is inconclusive.”

5. **Several detector summaries use the wrong estimand or include a detector whose current result contradicts the summary.**

   - [Section 3.4:698](paper/manuscript_v11.md:698) gives claimed-MAE AUROCs **B 0.38, A 0.28, C 0.99**. These reproduce the *naively pooled* values in `stats_supplement_v9.pooled_rho.*.auroc_pooled`.
   - Table 10 explicitly uses within-seed comparisons, weighted by positive–negative pairs. Recomputing that estimator gives **B 0.398693, A 0.270270, C 0.978022**, hence **0.40, 0.27, 0.98**.

   Both calculations exist, but the prose presents them as the same pooled benchmark. Adopt Table 10’s estimator consistently or label the alternative explicitly.

   Additional mismatches:

   - B infeasibility AUROC is **0.76**, not **0.77**, under Table 10’s estimator: lines 886 and 913.
   - C’s ensemble/self-report range in lines 882–883 is **0.95–0.98**, not 0.94–0.98.
   - [Section 4.4:1188](paper/manuscript_v11.md:1188) includes *k*-NN training distance, then summarizes these detectors as **0.94–0.98 on C**. C’s training-distance AUROC is actually **0.417582**, with CI **[0.157895, 0.611111]**.

   The last is substantive: the text assigns the successful ensemble result to a distance detector that performs below chance in this sample.

6. **A loss-tail statistic is presented as evidence about agreement across restarts.**

   [Manuscript:839](paper/manuscript_v11.md:839) says “final-loss spread is small,” supported by **loss-tail RSD 0.002–0.012**, then infers distinct basins with equally low claimed loss.

   The assigned methods distinguish these quantities:

   - `tier1c.tail_rsd_mean`: temporal behavior of the committed run;
   - `tier1c.restart_loss_std_mean`: dispersion across restart endpoints.

   I recomputed mean restart-loss standard deviations from `inverse_*.final_losses`: approximately **0.000524–0.002430 in loss units**. The quoted RSD is not that measurement. Report the appropriate statistic, its units, and the corresponding endpoint evidence.

7. **The manuscript answers a different question from the one it repeatedly claims to settle.**

   [Introduction:164](paper/manuscript_v11.md:164), [discussion:1255](paper/manuscript_v11.md:1255), and [cover letter:34](paper/mlst/cover_letter.md:34) turn the audit into a negative answer about the forward transferability diagnostic.

   However:

   - The demonstrated constant classifier is **surrogate MAE ≤ 5%**, not a decision rule based on *r*.
   - No inverse-certification cutoff or calibrated probability derived from *r* is evaluated.
   - H1a’s preliminary ordering holds descriptively.
   - Its seed-42 trend is nonsignificant; its pooled exploratory trend is nominally significant.

   Failure of a thresholded self-report is not itself failure of a specified *r*-based certification procedure. Frame the result as **the forward diagnostic does not, by itself, supply a demonstrated inverse guarantee**, rather than a tested certification rule being disproved.

   The same restraint is needed for “optimizing against **any** fixed learned objective … drifts” at [1137](paper/manuscript_v11.md:1137). The controls show pretenders can persist after removing each tested ingredient. They do not isolate the shared argmin step as the sufficient causal explanation.

8. **The checker’s PASS cannot establish manuscript consistency; a current central claim can be corrupted without detection.**

   I changed only the manuscript text **in memory**, replacing every “53 of the 171” with **“52 of the 171”**. The checker still returned:

   > Part A: 84/84 numbers matched; check_manuscript: PASS

   Concrete causes in the assigned script:

   - [Lines 88–92](scripts/check_manuscript_v10.py:88) check τ = 7.5 and 10, but omit the central τ = 2.5 claim.
   - [Lines 45–50](scripts/check_manuscript_v10.py:45) require strings somewhere in the body, without binding them to a particular claim.
   - [Lines 263–265](scripts/check_manuscript_v10.py:263) exempt everything from §3.10 through §3.12 as “two-arm,” including the D section.
   - [Lines 340–350](scripts/check_manuscript_v10.py:340) accept count values from multiple outcomes and scopes and retain that broad section exemption.

   Add claim-specific checks for τ = 2.5, control counts by structure, detector estimator consistency, and D. Narrow the legacy exemption to actual contrasts.

9. **The claimed reproducibility package is incomplete in the inspected snapshot.**

   [Manuscript:1407](paper/manuscript_v11.md:1407) promises the datasets, D inputs, checkpoints, and archived evidence. `results_pub/INPUTS_MANIFEST.txt` names files absent at their declared paths:

   - `upstream_inputs_pub/data/raw/struct_{A,B,C,D}_500_redesign.npz`
   - `archived_inputs/pub/pretrained_mphys_tmm_D_pub.pt`
   - `archived_inputs/pub/pretrained_m0_tmm_D_pub.pt`

   The A/B/C pretrained checkpoints are present and their hashes reproduce. The missing files prevent independent verification here of dataset filtering, nearest-neighbor lookup against training spectra, and D’s checkpoint provenance.

   This establishes a limitation of the supplied snapshot, not that a future archive cannot contain them. Complete the archive and replace the DOI placeholders before asserting public availability. Also fix [line 1390](paper/manuscript_v11.md:1390), which directs readers to “publicly released checkpoints” upstream while lines 225 and 1412 say that release ships none.

## 3. Should-Fix

- **Residual legacy framing in W1.** [Lines 1276–1278](paper/manuscript_v11.md:1276) say the published data-generation pipeline postdates “the release audited here,” although that published pipeline is now the main audited arm. “None is statistically supported” also needs to distinguish confirmatory inference from the nominally significant exploratory preliminary trend. I found no unlabelled legacy **95/179 or 53.1%** headline in the main text; S3 and S8 explicitly identify their arm. S4 should prominently identify its embedded specification as originally written for that arm.

- **Pre-specification summary is internally inconsistent.** Box 1 at [1197](paper/manuscript_v11.md:1197) associates seeds 42/123/777 with being written down “before anything runs,” whereas §2.7 explicitly cannot establish that for the extension. Table 5 promises every §3 analysis but does not separately list the amended from-scratch and constrained-control inferential analyses.

- **The family description miscounts test types.** [Line 390](paper/manuscript_v11.md:390) says eight *per-structure* Wilcoxon tests. There are **six per-structure plus two pooled Wilcoxon tests**, and two McNemar tests.

- **Cover-letter “by construction” is false.** [Cover letter:16](paper/mlst/cover_letter.md:16) contradicts the manuscript’s explicit statement at lines 578–581 that universal surrogate passage is observed, not guaranteed. The random and single-start controls themselves have surrogate failures.

- **Small numerical corrections.** At [manuscript:802](paper/manuscript_v11.md:802), the M0 mean shift is **+0.0645206 pp → +0.06**, not +0.07. At line 805, A’s HL upper limit **1.6746648 → 1.67**, not 1.68. At line 682, A’s seed-777 count is **1/19**, so the shared “of 20” qualifier is inaccurate.

- **Time summaries need labels.** Cross-solver **16–29 versus 156–3526 seconds** are ranges of **structure medians**, not ranges across individual designs. The claimed optimization time of **1.5–2.3 minutes/design** at line 1367 has no corresponding elapsed-time field in the inverse NPZ artifacts inspected.

- **Numbering and cross-references need cleanup.** The two-arm table at [975](paper/manuscript_v11.md:975) has no table number or caption; numbering jumps from Table 10 to “Table 11” for related work. The “first two data rows” at line 574 are training size and forward MAE, not the success rows. Figure references 1–6 resolve textually, but packaged filenames still call the spectrum and mechanism figures `fig6` and `fig7`, and an obsolete order-7 figure remains in `paper/mlst/figures/`.

- **Use nominal intervals honestly in the abstract.** Its [5.6, 14.0]% interval ignores repeated targets. The manuscript provides a cluster interval of [4.0, 14.5]%; use that or label the Wilson interval nominal.

- **Do not imply exact zero p-values.** Supplement S1 prints C’s p-values as `0.000`; use scientific notation or `<0.001`.

## 4. Checked and OK

**What I executed:** spectrum-level MAEs and gaps for all 18 A/B/C runs across both arms, D, and the six control conditions; classification at four tolerances; per-run rank correlations; feasibility cross-tabs; control location estimates and tests; cross-solver shifts and flips; ICCs; trend-test arithmetic; detector aggregation; checkpoint hashes; and the embedded protocol’s hashes.

The maximum discrepancy between recomputed and archived main-arm MAEs was below **10⁻¹⁴ percentage points** across the two arms. Comparing the archived recheck spectra against the original spectra gives exactly zero deviation on all 12 selected geometries per arm. This verifies the stored comparison, **not a fresh RCWA simulation today**.

I also compared all **360 per-design rows** in S6/S8.3 against their artifacts for indices, MAEs, Δ, status, taxonomy flags, truth MAE, normalized geometry distance, and support flags: no mismatches. The legacy order-7 aggregate flips reproduce: **A 1, B 6, C 3**, with confirmed counts **11→10, 11→11, 7→4**.

The following ledger covers the manuscript’s scientific quantitative claims in section order. Repeated claims in captions, discussion, abstract, and conclusion share the indicated source. Bibliographic years/pages and illustrative examples are distinguished from experimental results.

**R** = independently recomputed; **A** = matches stored artifact/metadata but not independently regenerated; **U** = not verifiable from the available assigned evidence; **Mismatch** = correction required.

| Manuscript location and quantitative claim | Artifact key or required source | Result |
|---|---|---|
| Abstract: 3 structures × 3 seeds; 179 valid, all claimed; 16 pretenders, 8.9% | `inverse_*.orig_indices`; `rcwa_*.A_rcwa_*`, `A_target_*`; `pooled_v8.totals` | **R: matches** |
| Abstract: Wilson [5.6,14.0]% | `pooled_v8.totals.pretender_wilson95_pct`; recompute Wilson(16,179) | **R: matches nominal interval** |
| Abstract: 53/171, 31%, at 2.5%; 6/179 at 7.5% | `pooled_v8.tau_sweep.by_tau.*.pooled`; raw spectra | **R: matches** |
| Abstract: M0 +0.14 [−0.07,+0.44]; feasible −0.18 [−0.58,−0.00] pp | `control_analysis_v10.conditions.{m0,feas}.primary` | Point estimates **R**; intervals **A**. Different estimands: median versus HL |
| Abstract: D 4/50; legacy 95/179, 53.1% | `structure_d_v10.structure_D`; legacy spectra and `pooled_v8.totals` | **R: matches** |
| §1.2: 3% forward MAE | Illustrative example, not an experimental estimate | No artifact required; keep explicitly illustrative |
| §1.3: 8 restarts, 6400 random draws | `inverse_*.n_restarts`, `iters`; `mechanism_v8`/baseline metadata | 8 and 8×800 **R**; draw-budget description **A** |
| §1.4: preliminary r A .72/C .34/B −.07; published B .96/A .83/C .65 | Embedded S4; `stats_supplement_v9.ca_trend.conventions.*.scores` | **A**; underlying pilot correlations not recomputed |
| §1.4: freeze date 2026-06-10 | Embedded S4; contemporaneous record would be required | Internal statement matches; precedence **U** |
| §1.5: all contribution counts/rates; detectors near .8 | Sources above; `detector_bench_v8.pooled` | Counts **R**; no contradiction in these numeric summaries |
| Table 1: dimensions B8/A10/C7; features 13/17/18; grid 400–1800 ×100 | `inverse_*.param_names`, `wavelengths`; `stats_*.phys_mean` | **R: matches** |
| Table 1: reliable samples B498/A499/C487 of500 | `finetune_*.n_good`; original datasets required | **A**; missing datasets prevent filtering recomputation |
| §2.1: ResNet-256-4; single/dual heads and output channels | Architecture/checkpoint construction; `finetune_*.loss` records channels | Channels **A**; architecture not independently established from scalar results |
| §2.1: pretrain 5000, rng99, 500 epochs, lr .001, batch2048, seed42 | `archived_inputs/pub/pretrain_*_pub.json.{n_tmm,epochs,lr,batch,seed}`; `phys_stats` | **A: matches recorded recipe** |
| §2.1: C bounds [50,720] versus legacy [50,400] | `stats_C_v8.{design_lo,design_hi,train_lo,train_hi}` in both arms | **R: matches** |
| §2.1: fine-tune lr .0003, decay .0001, 1000 epochs, batch512 | `finetune_*.{lr,weight_decay,epochs,batch}` | **A: matches** |
| §2.1: 350 train; 50 validation/test; pools399/398/387 | `finetune_*.{train_idx,val_idx,test_idx,n_good}` | Lengths, disjointness and arithmetic **R**; exact RNG replay not performed |
| Table 2: three dataset and checkpoint SHA prefixes | `results_pub/INPUTS_MANIFEST.txt`; checkpoint bytes | Checkpoint hashes **R**; dataset hashes **A**, files absent |
| §2.1: commits cb486b5/920b1bd; complex64/128; 64×64; orders9/13/17 versus5 | `rcwa_*.{upstream_commit,sim_dtype,rcwa_settings,adaptive,orders_min,orders_max}` | **A/R: metadata matches** |
| §2.1: material provenance, including Siefke2016 | Solver/material sources would be required | **U** beyond the saved material labels |
| §2.2/Fig.1: 100 channels; 500+300 iterations; lrs .05/.02; R8; N20 | `inverse_*.{wavelengths,iters,lrs,n_restarts,orig_indices}` | **R: matches** |
| §2.2: mid-box plus7 starts; exact restart RNG | Archived initialization or implementation required | **U** from final endpoints alone |
| §§2.3–2.5/Fig.1/Table3: τ5, k2, T1×3, T2 .05, T3 .5, T4 K100/σ1%/threshold5 | Embedded S4; `reliability_*.{tier1a,tier1b,taxonomy}` | Definitions match; T1/T2/counts **R**; K/σ are recorded-method claims |
| §2.5: 9 Spearman, 9 Fisher, 2 pooled primaries, 10 controls, 36 detector cells | `pooled_v8.multiplicity`; `control_analysis_v10`; detector list ×3 | Counts **R**; “8 per-structure” description wrong |
| §2.6/Table4: historical pilot numbers, ±22nm, grids,6 versus20 targets | Historical pilot evidence/reviews | **U**; not adopted as findings in this review |
| §2.7: protocol SHA-256 and git blob ID | Exact embedded S4 protocol bytes | **R: both match exactly** |
| §2.7: UTC timestamps, June/September chronology and dates | Contemporaneous records required; some dates echoed in metadata | **U** as evidence of when decisions were made |
| §2.7: control MDE23–35%, 80% power | `control_analysis_v10.conditions.*.power.mde_binary_at_80pct` | **A**; values match, power calculation not independently reconstructed |
| §3 opening:12 geometries,4/structure, zero spectral and MAE differences,6.2GPU-hours | `repro_check_v8.designs`; `rcwa_*_v8_reprocheck` arrays | **R:** zero differences; pub elapsed **6.24264h** |
| §3.1: B1.72/A1.76/C2.11%; printed1.65±.04/1.79±.04/2.05±.09; \|z\|≤1.7 | `finetune_*.final_test_mae_pct`; `pub_gate_table123.rows` | **A: matches gate**; printed ten-seed statistics not independently recomputed |
| §3.1: pilot5.17%; legacy3.2–4.8% | Pilot source; legacy `finetune_*.final_test_mae_pct` | Pilot **U**; legacy range **A**, explicitly contrasted |
| Table6: all success/pretender/flag counts and Wilson intervals | Seed42 spectra; `reliability_*.tier1a`, `tier1b` | **R: matches every count and interval** |
| Table6: maxΔ B12.11/A9.90/C4.66; amplification35.45/43.62/3.25 | Raw MAEs; `tier1b.{max_delta_pct,max_amplification}` | **R: matches** |
| Table6: ρ .12/.34/.75 and p .62/.14/<.001 | `discrimination.rcwa_vs_surr_PRIMARY` | ρ **R**; p-values **A** |
| Table6: T1/T2/T3/T4 and modes | `taxonomy.{t1,t2,t3,t4,mode_count,per_sample}` | Stored counts match; T1/T2 independently **R** |
| §3.2: 54 confirmed/6 rejected;85–95%; range .06–4.6% | Seed42 classifications; all-run surrogate arrays | Counts **R**; range is **all-run**, not seed42-only |
| §3.2: legacy B6 flags versus pub7; threshold6.37 versus3.44 | Both arms’ B `tier1b` and forward MAE | **R: matches** |
| Table7: H1a19/18/17, pooled57/59,55/60,51/60; H1b2/1/7 | Raw classifications | **R: matches; monotonicity labels correct** |
| Table7: Fisher .605 and .044; Holm1.000/.392 | Count tables; `pooled_v8.multiplicity` | Fisher **R**; Holm values match |
| §3.3: trend .026/.030; printed .269/.221; amplitude .981; seed42 .292 | `stats_supplement_v9.ca_trend.conventions` | **R: all reproduce** |
| §3.3:16 events,3 structures, chance1/6 for one strict ordering | Counts;3! permutations | **R/arithmetic: matches** |
| §3.3: amplitude A7.9/B8.9/C16.9 | `ca_trend.conventions.tmm_mae.scores` | **A: matches** |
| Table8: every pooled count, interval, flag total24, nine correlations | Raw arrays; `pooled_v8`, `reliability_*` | **R: matches** |
| §3.4: one failed A777;179/180; shared20 targets | `rcwa_A_s777.failed`; `inverse_*.orig_indices` | **R: matches** |
| §3.4: ICC C.79/B.10/A−.02; DEFF2.58/1.21/.96; effectiveN23/50/61 | Pretender matrices; `stats_supplement_v9.clustering.per_structure` | **R: matches** |
| §3.4: pooledDEFF1.33/effectiveN134; adjusted CI[5.2,15.0]; cluster[4.0,14.5] | `clustering.pooled` | Stored values match; independent4000-draw bootstrap approximately **[3.93,14.53]%** |
| §3.4: τ10 gives3/179; severity10mild/3moderate/3severe | Raw reference MAEs; `tau_sweep.severity_bins` | **R: matches** |
| §3.4: four nominal Spearman results; three C survive Holm | `multiplicity.spearman_p_nominal`, `spearman_p_holm` | **A: matches** |
| §3.4/Fig.3: pooledρ B−.01/A.33/C.76; block p .94/.080/.0002 | `stats_supplement_v9.pooled_rho` | ρ matches arithmetic; permutation p **A**, not rerun |
| §3.4: claimed-MAE AUROC .38/.28/.99 | `pooled_rho.*.auroc_pooled` | **R**, but conflicts with Table10’s specified estimator |
| §3.5:18000 solves; N17≈59%,N13≈7%,N9≈35%; A98%,C75%,B83% | `adaptive_orders_v10.{all,A,B,C}` | **R from counts:**58.6167/6.6944/34.6889%; per-structure values match |
| §3.5: A≈58min,B≈3min | `rcwa_*.elapsed`; `pooled_v8.timing` | **R: median summaries match** |
| §3.5/W8: “+5 to +15pp” convergence error | Companion convergence evidence | **U** in the permitted evidence; does not certify current adaptive solves |
| Fig.5: B123 target19 .25/15.9%; radii253/208; truth114/104; ratio.34 | `inverse_B_s123.{best_params,true_params}` and spectra | **R: matches** |
| Fig.5: target15 .24/.96% | Same artifacts | **R: matches** |
| Table9/§3.6: all nine pretender/claimed and confirmed counts | `random_baseline_*`, `restart0_*`, main arrays | **R: matches** |
| §3.6: random-vs-best Fisher .45/.34/1.00 | Recomputed2×2 tables | **R: matches** |
| §3.6: M0 forward1.89/2.59/1.96 versus1.72/1.76/2.11; A47% higher | `finetune_*_m0.final_test_mae_pct`, base values | **R/A:47.1493% increase; rounded values match** |
| §3.6:60 pairs, p.176, HL+.22, median+.14,27down/33up,6→4, p.688 | Paired raw spectra; `control_analysis_v10.conditions.m0` | **R: matches**; mean+.07 is wrong |
| §3.6: per-structure medians−.36/+.75/−.03, p.216/.0017/.622; AHL+.92 | Same | **R: matches**; A CI upper rounds1.67 |
| §3.7: T4 0/179, means .79–1.12%, max2.97% | `taxonomy.t4_mean_pert_mae_pct` | **R from per-design values**; perturbation predictions not regenerated |
| §3.7: RSD .002–.012; endpoint1.36–1.88 | `tier1c.tail_rsd_mean`; raw `endpoints_u` | Values match, but RSD is misidentified as restart spread |
| §3.7: T2 9–18; CMHOR2.18/p.246; sensitivity.75/specificity.37/PPV.10/base.089 | `stats_supplement_v9.t2_pretender`; classification flags | Values match; counts/performance arithmetic **R**, stored CMH inference **A** |
| §3.7: B/C T3 7/5 at seed42; A0 then2 | `reliability_*.taxonomy.t3` | **A: matches** |
| Table10: all36 AUROCs and all36 intervals;2000-draw cluster method | `detector_bench_v8.pooled.*.*` | Every cell matches; all36 weighted point estimates **R** from per-run AUROCs; CIs **A** |
| §3.7: all C pretenders in worst-ranked four | Raw C surrogate-MAE ranks | **R:** ranks1/4,1,1/2 across seeds |
| §3.7/§4.4: detector ranges and B infeasibility.77 | `detector_bench_v8.pooled` | **Mismatch**, detailed above |
| §3.7/compute:103–4595s, median1655; B158/C1662/A3485 | Raw `rcwa_*.elapsed` | **R:**102.798–4594.761; median1654.537 |
| §3.8: constraints .9P/.45P/10nm; infeasible33/60,33/59,13/60 | Geometry arrays; `feasibility_v8.constraints`, `pooled` | **R: matches** |
| §3.8: uniform infeasibility .77/.56/.52 | `feasibility_v8.monte_carlo.*.p_infeasible` | **A: matches**, Monte Carlo not rerun |
| §3.8: feasibility Fisher B.0029/A.50/C.58; B CMH.0036 | Geometry/status tables; `feasibility_v8.pooled` | Fisher **R**; CMH **A** |
| §3.8:60pairs,40improve/20worsen, Wilcoxon.0071/sign.013; HL−.18, median−.00, mean−.79 | Paired spectra; `conditions.feas.primary` | **R: matches**; CI endpoints **A** |
| §3.8: range−10.7 to+4.6;6→3; discordant4/1;p.375 | Same | **R: matches** |
| §3.8: Bmedian−.55/HL−1.39/mean−1.71; A−.15; C−.00; per-structurep | `conditions.feas.per_structure`; raw pairs | **R: matches** |
| §3.8: Holm.063 over10, .014 over2 | `conditions.feas.primary` | **R arithmetic: matches** |
| §3.8: “halves B” | B raw status; `per_structure[B].pretender_*` | **Mismatch:3→1** |
| §3.9: lookup20/20,19/20,11/20; medians2.5/3.2/4.8 | `lookup_null_v8.runs.*.nn_mae_pct` | **R from saved errors**; nearest-neighbor search itself **U** without datasets |
| §3.9: claimed medians .4/.4/1.1; truth1.6/1.7/1.8; truth passes20/20/18;100% preference | Raw claimed MAEs; `recoverability_v8.runs.*.surr_mae_at_truth_pct` | **R from arrays: matches** |
| §3.9: geometry distance .64/.78/.31 | `inverse_*.{best_params,true_params,design_lo,design_hi}` | **R: matches** |
| §3.10: all two-arm table entries; forward ranges3.19–4.84/1.58–2.11 | Both arms; `pub_vs_legacy_v10` | Counts **R**, forward summaries **A: match** |
| §3.10:44.1pp; Newcombe[35.2,52.1]; Fisher2.3×10⁻²⁰; structurep; cluster[33.0,54.8] | `pub_vs_legacy_v10.pooled_difference_pp`, `per_structure` | Difference/Fisher **R**; intervals match artifacts |
| §3.10/conclusion: sixfold |95/16| **R:5.9375**, reasonable rounding |
| §3.11: all60-design shift/count/flip/Pearson/McNemar/Wilcoxon results | `rcwa_*_v8_o5` versus adaptive; `cross_solver_v11` | **R: matches every stated result** |
| §3.11:16–29s versus156–3526s;10–120× | `cross_solver_v11.per_structure.*.median_elapsed_s_*` | **R as rounded structure-median summaries**, not individual ranges |
| §3.11: legacy31/60 | Legacy seed42 spectra | **R: matches**; decomposition remains unsupported |
| §3.12: D6parameters,11features,494/500,400–1800nm,50targets,seed42 | D inverse/stats/finetune metadata | Dimensions/features/targets **R**; reliable-count **A** |
| §3.12: D creation date; dataset/checkpoint hashes | Manifest and original dated inputs | Hash strings **A**; files absent, date **U** |
| §3.12: r+.905, amplitude10.51%, forward1.30% | `structure_d_v10.diagnostic.D`; `finetune_D.final_test_mae_pct` | **A: matches** |
| §3.12: band5–15%=3–7/50; falsification below3.4/above15;42/50 existed | `structure_d_v10.prediction`; contemporaneous record required | Band/arithmetic match; chronology **U** |
| §3.12:4/50,8%,CI[3.2,18.8]; MAEs5.3/6.4/12.3/15.2; w278>L181 | D raw spectra and geometry | **R: matches** |
| §3.12: infeasible20/50; all4; feasible0/30; Fisher.021 | D geometry/status | **R: matches** |
| §3.12:7Δflags, threshold2.61;ρ+.51,p.0002;T4zero/mean.92/max2.27 | D spectra; `reliability_D` | Counts/ρ/T4 summaries **R**; p **A** |
| §3.12:45/50 atN17; median1505s | `rcwa_D.{orders_min,orders_max,elapsed}` | **R: matches** |
| §3.12: both four-point sequences, rates, trendp.32/.97 | `structure_d_v10.orderings`; recomputed trend | **R: matches; both nonmonotone** |
| §§4.1–4.5: repeated counts, correlations, controls, T4 and timing | Corresponding rows above | Same outcomes; detector/feasibility interpretation errors remain |
| §4.5: legacyA9pretenders/0flags/threshold9.69; pubB7/3,A2/1; thresholds3.4–4.2 | Both arms’ seed42 arrays and `tier1b` | **R: matches**, explicit contrast |
| §4.7: P/λmin maxima1.5/2/2 | `stats_*.design_hi`; wavelength minima | **R: matches** |
| §4.7: legacyC300 versus pub350; bounds; “a third” outside normalized box | Both arms’ stats/geometry/finetune artifacts | Training/bounds match; approximate historical fraction not separately established here |
| Conclusion:16+4 events;31%;95/179; solver times | Sources above | Counts **R**; universal feasibility/physical-certainty claims false |
| Conclusion: optimization1.5–2.3min/design | Inverse runtime evidence required | **U** from inspected artifacts |
| Compute/end matter: GPU4070TiSUPER; torcwa0.1.4.2; legacy16–39s | `reliability_*.meta.env`; RCWA metadata/elapsed | Hardware/version **A**; elapsed **R** |
| Acknowledgements: AI/model/CLI versions and review scope | Authorship/process records required | **U**; not established by scientific artifacts |

One useful distinction is already correct: **Δ flags are not pretender classification**. Recomputing their pooled confusion matrix gives **12 true positives, 12 false positives, four missed pretenders, and 151 true negatives**. Preserve §4.5’s distinction wherever “Δ audit” is used as shorthand for certification.

## 5. Files read

| Assigned file | Lines | Function | Entire file read |
|---|---:|---|---|
| `paper/manuscript_v11.md` | 1,548 | Main scientific claims and interpretation | Yes |
| `paper/supplementary_v11.md` | 761 | Protocol and detailed two-arm accounting | Yes |
| `paper/mlst/cover_letter.md` | 45 | Editorial summary of the claims | Yes |
| `scripts/check_manuscript_v10.py` | 380 | Automated manuscript checks | Yes |

Supporting reads were restricted to the cited result artifacts, corresponding numeric arrays, provenance metadata/checkpoints, and relevant figure artifacts. Earlier review documents were neither read nor cited.
