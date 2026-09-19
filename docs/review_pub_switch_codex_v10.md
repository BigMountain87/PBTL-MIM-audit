## 1. Verdict

**Major revision, with a viable but materially different paper.** The strongest defensible result is: under the published pipeline, all 179 solver-valid committed designs pass the surrogate’s 5% check, but 16 fail reference-solver validation; this demonstrates that the threshold check does not guarantee success. The legacy comparison documents a large difference between two complete pipelines, not an identified improvement in surrogate quality. Crucially, the pub results also contain evidence **for** the preliminary-*r* incidence ordering and strong error discrimination on C. Those findings must be reported alongside non-certification: useful prediction and imperfect certification can coexist. Discounting them to preserve the earlier narrative would weaken the paper.

## 2. Must-Fix

1. **The two-arm contrast survives descriptively, but its causal interpretation does not.**

   All per-run entries in `results_pub/pub_vs_legacy_v10.json:per_run` reproduce from the source artifacts. The totals **95/179 versus 16/179**, and the **44.134 pp** difference, are correct.

   However, [interim_pub_vs_legacy.md:24](docs/interim_pub_vs_legacy.md:24) and [PLAN_v10_amendment_pub.md:106](docs/PLAN_v10_amendment_pub.md:106) overinterpret this as pipeline correction improving reliability. The outcome’s reference standard, target distribution, checkpoints, training pools, A’s wavelength band, and C’s feature representation/bounds all change. Even the forward MAEs are measured against different test distributions and reference labels.

   **Defensible wording:** “The pipeline-specific pretender fraction was 53.1% in the legacy execution and 8.9% in the published-pipeline execution. This bundled comparison does not identify contributions from surrogate training, target composition, physical modelling, or numerical fidelity.”

   A more accurate reference solver **could** reduce disagreement without a change in surrogate quality; it could also increase disagreement. Higher Fourier order has no monotonic relationship to MAE against a fixed target. Changing the target-generating solver adds another source of change.

   **Separation using existing inverse runs:** freeze archived committed geometries, surrogate predictions, target spectra, and wavelength grids; cross-evaluate those geometries with both reference configurations. To isolate truncation specifically, vary order while fixing materials, precision, grid, and other physics settings. Report paired MAE shifts and classification transitions. This requires additional reference calls, but no retraining or inverse optimization. Existing legacy order-7 artifacts already give A **9→10**, B **9→9**, C **13→16** pretenders at seed 42; they do not support a generic “higher order removes pretenders” explanation, but cannot resolve the full legacy/pub confounding. A common-oracle comparison still needs common targets to identify a comparable inverse-task improvement.

2. **The Newcombe and Fisher arithmetic is correct; the advertised inferential interpretation needs qualification.**

   [pub_vs_legacy_v10.py:39](scripts/pub_vs_legacy_v10.py:39) correctly implements the independent-proportions Newcombe hybrid-score interval. I reproduce **[35.2375, 52.0526] pp** and Fisher **\(p=2.27075\times10^{-20}\)**. Newcombe is appropriate for independent binomial samples; it does not itself accommodate the repeated targets. See the [statsmodels definition](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html).

   “Targets differ, so the runs are independent” at [pub_vs_legacy_v10.py:95](scripts/pub_vs_legacy_v10.py:95) is too strong. Different targets establish that there is no natural design pairing between arms; they do not establish all independence assumptions.

   The demonstrable dependence is **the same target repeated across training seeds**, not simply “20 targets within a run.” Conditional on a fixed trained model, distinct sampled targets need not constitute one correlated cluster merely because they share the model. Generalizing over training realizations introduces another level of uncertainty, with only three seeds.

   Label the Wilson intervals, Newcombe interval, and Fisher tests **nominal, ignoring repeated-target dependence**, including per-structure tests. The JSON already partly does this in `pooled_difference_pp.note`; the generated Markdown omits that note ([script:136](scripts/pub_vs_legacy_v10.py:136)).

   As a sensitivity check, independently resampling targets within each arm and structure, retaining their seed vectors, gave a **40,000-replicate percentile interval of approximately [33.3, 54.5] pp** for the difference. Thus the descriptive separation remains large after target clustering. This interval is conditional on the observed structures and trained models and does not remove confounding.

3. **The preliminary-*r* trend survives as positive evidence; the proposed dismissal does not.**

   For your stated rank scores:

   \[
   T=6.9106145,\qquad V=9.6856350,\qquad
   z=2.2205094,\qquad p_{\text{two-sided}}=0.0263842.
   \]

   This calculation is correct. An independent-observation, fixed-total exact calculation using absolute score extremeness gives approximately **0.0356**, so discreteness alone does not erase it.

   The expected correction in [interim_pub_vs_legacy.md:125](docs/interim_pub_vs_legacy.md:125) is unsupported. Applying the existing ICC estimator to pub pretender indicators gives:

   | Structure | ICC | \(1+2\,ICC\) |
   |---|---:|---:|
   | A | −0.019 | 0.962 |
   | B | 0.103 | 1.207 |
   | C | 0.791 | 2.582 |

   A excludes its incomplete target row for this ICC calculation. C’s five events occur on only **two distinct targets**; B’s nine occur on seven. These are substantially different from legacy’s ICCs.

   C occupies the middle score, so its large ICC contributes little to the linear trend’s variance. A null-variance inflation sensitivity calculation, conservatively flooring A’s design effect at 1, gives effective inflation **1.103**, \(z\approx2.114\), **\(p\approx0.0345\)**. This is an approximation, not a replacement confirmatory test, but it directly contradicts the expectation that clustering necessarily moves this result to 0.06–0.10.

   Also distinguish the **pre-specified ordering** from the **post hoc pooled trend test**: the manuscript itself classifies pooled analysis as exploratory ([manuscript:415](paper/manuscript_v10.md:415)). Specify multiplicity families consistently; do not introduce a correction selectively because this result is inconvenient.

   **Defensible wording:** “Pretender incidence follows the preliminary-*r* ordering in these three structures, with nominal evidence of a trend. Inference is limited by repeated targets, sparse events, and only three structure-level diagnostic values; this does not establish general predictive validity or design-level certification.”

   “The study cannot order three structures at this event count” is too categorical. The ordering is observed, and there is evidence for it. What the study cannot establish is a broadly generalizable calibration law.

4. **There is a second, stronger reversal: C’s self-report is informative.**

   From `results_pub/rcwa_C{,_s123,_s777}_v8.npz`, excluding failed calls:

   | Seed | Spearman \(\rho(\mathrm{RCWA},\mathrm{Surr})\) | Nominal p | Holm p across nine runs |
   |---|---:|---:|---:|
   | 42 | 0.7504 | 0.0001384 | 0.001107 |
   | 123 | 0.8286 | 0.00000640 | 0.0000576 |
   | 777 | 0.7188 | 0.0003559 | 0.002491 |

   Thus [manuscript:569](paper/manuscript_v10.md:569), “no usable rank information” and “neither survives a nine-test correction,” becomes **false**, not merely underpowered.

   C’s pooled empirical AUROC using the surrogate MAE alone is **0.9855**, versus the manuscript’s 0.68 at [line 722](paper/manuscript_v10.md:722). That estimate rests on only two distinct positive targets and does not establish a deployable certification threshold. Nevertheless, it prevents carrying over the blanket detector-failure narrative.

   Replace “self-certification is uninformative” with the precise distinction: **the fixed 5% pass/fail indicator does not distinguish failures because everyone passes; the continuous score can still discriminate.**

5. **Target pairing is supported by the intended pipeline, but `orig_indices` is not sufficient verification.**

   The intended chain is correct:

   - [finetune.py:34](src_v8/finetune.py:34) loads and splits before branching on initialization.
   - [common.py:286](src_v8/common.py:286) fixes test/validation membership with `SPLIT_SEED`; `subset_seed` affects training membership only.
   - [inverse.py:58](src_v8/inverse.py:58) samples the test split using fixed `TARGET_SEED`.
   - Feasibility changes the parameterization, not target selection.

   A changed `n_good` within a profile normally raises at [common.py:279](src_v8/common.py:279). But equal counts and row IDs do not establish equal data: changed spectra or reordered/replaced pools can preserve IDs. More concretely, `inverse.py` uses the archived fine-tune index map while reading target spectra from a **fresh `load_filtered()` call** ([inverse.py:41](src_v8/inverse.py:41), [131](src_v8/inverse.py:131)). Same-count filtering drift can therefore preserve recorded IDs while changing actual targets.

   There is also a control-identity hazard: [common.py:307](src_v8/common.py:307) selects a suffix based on **any structure’s** matching artifact and silently falls back to untagged inputs. If all M0 fine-tunes fail, the gate driver continues and inverse can load the base TL model under an `_m0` output name. Pairing passes perfectly, but it is not an M0 control. Partial tagged availability will generally fail noisily for missing structures instead.

   Require equality of wavelength arrays and every archived `A_target_*` channel, matching dataset provenance, and consistency between inverse and RCWA artifacts. For M0 require `init=="scratch"` and tagged model provenance; for feasibility require the feasible flag and intended base surrogate. Current local control outputs are absent, so actual control pairing cannot yet be verified.

6. **The analysis change is defensible as a disclosed amendment informed by base outcomes—not as an untouched pre-data plan.**

   [PLAN_v10_amendment_pub.md:117](docs/PLAN_v10_amendment_pub.md:117) says that no control outputs existed. That supports prospective specification relative to the **unseen control comparisons**, but the base outcomes are part of those comparisons and were already observed. “Therefore not post hoc” at line 122 is overly binary.

   Disclose together:

   - Original binary endpoint and three-seed control scope.
   - September 6 pipeline switch; September 7 reduction to seed 42 after observing 6/60 base pretenders.
   - September 8 endpoint promotion, including exactly which base results had been inspected.
   - Evidence that control outcomes were unavailable; archive dated code/logs, rather than relying on a mutable `missing` field.
   - Frozen estimand, direction, test settings, missing-pair handling, multiplicity across M0/feasibility, and reporting rules.
   - Retention of binary results irrespective of significance.

   **Defensible label:** “Analysis-plan amendment informed by base-arm results, fixed before control outcomes were available.”

   Distinguish the T55 primary outcome from the original audit’s primary discrimination statistic in [manuscript:342](paper/manuscript_v10.md:342). Any later expansion chosen after inspecting T55 Δ ([amendment:155](docs/PLAN_v10_amendment_pub.md:155)) needs an adaptive-inference plan or an exploratory label.

7. **`mde_binary()` simulates one valid paired scenario, not paired power generally; the quoted MDE is wrong under that scenario.**

   At [control_analysis_v10.py:45](scripts/control_analysis_v10.py:45), independent Bernoulli draws impose **zero within-pair association**. Applying McNemar is valid in that special case. The conditional null, equal probabilities of the two discordance directions, is correct.

   However, marginal rates do not determine paired power. Let \(q=P(\text{base}=1,\text{control}=1)\). Then discordance probabilities are \(p_0-q\) and \(p_1-q\). Power depends on \(q\), as described in [Stata’s paired-proportions documentation](https://www.stata.com/manuals/pss-2powerpairedproportions.pdf).

   Exact unconditional calculations for 60 pairs, \(p_0=0.10\), two-sided α=0.05 give approximately:

   | Association assumption | Control rate needed for 80% power |
   |---|---:|
   | Maximum concordance | 23% |
   | Independence—the script’s assumption | 32% |
   | Minimum concordance | 35% |

   Under independence, a rise to 28% has only **64.2%** power. The **27–29%** statement at [amendment:148](docs/PLAN_v10_amendment_pub.md:148) does not survive.

   Report sensitivity to feasible joint probabilities, both increases and reductions, the absolute difference as well as resulting rate, and simulation precision. “Cannot resolve anything below the MDE” is also false: 80% power is a probability criterion, not a detection boundary.

8. **The continuous primary needs a precise estimand and cannot stand alone as inverse-quality evidence.**

   [control_analysis_v10.py:87](scripts/control_analysis_v10.py:87) and [110](scripts/control_analysis_v10.py:110) use signed-rank tests, while reporting the median paired shift and its bootstrap interval. Signed-rank tests a symmetric zero-centered difference distribution; it is not an assumption-free test of the median. State its assumptions and distinguish the test from the median interval. See [SciPy’s documentation](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.wilcoxon.html).

   More fundamentally,
   \[
   \Delta_{\rm control}-\Delta_{\rm base}
   =(\mathrm{RCWA}_{\rm control}-\mathrm{RCWA}_{\rm base})
   -(\mathrm{Surr}_{\rm control}-\mathrm{Surr}_{\rm base}).
   \]
   Δ can improve merely because the control reports worse surrogate MAE, even if reference MAE is unchanged or worsens. This is particularly relevant to constrained optimization. Report paired shifts in **both component MAEs**, claimed success, and pretender status.

   The claimed continuous power **0.84** at [amendment:143](docs/PLAN_v10_amendment_pub.md:143) lacks a specified effect distribution/scale and is not implemented in this script. Substantiate it or remove it.

9. **The manuscript requires a substantive rewrite, not a headline substitution.**

   The following inventory identifies stale or unsupported claims caused by the switch. “Recompute” means the legacy result may remain in supplementary material, but cannot substantiate pub claims.

   | Manuscript location | Required change |
   |---|---|
   | [14–37](paper/manuscript_v10.md:14), abstract | Replace 84/95/53%, threshold counts, ≥45% across structures, order-7 robustness, ordering narrative, and mechanism/detector conclusions. **Ordering and discrimination change direction.** |
   | [150–170](paper/manuscript_v10.md:150), §1.4 | Preserve original hypotheses but disclose amendments. Non-significance would not itself establish non-certification; positive ordering is now observed. |
   | [174–184](paper/manuscript_v10.md:174), contributions | “Over half,” no supported ordering, mechanism and detector conclusions cannot carry over. |
   | [208–253](paper/manuscript_v10.md:208), §2.1 | Replace datasets/hashes/checkpoints, C’s 13 features with 18, A’s band, filtering, C’s training count/bounds, solver provenance and legacy-as-main framing. |
   | [336–356](paper/manuscript_v10.md:336), §2.5 | 5% is no longer approximately the forward-MAE scale; retain it as the fixed tolerance. Add T55 amendment separately from original primary statistic. |
   | [382–424](paper/manuscript_v10.md:382), §2.7 | Add September timeline and analysis-status rows for pub execution, two-arm contrast, and amended controls. June artifact timestamps do not document pub execution. |
   | [430–449](paper/manuscript_v10.md:430), results opening/§3.1 | Legacy bitwise reproducibility evidence cannot certify pub runs. Seed-42 forward MAEs become B/A/C **1.722/1.757/2.107%**; thresholds change accordingly. |
   | [453–474](paper/manuscript_v10.md:453), Table 6 | Replace solver settings and all derived rows except the verified 20/20 claimed counts. B/A/C confirmed **17/19/18**; pretenders **3/1/2**; Δ-flags **7/2/1**. |
   | [476–491](paper/manuscript_v10.md:476) | “None agrees more than 55%” becomes **85–95% agreement**; “roughly half” becomes 8.9% pooled. Separate binary saturation from continuous discrimination. |
   | [496–537](paper/manuscript_v10.md:496), §3.3/Table 7 | Recompute all orderings/tests/common-threshold counts. **H1a becomes monotone under preliminary r; published-r success becomes nonmonotone; amplitude-ordered incidence loses monotonicity.** Preliminary-r seed-42 max Δ also loses its former monotonicity: A/C/B **9.90/4.66/12.11 pp**. |
   | [548–567](paper/manuscript_v10.md:548), Table 8/pooled accounting | B/A/C pretenders **9/60, 2/59, 5/60**; confirmed **51/60, 57/59, 55/60**; pooled flags **24/179**. Missing call is **A777**, not C777. Replace ICCs/CIs. Threshold counts become **7/179 at 7.5%**, **3/179 at 10%**; severity bands **9 mild, 4 moderate, 3 severe**. |
   | [569–595](paper/manuscript_v10.md:569), discrimination/figure captions | **Strong reversal on C**, detailed above. Replace correlations, significant-run count, AUROCs, and threshold-sweep denominator annotations. |
   | [600–638](paper/manuscript_v10.md:600), §3.5/Figures 5–6 | Move legacy order-7 results and legacy worst-design illustration to SI or label explicitly; neither validates pub severity. |
   | [640–678](paper/manuscript_v10.md:640), §3.6 | All mechanism comparisons require pub controls; no local pub random-baseline artifact was present. |
   | [682–735](paper/manuscript_v10.md:682), §3.7 | Recompute taxonomy, associations, detector table and costs. C’s raw-MAE AUROC already contradicts carrying over the negative summary. |
   | [742–770](paper/manuscript_v10.md:742), §§3.8–3.9 | Recompute feasibility cross-tabs, lookup performance and recoverability. Legacy targets/endpoints cannot support pub conclusions. |
   | [785–843](paper/manuscript_v10.md:785), §§4.1–4.4 | Replace 3–5% forward MAE, “more than half,” **“only rate that orders is Δ-flag,”** mechanism/smoothness/detector conclusions and 53% cost-benefit statement. |
   | [883–888](paper/manuscript_v10.md:883), §4.5 | A’s example **reverses**: seed 42 has one pretender and two Δ-flags; the legacy “9 pretenders, 0 flags” example is not pub evidence. |
   | [920–921](paper/manuscript_v10.md:920) | Recompute the 11–16/20 box-edge claim. |
   | [929–984](paper/manuscript_v10.md:929), §4.7 | W1’s no-support claim changes; W2’s unequal bands and W6’s C-pool/bounds quirks disappear for pub. Update W5’s score range and W8’s solver description. W7 remains true while controls are absent, but must change when reported. |
   | [1000–1006](paper/manuscript_v10.md:1000), conclusion | Replace totals, order-7 claim, **“no r convention orders,”** mechanism and detector conclusions. |
   | [1024–1026](paper/manuscript_v10.md:1024), compute | Replace legacy timings and `920b1bd` provenance with measured pub costs and pub code pin. |

10. **The new oracle cache can silently invalidate solver comparisons or resumed runs.**

    [oracle.py:125](src_v8/oracle.py:125) keys cached spectra by kind/structure/suffix/index. On a hit, [line 135](src_v8/oracle.py:135) does not validate geometry, wavelengths, solver settings, materials, precision, or code provenance.

    Consequently, changing `--order` without a fresh tag can reuse old spectra while [rcwa_validate.py:90](src_v8/rcwa_validate.py:90) writes metadata describing the newly requested solver. Recomputed inverse endpoints under an existing suffix have the same risk. This is a concrete new correctness defect, **not evidence that current artifacts are already corrupted**.

    Include and validate a simulation-input/configuration fingerprint before accepting cache hits. This is essential before the cross-solver analysis recommended above.

## 3. Should-Fix

1. **Standardize trend scores.** The interim table uses equally spaced ranks, but [stats_supplement_v9.py:16](src_v8/stats_supplement_v9.py:16) uses actual diagnostic values. Pub nominal p-values using actual values are approximately **0.0261 preliminary r, 0.2686 printed r, 0.9809 amplitude**, versus the interim rank-score values 0.026/0.201/0.350. Neither scoring choice is inherently wrong; report which was chosen and avoid silently comparing different tests.

2. **Do not use “more seeds buy no power.”** [amendment:126](docs/PLAN_v10_amendment_pub.md:126) correctly identifies repeated targets, but repetition can still improve precision and characterize training variability. Pub A/B dependence is modest. The cost-based reduction is defensible; the categorical statistical rationale is not.

3. **The proposed target expansion exceeds the held-out pool.** [amendment:153](docs/PLAN_v10_amendment_pub.md:153) proposes adding 40 targets to the existing 20 from the same test split. [common.py:197](src_v8/common.py:197) provides only 50 test targets, leaving **30**, not 40. Sixty targets require a genuinely new holdout plan.

4. **Make the control inference scope explicit.** The bootstrap at [control_analysis_v10.py:101](scripts/control_analysis_v10.py:101) resamples pooled differences across structures. For a fixed equal allocation of three structures, stratified target-pair resampling better preserves the design. Neither method quantifies variability over trained seeds with seed-42-only controls.

5. **Preserve the adaptive solver’s own limitation statement.** Deleting an order-7 check makes sense when the operative orders are 9–17; it does not establish convergence. The new solver explicitly describes unconverged hard cases at its cap ([rcwa_struct_c.py:65](src_v8/upstream_cb486b5/src/simulation/rcwa_struct_c.py:65)). Report reference-fidelity certification, not convergence-certified physical truth.

6. **M0 is a training-recipe control.** [run_pub_gate.sh:14](scripts/run_pub_gate.sh:14) changes learning rate to \(10^{-3}\), versus TL’s \(3\times10^{-4}\). That is reasonable for reproducing the upstream scratch recipe, but it does not isolate initialization alone.

## 4. Checked and OK

- **Every `per_run` JSON entry reproduced exactly** from the archived RCWA and fine-tune files, including counts, forward MAEs, mean errors, and Δ summaries.
- **All nine pub RCWA artifacts identify adaptive operation, complex64, and `materials="jc"`**; solver-valid MAEs are finite. The CLI’s default `materials="legacy"` does not overwrite the pub module’s default JC selection.
- **179/179 solver-valid designs are surrogate-claimed in both arms.** Thus using claimed rather than valid denominators produces the cited rates correctly for these artifacts.
- **The Newcombe formula, Fisher arithmetic, and supplied Cochran–Armitage arithmetic are correct.**
- **Archived base targets are identical across seeds within each structure and arm**, including original IDs and every target-spectrum channel—not merely implied by the seed settings.
- **The intended TL/M0 split is initialization-independent, and feasibility does not alter target selection.** The concerns above concern artifact identity and drift protection.
- **Exact McNemar via a binomial test on discordant pairs is correctly implemented** at [control_analysis_v10.py:102](scripts/control_analysis_v10.py:102). Intersecting the two validity masks correctly selects complete pairs, provided excluded failures are separately reported.
- **The central non-guarantee survives.** Sixteen pub counterexamples defeat a perfect 5% self-certification guarantee. Positive ordering and useful discrimination on C do not contradict that narrower result.
- No files were modified.
