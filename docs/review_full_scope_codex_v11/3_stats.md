# Full-scope codex review — slice 3_stats (2026-09-12)

## 1. Verdict

**The headline arithmetic reproduces; several inferential interpretations and decision gates need correction.**

I read all **2,010 assigned lines**, without opening earlier review documents or the manuscript. Calculations used the assigned code and its NPZ/JSON data artifacts, executed with NumPy 2.3.5 and SciPy 1.15.1. No repository files were changed.

The structure mapping is **B: 9/60, A: 2/59, C: 5/60**. The pooled result is **16/179 = 8.94%**, with all 179 valid designs surrogate-claimed.

## 2. Must-Fix

**1. The paired control report presents a median and a signed-rank test as though they address the same estimand. They generally do not.**

Source: `scripts/control_analysis_v10.py:99–105,114–123,141–143`.

Wilcoxon signed-rank is appropriate for independent paired differences under its symmetry/location assumptions. It is not a distribution-free test of the ordinary median. A bootstrap interval for the median can be reported alongside it, but it is not the corresponding confidence interval for that test.

The feasibility differences are markedly asymmetric:

| Recomputed quantity | Feasibility control − base |
|---|---:|
| Median | −0.001122 pp |
| Median bootstrap 95% interval | [−0.209874, −0.000053] pp |
| Mean | −0.793077 pp |
| Hodges–Lehmann estimate, median of Walsh averages | −0.181118 pp |
| Range | [−10.660761, +4.605309] pp |
| Negative / positive differences | 40 / 20 |
| Signed-rank p | 0.0070527 |
| Two-sided sign-test p | 0.0134893 |

Thus, significance with a near-zero median is **not itself a contradiction**. There is directional evidence even from the sign test. But the supplied phrase “consistent but negligible” is defensible only if carefully restricted to the **pooled median**, with an explicit practical margin. It is not supported as a general description of the effect:

- B’s median change is **−0.545232 pp**.
- A’s is **−0.154964 pp**.
- C’s is **−0.0000279 pp**.
- The pooled median interval permits reductions approaching **0.21 pp**.
- Neither significance nor rounding to “−0.00” establishes practical equivalence.

Choose the estimand, report compatible inference, preserve sufficient precision, and replace an unqualified negligible-effect claim with a quantitative statement.

**2. Detector bootstrap intervals break the repeated-target dependence.**

Source: `src_v8/detector_bench.py:139–154`.

Each seed receives independently sampled target indices. The same target across seeds must instead travel together when estimating uncertainty over targets.

This matters in the actual artifacts: **C’s five pretenders occur on only two unique targets**, and its pretender ICC is **0.7912**. The bootstrap treats their repeated appearances as more independent evidence than the experimental design provides.

The arithmetic of all 36 intervals reproduces, but that does not validate their coverage. Resample target clusters within structure, retain seed alignment and failed-observation masks, and recompute the stratified AUROC within each replicate.

**3. Dependence-naive intervals and tests are not labelled nominal everywhere.**

Concrete omissions:

| Location | Problem |
|---|---|
| `make_evidence_json.py:142,159–167` | Pooled Wilson intervals lack an explicit independence/nominal qualification. |
| `detector_bench.py:152–160` | JSON and generated table present the seed-independent bootstrap intervals as ordinary 95% intervals. |
| `pub_vs_legacy_v10.py:97–107,131–144` | The difference has a nominal caveat in JSON, but the generated Markdown omits that note; arm intervals and per-structure Fisher tests also lack local qualification. |
| `structure_d_v10.py:58–59,109,118–132` | Three-seed A/B/C Wilson intervals and four-point trend tests are presented without accounting for repeated targets. |
| `feasibility_crosstab_v8.py:135,143` | Pooled Fisher and run-stratified CMH results ignore dependence between runs sharing targets. |

Stratifying CMH by run does **not** remove dependence between those strata.

By contrast, the paired controls use only seed 42, so their bootstrap does not have this particular repeated-seed defect. The target-cluster bootstraps in the pooled evidence and statistics supplement preserve the relevant grouping.

**4. The gate summary contradicts the paired control analysis and compares unmatched seed coverage.**

Source: `scripts/gate_summary_v10.py:41–51,104–132,172–195`.

The gate pools every available seed separately by condition. Current artifacts yield:

| Structure | Base pretenders/valid | M0 | Feasibility |
|---|---:|---:|---:|
| A | 2/59 | 2/20 | 0/20 |
| B | 9/60 | 1/20 | 1/20 |
| C | 5/60 | 1/20 | 2/20 |

It then applies independent-proportion Newcombe/Fisher methods, despite the controls sharing seed-42 targets with the base. These are neither the matched comparisons nor independent samples.

There are two additional concrete gate errors:

- At lines **121–132**, `n_sig` counts any two-sided significance, without checking the direction required by “TL has significantly more pretenders.”
- At lines **188–195**, feasibility is declared to “collapse” merely because its rate is below 25%. The current base rate is already **8.94%**. The feasible rate **3/60 = 5%** triggers “collapse” without testing a relative reduction.

Make the gate consume the paired analysis and express branch criteria relative to the appropriate matched baseline.

**5. The reported multiplicity correction does not cover the expanded inferential claims.**

Sources: `make_evidence_json.py:320–329`; `stats_supplement_v9.py:200–207`; `control_analysis_v10.py:100–127`; `detector_bench.py:154`; `gate_summary_v10.py:150–166`.

The implemented Holm corrections cover selected Spearman/Fisher families. They do not cover the added controls, convention-dependent trends, feasibility associations, arm comparisons or detector interval screening.

A concrete sensitivity calculation: treating the **eight control Wilcoxon tests plus two McNemar tests** as one family gives feasibility’s pooled p:

**0.0070527 → Holm-adjusted 0.0634741.**

That is not an argument that every result must belong to one enormous family. It shows why the family definition changes the conclusion. Specify confirmatory families and any hierarchical testing rule; label remaining results exploratory. The gate’s search across **36 detector×structure intervals** also needs to be acknowledged as selection, not pointwise confirmation.

## 3. Should-Fix

**The paired power simulation is valid, but its presentation and numerical stability need improvement.**

Source: `scripts/control_analysis_v10.py:45–69,128–133`.

The four-cell probabilities correctly implement the stated concordance assumptions. I reproduced both simulations and independently integrated exact McNemar power:

| Concordance | Archived M0 threshold | Archived feasibility threshold | Exact first 1-pp grid threshold |
|---|---:|---:|---:|
| Maximum | 23% | 23% | 23% |
| Independence | 32% | 33% | 32% |
| Minimum | 35% | 35% | 35% |

Under independence, exact power at a 32% control rate is **0.80003355**. Monte Carlo noise therefore explains the 32%/33% discrepancy between otherwise identical power scenarios.

Fixes:

- Use exact power here; the problem is small.
- Report rate differences as well as control rates: **+13, +22, +25 pp** on this grid.
- Explicitly label the search as **increases only**. It does not assess power to detect the reductions the feasibility control might produce.
- Avoid implying that continuous-outcome power has been calculated. Despite the docstring’s “Both” wording, only binary power is supplied.

For context, even a reduction from 10% to zero has only **56.28%** exact McNemar power at n=60.

**The “DEFF-corrected” trend calculation is a heuristic, not a demonstrated cluster-valid test.**

Source: `stats_supplement_v9.py:111,120,149–151`.

Dividing each structure’s event count and sample size by a different estimated design effect changes the weighting and pooled null proportion. It is not simply a variance correction to the original trend statistic. Label it an effective-sample-size sensitivity analysis, or replace it with inference that respects the target clusters.

The negative estimated ICC for A also produces **n_eff = 61.31 from 59 observations**. That can arise algebraically, but should not be presented as established additional information.

**Two denominator safeguards conceal undefined rates, and one hypothesis evaluates counts rather than rates.**

- `structure_d_v10.py:58` returns 0% when no designs are claimed through `max(claimed,1)`.
- `feasibility_crosstab_v8.py:125–126,138–139` similarly returns zero for an empty subgroup.
- `synthesize.py:88–95` evaluates success/flag **counts**, although the hypotheses concern **rates**.

Return null for empty denominators and evaluate rate hypotheses using their actual denominators. These do **not** explain an error in the present 16/179 headline: all main-arm valid designs are claimed, and the seed-42 synthesis denominators are all 20.

**D cannot yet be scored.**

`results_pub/structure_d_v10.json:structure_D` is null, and `rcwa_D_v8.npz` is absent. The script correctly reports that status. Its eventual binary “falsifying” label should be distinguished from a statistical rejection: `structure_d_v10.py:94–99` compares only the observed point estimate against the band.

## 4. Checked and OK

**Headline and reference errors.** I recomputed reference MAEs directly from archived reference and target spectra, checked them against NPZ MAEs and reliability JSON arrays, and verified target alignment across seeds.

| Structure | Seed 42 | Seed 123 | Seed 777 | Pooled |
|---|---:|---:|---:|---:|
| A | 1/20 | 0/20 | 1/19 | 2/59 |
| B | 3/20 | 3/20 | 3/20 | 9/60 |
| C | 2/20 | 1/20 | 2/20 | 5/60 |
| Total | 6/60 | 4/60 | 6/59 | **16/179** |

**Tau sweep.** Recomputed counts and the specified 2,000-replicate target-cluster intervals reproduce:

| τ (%) | Pretenders / claimed | Cluster-bootstrap 95% rate interval |
|---:|---:|---:|
| 2.5 | 53/171 | 22.22–39.77% |
| 3.75 | 32/177 | 11.17–26.01% |
| 5 | 16/179 | 4.43–14.53% |
| 7.5 | 6/179 | 0.56–6.70% |
| 10 | 3/179 | 0–3.89% |
| 15 | 1/179 | 0–1.69% |

The separate 20,000-replicate pooled cluster interval also reproduces: **3.954–14.525%**. The nominal Wilson interval is **5.58–14.03%**.

**ICCs and design effects.** All reproduce:

| Structure | ICC | DEFF | Effective n |
|---|---:|---:|---:|
| A | −0.018868 | 0.962264 | 61.3137 |
| B | 0.103371 | 1.206742 | 49.7207 |
| C | 0.791209 | 2.582418 | 23.2340 |

Reported pooled DEFF: **1.333150**. A’s ICC uses 19 complete target clusters; B and C use 20.

**Cochran–Armitage.** Recomputed pretender z and two-sided p values match the artifact:

| Convention | Nominal z; p | Implemented DEFF z; p | Seed-42 z; p |
|---|---|---|---|
| Preliminary | −2.224165; .026137 | −2.174286; .029684 | −1.053839; .291956 |
| Intermediate | −0.421144; .673650 | −0.756331; .449451 | −0.292353; .770017 |
| Printed | +1.106194; .268642 | +1.222606; .221478 | +0.440135; .659840 |
| TMM MAE | −0.023962; .980883 | −0.261586; .793640 | −0.106843; .914913 |

Because all valid designs are claimed, oracle-pass outcomes are complementary: opposite z, identical two-sided p.

**Detector AUROCs and bootstrap intervals.** All 36 pooled results reproduce from archived per-design detector values and labels, using the original RNG sequence:

| Detector | A | B | C |
|---|---|---|---|
| Ensemble SD | .730 [.421, 1] | .804 [.636, .943] | .945 [.846, 1] |
| Held-out MAE | .514 [.211, .812] | .745 [.555, .915] | .978 [.896, 1] |
| Ensemble-mean MAE | .378 [.167, .611] | .725 [.524, .897] | .978 [.913, 1] |
| Training kNN | .730 [.526, .914] | .797 [.651, .922] | .418 [.139, .736] |
| Pool kNN | .730 [.474, .944] | .791 [.637, .925] | .495 [.211, .776] |
| TMM versus surrogate | .622 [.353, .875] | .425 [.233, .632] | .231 [.087, .400] |
| TMM versus target | .622 [.368, .871] | .431 [.229, .637] | .220 [.057, .405] |
| Local perturbation | .919 [.778, 1] | .477 [.216, .745] | .659 [.486, .817] |
| Boundary flag | .662 [.559, .765] | .621 [.486, .730] | .462 [.214, .724] |
| Endpoint spread | .270 [0, .667] | .497 [.299, .698] | .736 [.579, .875] |
| Surrogate MAE | .270 [.105, .471] | .399 [.182, .634] | .978 [.895, 1] |
| Infeasible flag | .743 [.639, .861] | .765 [.684, .845] | .374 [.288, .445] |

These are the **implemented dependence-naive intervals**, not corrected intervals. Detector feature generation through the neural models was not rerun.

**Both paired controls.** Pairing assertions, all per-structure signed-rank p values, pooled median intervals, pooled p values and McNemar results reproduce:

| Control | Median shift [bootstrap 95%], pp | Wilcoxon p | Pretenders | Discordant base/control only | McNemar p |
|---|---|---:|---|---|---:|
| M0 | +.135233 [−.068674, +.443200] | .175566 | 6 → 4 | 4 / 2 | .6875 |
| Feasibility | −.001122 [−.209874, −.000053] | .007053 | 6 → 3 | 4 / 1 | .375 |

**Additional checks.**

- Geometry-derived feasibility tables and their CMH calculations reproduce.
- Recoverability geometry distances and spectral peak shifts reproduce.
- Lookup counts reproduce from archived per-target errors; underlying raw training spectra were not recomputed.
- Arm counts reproduce: legacy **95/179**, pub **16/179**. The implemented nominal Newcombe difference interval is **35.237–52.053 pp**, with Fisher p **2.27075×10⁻²⁰**. Its dependence limitation remains as identified above.

## 5. Files read

The opening table is the complete assigned-file ledger: **11 files, 2,010 lines, all read end to end**. No earlier review document was opened or used.
