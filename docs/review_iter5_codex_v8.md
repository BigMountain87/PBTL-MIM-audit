codex
## 1. Verdict

Weak reject / major revision for the planned paper, even assuming pending runs are consistent. The v8 pipeline fixes the prior fatal bugs well enough to make a paper possible, but the current claims draft still overstates mechanism, convergence certainty, and cross-structure generality. The strongest defensible paper is: “under this preregistered v8 protocol, all three transferred surrogates self-certify every inverse design, while RCWA rejects about half; the preregistered r-ordering is not supported except for Δ-flag counts.”

## 2. Must-Fix

1. **Convergence qualifier is mandatory for max-severity claims.** Order-7 spot checks shift top-error MAEs by up to **5.54 pp**:
   A max diff 0.49 pp; B max diff 5.54 pp; C max diff 5.42 pp. Do not claim “worst-case true error 31.6%” without saying it is order-5 RCWA pending full order-7 validation.

2. **C4 mechanism claim is overclaimed and partly numerically inconsistent.** Random baseline counts are A/B/C = **10/14/16** pretenders, but best-of-8 gradient is A/B/C = **9/9/13**, not “9/13/9” unless you silently switch structure order. Restart-0 is **8/9/9**, which does not “match” best-of-8 for C.

3. **Protocol/code mismatch for restart seeds.** Protocol says restart seeds are `rng(1000 + target_idx)`; [inverse.py](<PBTL-MIM working copy>/InverseTL_review/src_v8/inverse.py:96) uses `1000 + orig_dataset_idx`. This can change all inverse-design endpoints if rerun according to the frozen protocol.

4. **Do not say “exact p” for Spearman.** [reliability.py](<PBTL-MIM working copy>/InverseTL_review/src_v8/reliability.py:38) uses SciPy `spearmanr`, not an exact permutation p-value, and the CI is an approximate Fisher-z/Bonett-Wright calculation.

## 3. Should-Fix

C2 says “all pairwise Fisher p > 0.3 except Δ-flag A-vs-B p=0.020.” False as written: Δ-flag A-vs-C is **p=0.231**. It is nonsignificant, but not >0.3.

C5’s “~17 s per design” is only B-like. Current RCWA elapsed means are A **35.97 s**, B **18.16 s**, C **36.04 s**.

Seed-42 finetune JSONs lack a `seed` field even though current code writes one. That suggests artifacts were produced by a slightly older script than the checked-in source.

## 4. Minor

“Universal miscalibration” is too broad unless explicitly scoped to these three Paper-1 structures and available seeds.

“Multi-start loss dispersion small” is defensible for loss std, but endpoint spread is large: A **1.98**, B **1.64**, C **1.62** in u-space max-pair distance mean.

## 5. Claims Audit

C1 — **SUPPORTED with caveats.** Seed-42: surrogate pass **60/60**, RCWA pass **29/60**, pretenders **31/60**; Wilson CI for 31/60 = **[39.3%, 63.8%]**. Forward test MAE: A **4.84%**, B **3.19%**, C **3.86%**. Worst order-5 RCWA error is B **31.64%** at surrogate **0.61%**. Largest amplification is A **54.70x**, not the same sample.

C2 — **OVERCLAIMED / one wrong p-value.** H1a not monotone: A/C/B oracle success **11/7/11**. H1c not monotone: rho **+0.185/-0.036/-0.033**. H1b monotone: flags **0/3/6**. Fisher: success p **0.341, 0.341, 1.000**; pretender p **0.341, 0.341, 1.000**; flags p **0.231, 0.451, 0.020**.

C3 — **SUPPORTED for available seeds, overbroad for pending seeds.** Available rho range is B42 **-0.033**, B123 **-0.107**, C42 **-0.036**, C123 **+0.529**, A42 **+0.185**. Threshold calibration is broken in all available reliability runs: surrogate pass **20/20** each.

C4 — **OVERCLAIMED / partly wrong.** Random pretenders A/B/C **10/14/16** versus gradient **9/9/13**. Restart-0 pretenders **8/9/9**. This supports “surrogate argmin selection is dangerous,” but not the stronger “not restart selection” claim, especially for C.

C5 — **OVERCLAIMED.** T4 is indeed **0/60**; T4 max perturbation MAE: A **1.55%**, B **2.87%**, C **1.89%**. But T2 flags are abundant (**15/14/11** A/B/C), endpoint spread is large, and RCWA cost is not ~17 s except B.

C6 — **SUPPORTED.** A has **9 pretenders**, **0 Δ-flags**, threshold **9.69%**, max Δ **7.38%**. Across seed-42, Δ-flags catch only **9/31** pretenders.

## 6. Weakness-Handling Audit

W1 — Partly sufficient. Disclose both r sets, but corrected r makes the preregistered H1 less clean. Treat corrected-r analysis as sensitivity/exploratory unless Paper 1 correction is public first.

W2 — Insufficient for broad cross-structure claims. Dimensionless metrics do not remove wavelength-band/physics confounding. A broadband A rerun should be required if “universal” remains.

W3 — Sufficient only with cautious language. Current C1/C4/C5 wording is not cautious enough.

W4 — Sufficient if framed as companion/testbed evidence, not general SciML law.

W5 — Partly insufficient. Range restriction is not just attenuation; all designs passing surrogate τ makes threshold calibration degenerate.

W6 — Disclosure is necessary but not enough for C. The C out-of-range normalization quirk is a plausible failure driver; a corrected-bounds ablation would blunt reviewer attack.

## 7. Code Audit

[inverse.py](<PBTL-MIM working copy>/InverseTL_review/src_v8/inverse.py:96): restart RNG uses original dataset index, not protocol target index. Consequence: rerunning with frozen protocol can change endpoints, counts, and max errors.

[reliability.py](<PBTL-MIM working copy>/InverseTL_review/src_v8/reliability.py:38): Spearman p-values are not exact. Consequence: reported inference language is too strong.

[random_baseline.py](<PBTL-MIM working copy>/InverseTL_review/src_v8/random_baseline.py:28) and [restart0_validate.py](<PBTL-MIM working copy>/InverseTL_review/src_v8/restart0_validate.py:24): controls are hardwired to seed-42 artifacts. Consequence: C4 is seed-42 only.

[rcwa_validate.py](<PBTL-MIM working copy>/InverseTL_review/src_v8/rcwa_validate.py:22): order override mutates module settings, but top-sample order-7 artifacts show large B/C shifts. Consequence: order-5 severity numbers are not convergence-stable yet.

## 8. Knockout Question

If the claimed failure is a property of transferred surrogates rather than of C’s normalization mismatch, cross-band structure differences, and order-5 RCWA noise, why do the mechanism and severity claims survive only as seed-42/order-5 summaries, with C changing materially under restart-0 and B/C top errors moving by ~5 percentage points under order-7?
