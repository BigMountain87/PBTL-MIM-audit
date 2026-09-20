# Reliability of Physics-Transferred Neural Surrogates in Gradient-Based Inverse Metasurface Design: A Reference-Solver Audit

**Sang-Bae Choi¹, Je-Min Choi², Joonhyub Kim², Chang-Mo Kang²**
¹Independent Researcher, Busan, 46264, Republic of Korea
²Department of Nanomechatronics Engineering, Pusan National University, Busan, 46241, Republic of Korea

Corresponding authors: Sang-Bae Choi (sbchoi129@gmail.com) and Chang-Mo Kang (fd1kcm@pusan.ac.kr)

---

## Abstract

Physics-based transfer learning produces data-efficient neural-network surrogates for
nanophotonic absorbers. A companion study introduced a pilot-set transferability
diagnostic that anticipates the forward gain: the shape correlation *r* between a cheap and an expensive solver, read with their amplitude error. We ask how well this diagnostic and the held-out forward error predict whether such a surrogate can serve inside a gradient-based inverse-design loop. We fixed a reliability protocol, its thresholds and three hypotheses in an internal record before the main runs, marked later extensions as such, and applied the protocol to that study's published pipeline: three metal–insulator–metal absorber
structures at three training seeds each, with a rigorous coupled-wave reference solver called once per committed geometry to count pretenders, defined as designs that the surrogate certifies but the solver rejects. All 179 valid committed designs pass the surrogate's 5 % check and
the solver rejects 16 (8.9 %; 95 % confidence interval (CI) [4.0, 14.5] target-clustered, [5.6, 14.0] nominal; one further solve failed), so the pass verdict screened nothing out. At 2.5 % tolerance
53 of the 171 designs still certified are pretenders (31 %); at 7.5 %, 6 of 179. A surrogate trained from scratch produces pretenders (6 → 4 at seed 42, 16 → 17 over three seeds), confining the optimizer to the generator-feasible region reduces but does not remove them (6 → 3; 16 → 5), and budget-matched random search produces them too.
Every control still commits to the argmin of a surrogate that underestimates its own error at the geometry it selects. A fourth structure audited out of sample gives 4 of 50, inside its recorded band.
The identical protocol on the release this pipeline superseded, at its own solver settings, gives 95 of 179 (53.1 %).
Such reliability claims therefore apply to a specific release.
**Keywords:** neural surrogate; inverse design; transfer learning; metasurface absorber; offline model-based optimization; reliability audit; pre-specified protocol

---

## 1. Introduction

### 1.1. Problem statement

Learned surrogates are now routinely embedded in optimization loops across the computational sciences. A network is trained to imitate an expensive simulator, and the designer then optimizes against the network because the simulator is too slow to query at every iteration. This setting changes which error matters. The loop commits to the surrogate's error at the single point it selects, not to its average accuracy, and it actively seeks the points where the surrogate is most optimistic. Yet the decision to trust a surrogate in that role is commonly made on a forward metric alone, such as a test error or a transferability score computed before training. The reliability of that decision is rarely audited, and still more rarely under a protocol fixed in advance.

Our testbed is nanophotonic absorber design, where surrogates are standard equipment and
several reviews survey the field [3–7]. The structures are metal–insulator–metal (MIM) absorbers [2]. The surrogates come from the companion study [1], which pre-trains on a cheap transfer-matrix-method (TMM) solver, fine-tunes on a rigorous coupled-wave-analysis (RCWA) one, and introduces a
pilot-set diagnostic that anticipates which structures gain from the transfer: the shape
correlation *r* between the two solvers, read jointly with the operating-band amplitude
error. Transfer between physical scenarios [8], design tasks [9] and physical priors
[10, 11] has precedent in this field, as does negative transfer [12], which is why such a
diagnostic exists at all.

The question this paper asks is the general one: how well does a forward transferability diagnostic predict a surrogate's *inverse* reliability? We treat the question as falsifiable. The protocol, its thresholds and three ordered hypotheses (H1a–c, §1.4) were fixed before the main runs, and three structures spanning the diagnostic's range were run through one codebase at three training seeds each.

### 1.2. Forward → inverse risk asymmetry

Held-out forward evaluation estimates an average error on its test distribution. A small surrogate error contributes additively to the average mean absolute error (MAE). Inverse optimization is
categorically different. The optimizer
$x^\* = \arg\min_x \mathcal{L}\!\left(f_{NN}(x),\, y_{\text{target}}\right)$ *actively
searches* for geometries that minimize the surrogate-reported loss. The optimizer
therefore preferentially steps into any region where the surrogate underestimates the true
target-matching loss. A 3 % forward MAE thus becomes an *exploitation target* in the
inverse direction, rather than an additive noise floor. This is the same pathology that
the offline model-based-optimization literature calls oracle gaming or surrogate
exploitation [13–16]. Optimizing against a fixed learned objective drives the solution
into regions where the objective is over-optimistic. Forward metrics (MAE, R², spectrum
match) therefore do not by themselves establish inverse reliability. An optimizer-aware,
oracle-in-the-loop reliability framework is required.

### 1.3. Related work

Managing an approximate model inside an optimization loop is an old problem. Trust-region
model management [17] and the surrogate-management framework [18] let a cheap model drive
the search. In those methods, evaluations of the expensive function decide whether each
step is accepted and keep the iteration convergent. The surrogate-based and
surrogate-assisted optimization literatures have since developed that model-management
bargain [19, 20]. Multi-fidelity modelling treats the same two-solver structure
explicitly. An autoregressive or composite model relates a cheap low-fidelity source to a
scarce high-fidelity one [21–23]. Sequential multi-fidelity optimisation goes further by
choosing, per query, which fidelity to evaluate [60]. We read the TMM→RCWA transfer
audited here as a multi-fidelity scheme in which the cheap solver enters through
pre-trained weights rather than through an explicit discrepancy term. That reading is
ours, not a result of those works. What these sequential model-management methods have in
common is a step at which the expensive model is consulted about the cheap model's answer.
A forward transferability score lacks that step. The protocol below keeps only that step,
in its cheapest form: one high-fidelity call at the committed point. It gives up the
iterative guarantees that [17, 18] obtain by consulting the expensive model throughout.

The failure mode we measure is the one offline model-based optimization has named.
Optimizing against a fixed learned objective drives the solution into regions where that
objective is over-optimistic. The remedies proposed respond to it in different ways.
Conditioning by adaptive sampling [24] and weighted retraining [25] adapt the proposal or
generative distribution toward designs the data support. Model inversion networks [13]
learn the map from objective value to design directly. Autofocused oracles [16] and robust
model adaptation [26] re-weight or adapt the predictive model around the designs the
search proposes. Design-Bench provides a standardised benchmark suite for the setting
[15], and a recent review surveys the field [27]. The same structure appears wherever a
proxy is optimized: regressional and extremal Goodhart effects [28], reward hacking [29].
Best-of-*n* selection against a learned reward model is the sharpest analogy to our
commitment rules. Gao et al. [30] show that selecting the best of *n* samples by a proxy
can raise the proxy score while the gold score falls. Choosing the argmin of an imperfect
surrogate over eight restarts, or over 6400 random draws, is the same operation. This is
why budget-matched random search produces pretenders as gradient descent does
(Section 3.6). In silico sequence-design benchmarking has independently reported the
following about learned oracles used to score generated candidates [31]. Such oracles
disagree with one another by architecture and even by training seed. They also generalise
poorly to the out-of-distribution sequences that design methods propose.

The companion study itself recorded a preliminary inverse-design probe on its earlier
visible-band Structure-A surrogate ([1], Supplementary S13). Gradient optimization
exploited the surrogate, and the committed designs failed under two RCWA solvers. That
qualitative observation is the antecedent of this audit. The audit turns it into a
pre-specified, multi-structure, multi-seed accounting on the published pipeline.

Two responses to an unreliable surrogate must be distinguished. (i) *Surrogate-improving*
calls spend the simulation budget on the model. Active learning selects the expensive
simulations that most improve the surrogate [32]. Physics-driven training folds the solver
into the training objective of a design-generating network [33]. (ii) *Certification-only*
calls spend the budget on checking the committed answer and change the model not at all.
This is the protocol of this paper. It differs from the iterative model management of
[17, 18] in consulting the expensive model once, at the end. In our judgement the first is
the better long-run investment. The second is what a practitioner who has already trained
a surrogate can do today. Its cost is one full-spectrum solve per design, by construction
of the protocol (103–4595 s here, Section 3.5). This paper measures how much that single
call buys.

Finally, there is the reliability of the claim itself. Pre-registration separates
confirmatory from exploratory analysis [34]. It has recently been argued for predictive
modelling specifically [35]. Reproducibility programmes ask for the code and artifacts
that let a reader re-derive a number [36]. Overoptimistic claims in machine-learning-based
science have been traced to leakage across many fields [37]. For machine learning applied
to fluid-related partial differential equations, they have been traced to weak baselines
and reporting biases [38]. We follow that line by freezing the protocol, its thresholds
and its hypotheses before the main runs, labelling every post-hoc analysis as such, and
releasing the artifact behind every number.

### 1.4. Hypotheses and pre-specification

The protocol was frozen on 2026-06-10, before the seed-42 runs. It is reproduced in
Supplementary S4 as preserved. The two items added after the freeze are marked there. It
states the research question, the metric hierarchy, the taxonomy, every threshold, the
target count, the seeds, and the statistical analysis plan. Three pre-specified orderings
test whether downstream reliability degrades as *r* falls, across the three structures A
(median *r* = +0.72), C (+0.34), B (−0.07). Those are the *preliminary* pilot values
available when the protocol was frozen (2026-06-10). They are superseded by the published
Table 5 of [1], which gives B +0.96, A +0.83, C +0.65 (Section 2.1, W1). The protocol
keyed the orderings to the shape correlation alone. The printed study reads *r* jointly
with the operating-band MAE, so we report that ordering as well
(Section 3.3). All three conventions are analysed:

- **H1a** — oracle-confirmed success rate is non-increasing as *r* falls (A ≥ C ≥ B).
- **H1b** — Δ-flag rate is non-decreasing as *r* falls (A ≤ C ≤ B).
- **H1c** — ρ(RCWA MAE, Surr MAE), the informativeness of the surrogate's
  self-report about true quality, is non-increasing as *r* falls.

Any outcome is reportable. Confirmation makes the audit a direct sequel to the diagnostic
of [1]. If the hypotheses are not supported, the result would show that the forward diagnostic alone does not provide a demonstrated inverse guarantee; the pre-specified audit protocol would remain the methodological contribution.

### 1.5. Contributions

1. **A pre-specified reliability protocol** (Sections 2.3–2.5). It fixes a pretender
   definition, a reference-solver gap audit, a fake-optimum taxonomy with every test
   measured, calibrated random nulls and a full analysis plan. It was frozen beforehand
   and can be reused on any surrogate-in-the-loop design pipeline.
2. **A non-certification result for the threshold check** (Sections 3.2–3.4). Every valid
   committed geometry passes the surrogate's own 5 % check, and the solver rejects 16 of
   179. On these designs the pass/fail verdict therefore screened out no solver failure.
   The *continuous* self-report can still discriminate on some structures (Section 3.7).
   How many designs the solver rejects depends on the tolerance demanded: 8.9 % at
   τ = 5 %, 31 % at τ = 2.5 %.
3. **Three controls that rule out the obvious candidate causes as necessary** (Sections 3.6, 3.8 and Table 9). Pretenders persist without transferred weights, without the gradient
   loop and without geometric infeasibility. Each control still commits to the argmin of a surrogate that underestimates its own error at the geometry it selects.
4. **A release-dependence result** (Section 3.10). The identical protocol on the release this pipeline superseded, at that release's own solver settings, gives 53.1 % against 8.9 %.
   A reliability claim of this kind therefore characterizes a specific release rather than a structure or a method, which is an argument for auditing the pipeline that is actually shipped.
5. **An out-of-sample structure** (Section 3.12). A fourth absorber, generated after the
   protocol was frozen, was audited on its full 50-design held-out split against a rate
   band recorded before its runs. It gives 4 of 50 (8.0 %), inside the band, and every one
   of its pretenders is again a geometry committed outside the generator's feasible region.

Where the diagnostic and the cheap detectors do carry signal, it is reported with its limits. Pretender incidence follows the pre-specified *r*
ordering across these structures (Section 3.3), and two of the oracle-free detectors reach an area under the receiver operating characteristic curve (AUROC) near 0.8 or above on two of them (Section 3.7). Both rest on event counts too
small to establish a rule.


Two qualifications travel with every number above. The verdicts are conditional on the
pinned reference solver, whose own raster and truncation limits are measured in Section 4.7
(W8). And the Structure-D prediction record, written before its runs, was amended in its
count conversion and falsification bound (the 5–15 % band itself unchanged) while partial
oracle results existed on disk, uninspected (Section 3.12).

The operational form is the checklist of Box 1 (Section 4.4).

---

## 2. Methods

### 2.1. Surrogates and the three structures

All three surrogates share the M_TL+phys architecture of [1]. That architecture is a
ResNet-256-4 backbone with a Sigmoid head. Its input is a normalized wavelength
concatenated with normalized geometry parameters and structure-specific physics
features. Its output is absorptance. The three structures span the *r* range and differ in dimensionality and output channels. Throughout the paper, tables and figures list them in the order of the published *r* of [1], B (+0.96), A (+0.83), C (+0.65), because the pre-specified hypotheses concern monotonicity in *r* (Section 1.4); the preliminary *r* in force when the protocol was frozen orders them A, C, B, and Table 7 tests both orderings:

**Table 1. The three absorber structures** (design dimensionality, physics-feature
count, wavelength grid, filtered dataset and surrogate head; columns ordered
by the published *r*; the *r* values, preliminary *r* values and filtered yields are
published measurements of [1] — Table 5, Supplementary S19 and Methods — restated here as
the inputs to this audit). The main text audits the published pipeline of [1]; the
as-submitted release, whose datasets, bands and feature sets differ, is the second arm of
Section 3.10 and is described in Supplementary Section S8.

| | Structure B (*r* = +0.96; prelim. −0.07) | Structure A (*r* = +0.83; prelim. +0.72) | Structure C (*r* = +0.65; prelim. +0.34) |
|---|---|---|---|
| Physics | ring-disk Fano | asym. dual-dielectric dual-cavity | anisotropic rect. (TE/TM) |
| Design params *d* | 8 | 10 | 7 |
| Physics features | 13 | 17 | 18 |
| Wavelength grid | 400–1800 nm × 100 | 400–1800 nm × 100 | 400–1800 nm × 100 |
| Dataset (reliable/500) | struct_B_500_redesign (498) | struct_A_500_redesign (499) | struct_C_500_redesign (487) |
| Output channels | A, R | A, R | A_TE,R_TE,A_TM,R_TM |
| Surrogate head | single | single | dual |

Each surrogate is fine-tuned from a TMM-pretrained M_TL+phys checkpoint, loaded
**strict**. The published pipeline of [1] did not ship its checkpoints at the audited commit. They were
therefore regenerated here with its own pretraining recipe and archived with their
hashes (Table 2). The recipe is `pbtl_{A,B}_redesign.py` and `pbtl_C_v2_redesign.py`,
steps 1–2 at commit `cb486b5`: 5000 TMM spectra drawn with `rng(99)`, 500 epochs,
lr 10⁻³, batch 2048, seed 42. Faithful loading requires the following.
(i) The geometry input is ordered `[wl_norm, params_norm]`, wavelength first.
(ii) `params_norm` uses the **training bounds of the checkpoint**. In this pipeline
these bounds coincide with the dataset sampling box (for C, Wx,Wy ∈ [50,720] nm). The
as-submitted release trained C on a narrower [50,400] box than it sampled (see
Section 4.7). (iii) The physics features are normalized with the **TMM-set
statistics**, recomputed deterministically. (iv) The upstream recipe is followed: mean squared error (MSE) over all output channels, AdamW [39] lr 3 × 10⁻⁴, weight decay 10⁻⁴, cosine
annealing, 1000 epochs, batch 512, best-val checkpoint, seed 42. Data hygiene and
the train/val/test split mirror [1] exactly. The pipeline's `reliable` mask selects
499 / 498 / 487 of the 500 generated samples for A / B / C. A and R are clipped to
[0,1]. A `rng(42)` permutation assigns test = last 50, val = previous 50 and the
fine-tune train set = first 350 of the permuted remainder (A 350/399, B 350/398,
C 350/387). The same pipeline is run at training-subset/training seeds 123 and 777
for the multi-seed extension.

**Data, checkpoint and solver provenance.** The audit is pinned to one version
of [1]. The audited inputs are listed below by SHA-256 prefix. The full digests are
in `INPUTS_MANIFEST.txt` in the repository.

**Table 2. Provenance of the audited inputs** (SHA-256 prefixes of the filtered
dataset and of the TMM-pretrained checkpoint that every run starts from).

| structure | dataset (SHA-256) | TMM-pretrained checkpoint (SHA-256) |
|---|---|---|
| B | `a75c4de5…` | `892aabda…` |
| A | `9dea0fa6…` | `66aa7db1…` |
| C | `57bb2b99…` | `84fe0a59…` |

The three dataset digests equal those listed for `struct_{A,B,C}_500_redesign.npz` in
the companion's own data manifest at the same commit. The audited data are therefore
the files behind its printed tables, not a regeneration. All simulation code is
vendored at upstream commit `cb486b5`, the commit behind the printed tables of [1].
Its solver settings are a 64 × 64 real-space grid with a **per-wavelength adaptive
Fourier order**. The truncation is set to N = 9, 13 or 17 by a fixed rule. For
Structures A and C the rule depends on the smallest metal feature, the patch aspect
ratio and the wavelength. For Structure D it depends on the smallest feature and air
gap alone. For Structure B it depends on the period-to-wavelength ratio.
Supplementary S16 of [1] describes the rule as feature-size and anisotropy based. The
per-structure thresholds are those of the vendored `rcwa_struct_{a,b,c,d}.py`. The
rule is a heuristic, not a per-solve convergence test. The raster's Fourier couplings
are free of index wrap-around only up to N = 15 (Section 4.7, W8), and Section 3.5
reports the orders it actually chose. The vendored solver runs in complex64
arithmetic, with Johnson & Christy tabulated constants for Cr, a Malitson Sellmeier
model for SiO₂ and the measured Siefke (2016) constants for TiO₂. The as-submitted
release (commit `920b1bd`) ran a fixed Fourier order 5 in complex128 with a Devore
Sellmeier model for TiO₂, on datasets with different bands and hygiene. Numbers from
the two pipelines are therefore not interchangeable. Every comparison in Section 3.1
is made against the published tables. The two arms are compared as two independent
runs of the identical protocol in Section 3.10. The design-by-design comparison of
the solver alone is Section 3.11.

### 2.2. Inverse-design pipeline

For each target absorptance spectrum $A^\*(\lambda)$ over 100 wavelength
channels (C: the transverse-electric (TE) and transverse-magnetic (TM) polarizations jointly) we solve (Figure 1)

$$
u^\* = \arg\min_{u \in [0,1]^d}\ \frac{1}{100}\sum_\lambda
       \big(f_{NN}(g(u), p(g(u),\lambda)) - A^\*(\lambda)\big)^2 ,
$$

where the optimization variable $u$ is the **normalized** geometry. It is mapped to
physical units $g(u)$ on the dataset sampling box and **clamped to $[0,1]$ after
every step**. We use Adam [41] on $u$, lr 0.05 × 500 iters then 0.02 × 300 iters.
We run **R = 8 multi-start restarts** per target (one mid-box draw plus seven uniform
draws). All eight runs proceed as one batched tensor. The committed geometry is the
restart with the lowest final surrogate loss. All eight endpoints are archived for
the Tier-1C dispersion diagnostic. Physics features are recomputed differentiably
from the current geometry at every iteration with the §2.1 TMM normalization.

**Targets.** N = 20 per structure, drawn `rng(42).choice(test_split, 20,
replace=False)`. The targets are random, from held-out data, and drawn by the
**identical** mechanism for all three structures. Restart seeds use
`rng(1000 + original-dataset index of the target)`, a fixed pre-data quantity. They
are recorded for exact reproducibility.

### 2.3. Reliability metric hierarchy

**Tier 1A — success rate and pretenders.** At threshold τ = 5 %, a design is
*surrogate-claimed* if Surr MAE ≤ τ and *oracle-confirmed* if RCWA MAE ≤ τ. A
**pretender** is a design with Surr MAE ≤ τ **and** RCWA MAE > τ. This definition
was fixed in the frozen protocol, before any run of this study. Wilson 95 % CIs [42]
are reported on all counts, including zeros.

**Tier 1B — oracle revalidation gap.**
$\Delta_i = \mathrm{MAE}(A_{\mathrm{RCWA}}(x_i^\*), A_i^\*) -
\mathrm{MAE}(f_{NN}(x_i^\*), A_i^\*)$. A design is flagged when
$\Delta_i > k\cdot\mathrm{MAE}_{NN,\text{test}}$ with k = 2. We report max Δ. We
report the amplification RCWA/Surr only with the small-denominator caveat.

**Tier 1C — convergence.** Multi-start dispersion plus the loss-tail relative standard deviation (RSD) of the committed run. Dispersion is measured as the std of the 8 final losses and the max
pairwise endpoint distance in u-space.

> **Figure 1.** Protocol overview. (a) Pipeline: the TMM-pretrained checkpoint
> regenerated with the recipe of [1] is strict-loaded and fine-tuned (n_train 350 per
> structure; training seeds 42/123/777, the latter two being the post-freeze extension of Section 2.7); 20 held-out RCWA targets per structure are inverted by Adam on the
> normalized geometry u with 8 restarts (500 + 300 iterations); the restart with the
> lowest final surrogate loss is committed; the committed geometry is evaluated once
> by the RCWA reference solver at its per-wavelength adaptive order; Tier 1A/1B/1C and the
> T1–T4 taxonomy are accounted. (b) Pre-specified thresholds: τ = 5 %, k = 2
> (Δ-flag threshold = k × forward MAE), T1 factor 3, T2 ε = 0.05, T3 factor 0.5,
> T4 K = 100 perturbations with σ = 1 % of the box width and a 5 % flag on the mean MAE between the perturbed and the committed surrogate spectra.

**Notation.** *Structure* denotes one of the absorber families A, B and C of the
protocol, or the fourth family D of Section 3.12. *Surrogate MAE* is the mean
absolute error of the surrogate's predicted spectrum against the target. The
*reference-solver MAE* is the same quantity computed from the RCWA solver at the same
geometry. Both are taken over the 100 wavelength channels and expressed in percentage
points. A design is *surrogate-claimed* when its surrogate MAE is at or below τ, and
*oracle-confirmed* when its reference-solver MAE is at or below τ. A **pretender** is
surrogate-claimed but not oracle-confirmed. Δ is the reference-solver MAE minus the
surrogate MAE at the committed geometry. A design is *Δ-flagged* when Δ exceeds k
times the surrogate's forward test error. ρ(RCWA MAE, Surr MAE) is the Spearman
correlation of the two errors across the N targets of one run. The *committed
geometry* $g(u^*)$ is the geometry selected by the commitment rule. *Preliminary r*
and *published r* denote the two transferability conventions of Section 2.1.

### 2.4. Fake-optimum taxonomy

Table 3 defines the four fake-optimum types. It also gives the diagnostic by which
each type is measured and the null against which each is compared.

**Table 3. Fake-optimum taxonomy** (the four types and the diagnostic each one is
measured by; nulls are defined in this table and in the protocol, Supplementary Section S4).

| Type | Description | Diagnostic |
|---|---|---|
| **T1** Oracle divergence | RCWA disagrees by more than a factor of three | $\mathrm{MAE}(A_{\mathrm{RCWA}}, A^\*) > 3\cdot\mathrm{MAE}(f_{NN}, A^\*)$ |
| **T2** Box-edge geometry | recovered geometry pressed to the design box | any normalized parameter within 0.05 of 0 or 1; vs *d*-dim uniform null |
| **T3** Mode collapse | distinct targets funnel to one region | NN distance < 0.5 × median pair distance; vs N = 20 uniform null |
| **T4** Surrogate-local smoothness | surrogate spectrum under K = 100 Gaussian perturbations of u*, σ = 1 % box width | mean perturbation MAE > 5 %, where the perturbation MAE is the MAE between the surrogate spectrum at the perturbed geometry and at the committed geometry (measured, not merely defined) |

### 2.5. Pre-specified thresholds and statistical plan

The thresholds are as follows. τ = 5 % is the engineering tolerance, the scale of the
forward MAE. k = 2 is a doubling buffer above the forward noise floor. The T1 factor is 3.
For T2, ε = 0.05 (line-edge fabrication scale). The T3 factor is 0.5. The T4 threshold is
5 %.

The pre-specified analysis plan:

1. **Primary discrimination statistic: ρ(RCWA MAE, Surr MAE).** This is a Spearman
   correlation. Its asymptotic p comes from SciPy [43], and its 95 % CI is the
   Bonett–Wright Fisher-z interval [44]. No exact or permutation inference is claimed.
   This is the clean quantity that H1c tests: whether surrogate confidence tracks the
   reference solver.
2. ρ(Δ, Surr MAE) is reported **only as secondary with a coupling caveat**. Δ contains
   −Surr by construction, so its negative bias is partly tautological. No "confidence
   inversion" language is used unless an effect survives in ρ(RCWA, Surr) itself, and none
   does.
3. Cross-structure count comparisons use **Fisher exact** tests [45]. We state outright
   when differences at N = 20 are not significant.
4. T3 clustering can shrink the effective N. The cluster structure is reported alongside
   every count-based CI.
5. All seeds are fixed and logged: split 42, train-subset 42, targets 42, restarts
   `rng(1000 + original-dataset index of the target)`, training 42, with the multi-seed
   extension at 123 and 777. The restart-seeding sentence and the extension are the two
   post-freeze additions recorded in Section 2.7.

**Test families and what is confirmatory.** The protocol names three confirmatory
comparisons: the H1a–c orderings at seed 42. It does not specify a family-wise
correction. Everything else in Section 3 is exploratory and is reported with nominal
two-sided *p*-values (α = 0.05 throughout). The following corrections are applied as
sensitivity analyses rather than as decision rules. Holm is applied over the nine
per-run Spearman tests and over the nine pairwise Fisher tests of Section 3.3. Among
those nine Fisher tests there are six distinct tables, since the oracle-success and
pretender comparisons coincide when every design is surrogate-claimed; Holm over nine
is the more conservative. Holm is also applied over the two pooled control primaries of
Sections 3.6 and 3.8, and over all ten control tests (six per-structure and two pooled
Wilcoxon, two McNemar). For the 36 detector × structure intervals of Table 10, an
interval excluding 0.5 is read as a selection from that screen, not as a pointwise
confirmation. Where several seeds share targets, the shared-target dependence is
handled by one of three distinct procedures, which are not used interchangeably. The
first is a cluster bootstrap over targets (Table 10; the percentile intervals of
Section 3.4 and its 20→50 extension). The second divides each structure's counts and
denominators by its design effect before the test (the pooled trend of Section
3.3; the effective-N Wilson interval of Section 3.4). The third is a plain nominal Wilson
interval, labelled as such, when neither adjustment is applied.


### 2.6. Pilot-study defects corrected in the final protocol

Two pilot studies preceded the protocol. A full AI-assisted internal audit of their code
found the eight defects listed below. Every one was fixed before the protocol was frozen,
and no result reported in this paper depends on a pilot run. The complete account is given
in Supplementary Section S5. It includes the pilot headlines that were consequently
dropped.

**Table 4. Pilot-study defects and the corrections carried into the final protocol**
(the complete account is Supplementary Section S5).

| # | pilot defect | corrected protocol |
|---|---|---|
| 1 | B/C inverse run on 380–780 nm against 400–1800 nm data | per-structure native grids, pinned in artifacts |
| 2 | Adam on raw nm — optimizer frozen within ±22 nm | normalized variables + multi-start (§2.2) |
| 3 | divergent RCWA rows leaked into train/test/targets — an instance of the leakage failure catalogued across ML-based science [37] | Paper 1 filter (§2.1) |
| 4 | permuted pretrain input columns; phys stats recomputed on RCWA; C-bounds mismatch | faithful strict loading (§2.1) |
| 5 | T4 reported but never computed | implemented and measured (§2.4) |
| 6 | ρ(Δ, Surr) "confidence inversion" headline (coupling artifact) | primary stat = ρ(RCWA, Surr) (§2.5) |
| 7 | A used 6 curated targets vs B's 20 random — the weak-baseline and reporting-bias pattern documented for ML applied to PDE problems [38] | identical N = 20 random targets for all structures |
| 8 | single deterministic init; `manual_seed` no-op | R = 8 multi-start, archived endpoints |

### 2.7. Pre-specification record

The protocol was frozen on **2026-06-10**. That date precedes the seed-42 runs of the
as-submitted arm (Section 3.10) and is three months before the published-pipeline runs
that form the main text. The protocol is reproduced verbatim in Supplementary Section S4.
Its SHA-256 is `2e973ba41a82346f4724ff57aaf941dc79534a6e3d485c64dc6c61dd8c559e05` and its
git blob id is `e6d020fc3b69011fed4fc39834712da09621b0a9`.

We state the timeline in full rather than claiming more than the record supports. The
seed-42 artifacts carry file timestamps of 2026-06-10 14:38–17:59 UTC, consistent with the
stated freeze. The protocol file itself was last modified on 2026-06-11 09:39 UTC. Two
items in the file as it now stands post-date the freeze. The first is the restart-seeding
sentence. It was appended after an AI-assisted code review pointed out that the as-run
seeding was `rng(1000 + original-dataset index of the target)` rather than the wording
first used. The sentence records the as-run definition, which is a fixed, pre-data
quantity, and all results use it. The second is the parenthesis in §5 item 5 that names
training seeds 123 and 777 as a multi-seed extension. It was appended after the seed-42
runs. The seed-123 and seed-777 artifacts carry timestamps of
2026-06-10 23:46 to 2026-06-11 07:51 UTC, earlier than the file's last modification. The
record therefore cannot show that the extension was written down before those runs.
Table 5 accordingly classifies it as an extension rather than as pre-specified. What does
**not** exist is a contemporaneous public timestamp. The protocol was not committed to
version control or deposited anywhere before execution. The surviving protocol is
reproduced in Supplementary S4 as preserved, with its two post-freeze additions marked.
The repository commit that contains it post-dates the runs. Readers should weigh the
pre-specification accordingly. The freeze is documented internally, and the analysis plan
visibly constrains what we report, but it is not third-party verifiable for this paper.

**Amendments made in September 2026, and what was known when.** The audit was designed
against the release of [1] that was current when the protocol was frozen. Four changes
followed. Each is recorded in the release archive with its date:

1. **2026-09-06 — the audited release changed.** The companion's printed pipeline was
   found to differ from the release the protocol was written against. The differences are
   regenerated pools, per-wavelength adaptive Fourier order, complex64, Johnson–Christy
   materials, and for Structure C a corrected normalization box and an 18-feature set. The
   whole protocol was therefore re-run on the printed pipeline, which became the main
   text. The original arm is reported beside it in Section 3.10. Thresholds, hypotheses,
   target rule and metric hierarchy were not touched. The same plan added the two control
   experiments of Sections 3.6 and 3.8, which the frozen protocol does not contain.
2. **2026-09-07 — the control experiments were reduced to the canonical seed.** After the
   printed-pipeline rates were known (6 of 60 pretenders at seed 42), the from-scratch and
   feasibility controls were run at seed 42 only rather than at three seeds. The rationale
   recorded at the time was statistical: at this base rate, three seeds would not give the
   binary comparison power. An AI-assisted review two days later judged that argument too
   strong. The seeds share one target set (Section 3.4), so they add correlated replicates
   rather than independent designs, but replicates do add some precision. The decision was
   kept on the ground that does hold, namely cost. Both the original and the revised
   reasoning are in the release archive. In any case, the decision was a scope decision
   informed by the main-arm rate, and it is disclosed as such.
3. **2026-09-08 — the controls' primary outcome was changed, before any control output
   existed.** The frozen protocol contains no control experiments. The from-scratch and
   feasibility controls were added by the plan of 2026-09-06 (amendment 1), after the
   as-submitted arm's results were known. Their endpoint was the pretender count. At a base rate near 10 %, that comparison reaches 80 % power only for a rise to 23–35 %, depending on the paired conditions' joint distribution.
   The primary outcome therefore became the **paired shift in Δ**, tested by Wilcoxon
   signed-rank with a bootstrap interval on the median. The pretender count was retained
   as a secondary exact-McNemar comparison **and reported whatever it shows**, together
   with the minimum detectable effect. Those four elements are what the dated amendment
   fixed: pairing, the continuous primary, the binary secondary, and MDE reporting. The
   amendment did not specify how incomplete pairs are handled, nor a multiplicity
   correction across the two controls. The analysis script drops a pair whose solver call
   failed on either side. The two control comparisons are reported at their nominal *p*,
   with Holm corrections over the two primaries and over all ten control tests as
   sensitivity analyses (Section 2.5). Both choices were made when the script was written
   and are stated here rather than claimed as pre-specified. What was already known was
   the main arm's rates. What did not yet exist was any control run; the archived analysis
   file of that date records all six control conditions as absent. This is therefore an
   amendment informed by the main-arm results and fixed before the control results. It is
   not a pre-data plan, and not a choice made after seeing the comparison it governs.
4. **2026-09-16 to 09-19 — the two controls were extended to seeds 123 and 777 after
   all.** Amendment 2 recorded the decision to run them at seed 42 only. After the seed-42
   results and their analysis were written, an AI-assisted review recommended the
   extension as the more informative of two optional computations. The ground was the one
   that amendment 2 had already conceded: correlated replicates still add precision. The
   runs used the identical pipeline and the identical analysis script. Each seed was
   analysed exactly as seed 42 was. There is also a pooled block, whose interval resamples
   targets with their three seeds together (Section 3.4). The extension is post hoc. It is
   reported in Sections 3.6 and 3.8 as a replication of the seed-42 analysis, which is
   unchanged and remains the comparison fixed by amendment 3. The pooled signed-rank and
   McNemar *p*-values are labelled nominal because the 179 pairs are 60 targets × 3
   correlated seeds.


Table 5 states, for every analysis in Section 3, whether it was pre-specified in that
protocol or added afterwards.

**Table 5. Pre-specification status of every analysis in Section 3** (pre-specified,
extension, or post hoc exploratory, with the protocol section that
fixes each pre-specified item).

| analysis | status | protocol section |
|---|---|---|
| H1a–c orderings, seed-42 Fisher and Spearman tests (§3.3) | pre-specified | §1, §5 |
| Tier 1A–1C accounting and the T1–T4 taxonomy (§3.2) | pre-specified | §4 |
| Order-7 spot check of the three highest-Δ geometries | pre-specified, superseded (§3.5) | §3.5 |
| Seeds 123 and 777 (§3.4) | extension; timing relative to the runs not established | §5 item 5 |
| Full order-7 revalidation of all seed-42 designs (legacy arm only) | post hoc exploratory | — |
| Pooled counts with Wilson intervals, intraclass correlation (ICC) and design effects (§3.4) | post hoc exploratory | — |
| Re-analysis under the published *r* and the amplitude ordering (§3.3) | post hoc exploratory | — |
| Random-search and single-start commitment rules (§3.6) | post hoc exploratory | — |
| From-scratch (M0) and feasibility-constrained controls, paired Δ primary (§3.6, §3.8) | added 2026-09-06 after the as-submitted arm (not in the frozen protocol); analysis amended 2026-09-08 after the main-arm rates, before any control output (§2.7) | — |
| T2-versus-pretender association, CMH (§3.7) | post hoc exploratory | — |
| Holm correction over the nine tests (§3.3, §3.4) | post hoc exploratory | — |
| τ sweep and the severity bands (§3.4) | post hoc exploratory | — |
| Power and minimum-detectable-effect analysis (§3.3, §3.6) | post hoc exploratory | — |
| Cochran–Armitage trend over the pooled counts and the seed-pooled block-permutation ρ (§3.3, §3.4) | post hoc exploratory | — |
| Control replication at seeds 123 and 777 with target-clustered pooling (§3.6, §3.8) | post hoc (§2.7 amendment 4) | — |
| Two-arm comparison: Newcombe interval, Fisher tests, cluster bootstrap (§3.10) | post hoc exploratory (§2.7 amendment 1) | — |
| Surrogate-side detector benchmark (§3.7) | post hoc exploratory | — |
| Geometric feasibility cross-tabulation (§3.8) | post hoc exploratory | — |
| Lookup null and recoverability (§3.9) | post hoc exploratory | — |
| Structure D out-of-sample band (§3.12) | recorded before the D runs; amended while 42/50 oracle results existed on disk, uninspected | — |
| Fixed-order-5 cross-solver probe (§3.11) | post hoc exploratory | — |
| 20→50 target-set extension (§3.4) | post hoc, unregistered; nested (the 20 protocol targets plus 30 more, same trained surrogates), not an independent sample | — |

---

## 3. Results

Every table and figure is regenerated deterministically from the archived per-run
artifacts (SHA-256 manifest; Supplementary Sections S1, S2 and S6). Re-executing the GPU
training stages reproduces the results statistically rather than bitwise, because CUDA
training is not deterministic. The reference-solver stage, by contrast, is bitwise
reproducible. Re-simulating 12 committed geometries (four per structure) with the cache
bypassed, against the vendored upstream commit, returns the archived absorptance spectra
exactly on both arms. The maximum absolute deviation is 0 in every spectrum and 0
percentage points in the resulting MAE (`repro_check_v8.json` in each arm's archive). The
published-pipeline recheck at adaptive order took 6.2 GPU-hours. Every archived spectrum
also carries a fingerprint of the solver code and settings that produced it. On every
read, the spectrum is re-validated against the live solver identity. The multi-seed runs
are reported in Section 3.4 and the orders the solver selected in Section 3.5.

### 3.1. Forward fine-tune fidelity against the published tables of [1]

The fine-tuned surrogates reproduce the forward performance that [1] prints, confirming
faithful loading. The test absorptance MAE (A for Structures A/B; the mean of A_TE and
A_TM for C) is **B 1.72 %, A 1.76 %, C 2.11 %** at seed 42. The printed ten-seed values
are 1.65 ± 0.04, 1.79 ± 0.04 and 2.05 ± 0.09 % (|z| ≤ 1.7 on every structure;
`pub_gate_table123.json`). The pilot value of 5.17 % arose from unfaithful checkpoint
loading. The as-submitted release audited in Section 3.10 reaches only 3.2–4.8 %, because
of its narrower bands, looser hygiene and the Structure-C bound mismatch (Section 2.1).
These forward MAEs set the per-structure Δ-flag thresholds of k = 2 × MAE:
**3.44 / 3.51 / 4.21 % for B / A / C**.

### 3.2. Per-structure reliability (seed 42)

Every valid committed geometry was revalidated by the RCWA reference solver (torcwa
0.1.4.2 [46], 64 × 64 grid, per-wavelength adaptive Fourier order
N ∈ {9, 13, 17} = data-generating fidelity; Section 3.5). The cross-structure picture,
with structures ordered by the published *r* (B > A > C; Figure 2), is as follows:

**Table 6. Per-structure reliability at training seed 42** (Tier 1A–1C accounting,
taxonomy counts and the primary discrimination statistic; columns ordered by the
published *r* of [1], Table 5; preliminary *r* from [1], Supplementary S19).

| Quantity | B (*r* = +0.96; prelim. −0.07) | A (*r* = +0.83; prelim. +0.72) | C (*r* = +0.65; prelim. +0.34) |
|---|---|---|---|
| n_train | 350 | 350 | 350 |
| Forward test MAE | 1.72 % | 1.76 % | 2.11 % |
| **Surrogate-claimed success** (τ = 5 %) | **20/20** | **20/20** | **20/20** |
| **Oracle-confirmed success** | 17/20 | 19/20 | 18/20 |
| **Pretenders** [Wilson 95 % CI] | 3/20 [5.2, 36.0] | 1/20 [0.9, 23.6] | 2/20 [2.8, 30.1] |
| Δ-flagged (k = 2) | 7/20 | 2/20 | 1/20 |
| Δ-flag threshold | 3.44 % | 3.51 % | 4.21 % |
| Max Δ (pp) | 12.11 | 9.90 | 4.66 |
| Max amplification | 35.45× | 43.62× | 3.25× |
| ρ(RCWA, Surr) — **primary** | +0.12 (p = 0.62) | +0.34 (p = 0.14) | +0.75 (p < 0.001) |
| T1 / T2 / T3 / T4 (of 20) | 17 / 15 / 7 / 0 | 16 / 14 / 0 / 0 | 1 / 9 / 5 / 0 |
| Distinct modes (of 20) | 16 | 20 | 17 |

The two success rows carry the result. **Every one of the 20 committed designs passes the
surrogate's own success check on all three structures, and the reference solver still
rejects 1 to 3 of them per structure.** The threshold test is the part that screens
nothing out here. An optimizer that drives the surrogate's own error to 0.06–4.6 % leaves
every design below τ. The verdict therefore cannot separate the 54 designs the solver
confirms from the 6 it rejects. That is an observed property of this optimizer on these targets and does not generalize by itself. A weaker surrogate or a harder target set could fail its own
check. The result here is a weaker statement than the one the same protocol supports on
the as-submitted release, where the solver confirmed fewer than half (Section 3.10). Here
the solver agrees 85–95 % of the time.


Two rows move in the opposite direction to the pretender counts. B carries **more** Δ-flags than in the as-submitted release (7 of 20
against 6), even though it has a third as many pretenders. The reason is that the flag
threshold is k = 2 times each surrogate's own forward MAE, and that MAE has halved (1.72 % here against 3.19 %, so the threshold fell from 6.37 % to 3.44 %). Section 4.5 returns to this. C's primary discrimination statistic is
ρ = +0.75 (p < 0.001). On that structure, therefore, the surrogate's continuous
self-report does order the reference-solver error, even though its pass/fail verdict does
not.

> **Figure 2.** Self-reported versus reference-solver error at every valid committed geometry, for all nine runs (three structures × training seeds 42, 123 and 777; circles, squares and triangles respectively). Panels are ordered by the published *r* of [1] (B +0.96 > A +0.83 > C +0.65), the order used in every per-structure figure and table. Both axes are the mean absolute
> error (MAE) of the absorptance spectrum against the target over the 100
> wavelength channels, in percentage points: on the abscissa as the surrogate
> reports it at its own committed geometry, on the ordinate as the RCWA reference
> solver measures it at the same geometry. Dashed lines mark the success
> threshold τ = 5 %; the shaded upper-left quadrant is the *pretender* region,
> where the surrogate MAE is at or below τ and the reference-solver MAE is above
> it. Every point lies left of the vertical line — the surrogate certifies every
> design — while 16 of the 179 lie above the horizontal one. The counts printed in
> each panel pool the three seeds.

### 3.3. Pre-specified hypothesis tests

The three pre-specified orderings are evaluated under both *r* conventions in Table 7.
**Of the two conventions, the pre-specified one is the only one under which H1a holds in
direction (Fisher *p* = 0.605, so not confirmed). H1b and H1c hold under neither.** H1a is
the hypothesis that oracle success is non-increasing as *r* falls. It is monotone along
the preliminary order A, C, B at seed 42 (19 / 18 / 17) and over the pooled runs (57/59,
55/60, 51/60). Under the published *r* it is not. The direction is the one the protocol
predicted, but at these counts a pairwise test cannot resolve it. The extreme pair A-vs-B
gives Fisher *p* = 0.605.

**Table 7. Pre-specified orderings under the two *r* conventions** (seed-42
counts, pooled in parentheses; "monotone" = non-increasing for H1a/H1c,
non-decreasing for H1b, along the stated order).

| Hypothesis | Quantity (seed 42; pooled) | Monotone under preliminary *r* (A, C, B)? | Monotone under published *r* (B, A, C)? | Smallest pairwise *p* (nominal; Holm) |
|---|---|---|---|---|
| H1a | oracle success 19 / 18 / 17 (57/59, 55/60, 51/60) | yes | no | 0.605 (1.000) |
| H1b | Δ-flag rate 2 / 1 / 7 (7/59, 1/60, 16/60) | no | no | 0.044 (0.392) |
| H1c | ρ(RCWA, Surr) +0.34 / +0.75 / +0.12 | no | no | — |
| — | worst-case severity max Δ 9.9 / 4.7 / 12.1 pp | no | no | — |

A Cochran–Armitage trend test on the pooled pretender counts, scored by the *r* values
themselves rather than by rank, is **nominally significant under the pre-specified
convention** (*p* = 0.026). It remains so after the design effect of the shared targets is
applied (*p* = 0.030, Section 3.4). That correction is an approximation: it divides each
structure's counts and denominators by its design effect before the trend test, and it is
not a cluster permutation test. Under the published *r* it is not (0.269 nominal, 0.221
corrected), and under the amplitude ordering it is flat (0.981 nominal). Restricted to the
canonical seed the same test gives *p* = 0.292. The trend therefore rests on the pooled
runs, which Section 2.7 classifies as exploratory.

Two things bound how much this supports the diagnostic. First, the trend is carried by
**16 events across three structures**. Three points admit a monotone ordering by chance
with probability 1/6 under any single convention, and we tested three conventions. Second,
the pairwise tests that would localise the effect do not reach significance, nor could
they easily. With 20 designs per structure a pairwise Fisher test has 80 % power only for
a rate difference of 0.44, and the per-run Spearman test only for |ρ| ≥ 0.65
(`stats_supplement_v9.json`, `power`). Pretender incidence is therefore **consistent with** the pre-specified ordering in these three structures; the ordering itself is not established.


**Ordering under the amplitude component.** The printed study reads *r* together with the
operating-band TMM MAE. It states that the forward benefit follows the amplitude ordering
(A 7.9 % < B 8.9 % < C 16.9 %), not *r*. Ranked that way, the pooled pretender counts are
A 2/59, B 9/60, C 5/60. They are not monotone, and the trend test is flat (*p* = 0.981).
On the published pipeline, therefore, it is the *preliminary* shape correlation that
tracks inverse reliability, not the companion's own preferred joint reading. On the
as-submitted release the two conventions swap. Its pooled counts (A 27/60, B 30/60,
C 38/59; Supplementary Section S8) are monotone in the amplitude ordering (trend
*p* = 0.030 nominal, 0.079 design-effect-corrected) and not in the preliminary one
(*p* = 0.62). Which convention tracks reliability is therefore itself release-dependent
(`stats_supplement_v9.json` of each arm).


### 3.4. Multi-seed robustness

Pooling the three seeds (42 / 123 / 777) tests whether the seed-42 result is specific to that seed:

**Table 8. Reliability pooled over the three training seeds** (42 / 123 / 777;
unadjusted Wilson intervals on the pretender rates (the design-effect-adjusted ones are plotted in Figure 3a); *r* column headings as in Table 6, from [1]).

| Quantity | B (*r* = +0.96; prelim. −0.07) | A (*r* = +0.83; prelim. +0.72) | C (*r* = +0.65; prelim. +0.34) | Pooled |
|---|---|---|---|---|
| Surrogate-claimed success | 60/60 | 59/59† | 60/60 | **179/179** |
| Oracle-confirmed success | 51/60 | 57/59 | 55/60 | 163/179 |
| **Pretenders** [Wilson 95 % CI] | 9/60 [8.1, 26.1] | 2/59 [0.9, 11.5] | 5/60 [3.6, 18.1] | **16/179 (8.9 %) [5.6, 14.0]** |
| Δ-flagged (k = 2) | 16/60 | 7/59 | 1/60 | 24/179 |
| ρ(RCWA, Surr) per seed | +0.117 / +0.146 / −0.119 | +0.340 / +0.454 / +0.186 | +0.750 / +0.829 / +0.719 | range −0.119 … +0.829 |

†One A geometry at seed 777 was degenerate and its solver call failed, so A's
denominator is 59 (179 valid of 180). In the as-submitted release the failing geometry
was in C instead.

The pooled rate is **16/179 = 8.9 %** (per-run counts and intervals in Supplementary
Table S1; Figure 3a). Its nominal Wilson 95 % CI is [5.6, 14.0] %. The 20 targets of a
structure are bit-identical across its three seeds, however, so the runs are correlated.
The intraclass correlation (ICC) is 0.79 on C, 0.10 on B and −0.02 on A, for design effects of
2.58 / 1.21 / 0.96 and effective sizes of 23 / 50 / 61. The pooled design effect of 1.33
leaves the information of ~134 designs and widens the interval to [5.2, 15.0] %. A cluster
bootstrap gives [4.0, 14.5] %; its lower endpoint sits on a discreteness boundary, 7
versus 8 of 179, and moves between 3.9 and 4.5 % across bootstrap seeds, while the upper
endpoint is stable. We therefore treat the per-seed counts (B 3/3/3, A 1/0/1, C 2/1/2 of
20; A at seed 777 of 19) as primary. The rate is strongly τ-dependent (Figure 4), and this
is the most consequential number in the section. Tightening the tolerance to τ = 2.5 %
turns **53 of the 171 designs the surrogate still certifies into pretenders (31 %)**.
Loosening it to 7.5 % leaves 6/179, and to 10 % leaves 3/179. Of the τ = 5 % pretenders,
10 are mild (5–7.5 %), 3 moderate and 3 severe (≥ 10 %). The pretender rate is therefore a
property of the tolerance the design task demands, not a fixed property of the surrogate.


The same trained surrogates were also run on 50 targets per structure, as a post-hoc
target-set extension. This extension is not a redefinition of the primary 20-target
protocol above, and it is not an independent replication of it. The targets are the
original 20 plus 30 more drawn by the identical mechanism. The 20-target counts of Table 8
are therefore a subset of the 50-target counts below, not a separate sample. Pooling the
three seeds over 50 × 3 = 150 target-seed slots per structure gives a pooled pretender
rate of **42/447 = 9.4 %**. On A, 147 slots are valid, because three slots hit the same
class of degenerate-geometry solver failure as in the 20-target run; B and C have 150
each. The nominal Wilson 95 % CI is [7.0, 12.5]; the design-effect-adjusted interval is
[6.8, 12.9] at n_eff = 348; a cluster bootstrap over targets (20,000 reps) gives
[6.3, 12.8]. These intervals overlap the 20-target estimate of 8.9 % [5.2, 15.0]. The
per-structure ordering is unchanged
(A 5/147 = 3.4 %, C 14/150 = 9.3 %, B 23/150 = 15.3 %). The per-structure design effects
are similar in direction and magnitude (A 0.94, B 1.17, C 2.38 at 50 targets vs.
0.96 / 1.21 / 2.58 at 20 targets). The τ-sensitivity pattern is repeated on the larger
set. Tightening τ to 2.5 % turns 155 of the 426 still-certified designs into pretenders at
50 targets (36.4 %, cluster bootstrap [30.9, 42.2]). This is the same qualitative collapse
as the 31 % figure at 20 targets. Because the two counts overlap and share the same
trained surrogates, this is descriptive consistency with the original estimate, not an
independent confirmation of it. We therefore continue to treat the 20-target result as the
pre-specified primary estimate. We report the 50-target numbers as an unregistered
sensitivity check. On its own, that check does not change the sign or approximate
magnitude of the primary estimate.


Whether the continuous self-report carries rank information depends on the structure.
ρ(RCWA, Surr) spans −0.119 to +0.829 over the nine runs. Four runs are nominally
significant: all three C runs (+0.750, +0.829, +0.719; p ≤ 0.0004) and A `s123`
(+0.454, p = 0.044, though its Fisher-z interval [−0.01, 0.76] touches zero). **The three
C runs survive the nine-test Holm correction** (adjusted p = 0.0011, 0.00006, 0.0025),
while A `s123` does not (0.27). (Exploratory) seed-pooled block permutation gives
ρ = −0.01 (B, p = 0.94), +0.33 (A, p = 0.080) and +0.76 (C, p < 0.001: no exceedance in
5000 permutations). As a detector of pretenders, the surrogate's own claimed MAE has a
pooled AUROC (Table 10's seed-stratified estimator) of 0.40 on B, 0.27 on A and **0.98 on
C**. On B and A the pretenders are, if anything, the designs the surrogate is *most*
confident about. On C, by contrast, the surrogate's error ranking is nearly perfect, even
though its pass/fail verdict admits every design (Section 3.7).

> **Figure 3.** Transferability versus inverse reliability. A design is
> *surrogate-claimed* when the surrogate's own mean absolute error (MAE) against
> the target is at or below τ = 5 %, and a *pretender* when it is claimed but the
> RCWA reference solver puts the MAE above τ. (a) Pretenders per surrogate-claimed
> design for each run (markers: training seeds 42, 123 and 777, with Wilson 95 %
> intervals) and the three-seed pooled rate (bar) with a design-effect-adjusted
> Wilson interval; dotted line: coin flip. Tick labels give the published *r*
> and, in parentheses, the preliminary *r* in force when the protocol was frozen —
> both are measurements of [1] (Table 5; Supplementary S19); the inverse-reliability
> quantities are from the present study.
> (b) ρ(reference-solver MAE, surrogate MAE) per run with Bonett–Wright 95 %
> intervals; open markers are the four nominally significant runs. (c) Seed-pooled
> within-structure ρ with block-permutation p (targets permuted jointly across
> seeds; exploratory).

> **Figure 4.** Sensitivity to the success threshold (exploratory; all nine runs,
> training seeds 42, 123 and 777 pooled). (a) Pretenders per surrogate-claimed
> design as τ is varied, per structure and pooled, where at each τ a design counts
> as claimed if the surrogate's mean absolute error (MAE) is at or below τ and as a
> pretender if the reference-solver MAE additionally exceeds τ; bands are a
> cluster bootstrap over the 20 targets of each structure and the labels give the
> pooled pretender/claimed counts, whose denominator falls from 179 to 171 as τ
> tightens to 2.5 %. (b) Distribution of the reference-solver MAE among the τ = 5 %
> pretenders, with the post-hoc mild / moderate / severe band edges at 7.5 % and
> 10 %.

### 3.5. Per-wavelength adaptive order of the reference solver

The published pipeline chooses the Fourier truncation per wavelength from N ∈ {9, 13, 17}
by the companion's feature-size rule rather than fixing it. The fidelity of the audit
therefore varies with the geometry it is asked to resolve, not with a number set in
advance. As a result, the audit evaluates every committed geometry at exactly the fidelity
its training labels were generated at. The same vendored routine decides the order. This
is what "reference fidelity" means throughout. Behind the nine runs of Sections 3.2 and
3.4 are 18 000 wavelength solves (180 committed designs × 100 wavelengths, the unresolved
design included). Across those solves the solver used N = 17 on **59 %**, N = 13 on 7 %
and N = 9 on 35 % (`adaptive_orders_v10.json`). The split is strongly per structure.
Structure A runs at N = 17 on 98 % of its wavelengths and C on 75 %, while B is at N = 9
on 83 %. That is also where the cost goes. A committed A geometry takes about 58 min to
validate, against 3 min for B. That cost is the one the companion's convergence study
reports per wavelength at N = 17 (≈ 28 s; [1], Supplementary S16), multiplied by the
100-wavelength grid. Nearly every A geometry, in the dataset as in the committed set, has
a metal feature or gap below the rule's 150-nm threshold. It is therefore solved at N = 17
throughout.


This replaces the fixed-order-7 revalidation the protocol specified. That check existed to
bound the error of a solver pinned at order 5. The companion's convergence study ([1],
Supplementary S16) reports N = 5 absorptance discrepancies of ≈ 2 pp on A, ≈ 3 pp on B and
≈ 5 pp on C (TE), reaching ≈ 6 pp on A and ≈ 15 pp on C at long wavelengths. Re-running at
7 would now *lower* the fidelity of a solver already at 9 to 17. The order-7 results for
the as-submitted arm, where they do apply, are in Supplementary Section S3. What the
adaptive order does **not** give is a convergence proof. The solver reports cases it
cannot converge at its cap of 17, so these are reference-fidelity statements (Section 4.7,
W8). Section 3.11 re-solves the same committed geometries, on the same 64 × 64 raster, at
a fixed order 5. It measures what the order alone contributes.

> **Figure 5.** Structure B, seed 123, absorptance channel. (a) Worst-case pretender
> (target 19 of 20): target spectrum, surrogate prediction at the committed geometry
> g(u*) with self-reported MAE 0.25 %, and the reference solver at the same geometry with
> MAE 15.9 %. The committed ring has **R_in = 253 nm against R_out = 208 nm** — an inner
> radius larger than the outer one, which the generator's sampling rule
> (R_in ≤ R_out − 10 nm) can never produce, so the optimizer has placed its optimum
> outside the generator-feasible region entirely rather than merely at its edge. Its true
> counterpart is an ordinary ring (R_out = 114 nm, R_in = 104 nm, 2R_out/P = 0.34).
> (b) Best oracle-confirmed design of the same run (target 15: self-reported MAE 0.24 %,
> reference-solver MAE 0.96 %), whose self-report is indistinguishable from the
> pretender's. Target spectra are held-out samples of the published Structure-B dataset
> of [1]; the surrogate and reference spectra at the committed geometries were computed
> for this study.

### 3.6. Commitment rules and the from-scratch control (exploratory)

The question is whether the pretender phenomenon is a gradient-exploitation artifact or
a generic consequence of committing to the surrogate's best point. At seed 42 we compare three commitment rules; random search is budget-matched to best-of-eight gradient descent:

**Table 9. Pretenders under three commitment rules, with random search budget-matched to best-of-eight gradient descent**
(seed 42; *r* headings from [1]; pretenders / surrogate-claimed designs, the same quantity Figure 6 plots, with
the oracle-confirmed count in parentheses.
Supplementary Sections S2 and S8.2 give both quantities for both arms.)

| Commitment rule | B (*r* = +0.96; prelim. −0.07) | A (*r* = +0.83; prelim. +0.72) | C (*r* = +0.65; prelim. +0.34) |
|---|---|---|---|
| Budget-matched random search | 6/20 (14 confirmed) | 4/20 (16 confirmed) | 1/16 (15 confirmed) |
| Gradient, best of 8 restarts | 3/20 (17 confirmed) | 1/20 (19 confirmed) | 2/20 (18 confirmed) |
| Gradient, single start | 4/19 (15 confirmed) | 4/19 (15 confirmed) | 4/15 (11 confirmed) |

(pretenders over surrogate-claimed designs, so the denominator is the number of
designs each rule certified; Figure 6; full table with both denominators, mean oracle MAEs, and
Fisher tests in Supplementary Section S2). **Every rule produces pretenders**. The
gradient loop is therefore **not necessary** for the phenomenon, in the same sense that
Section 3.8 uses for transferred weights and geometric infeasibility: removing it does not
remove pretenders. This comparison does not, however, isolate the gradient loop's own
contribution, if any. Budget-matched random search produces more pretenders than
best-of-eight gradient descent on B and A (6/20 against 3/20; 4/20 against 1/20) and
fewer on C (1/16 against 2/20). None of the three comparisons is significant at these
counts (Fisher p = 0.45 / 0.34 / 1.00). Single-start gradient is at or above best-of-eight
everywhere (4/19, 4/19, 4/15). This argues against restart selection pressure being what
*creates* pretenders. As with the other two controls, however, it does not by itself
identify what does. The pub-arm counts no longer support the stronger claim that a weaker
optimizer is *reliably* worse. On C, random search both certified fewer designs (16 of 20)
and had the single lowest pretender count. Pretenders occurred under all three commitment
rules tested. This is consistent with the surrogate-gaming pathology of offline
model-based optimization [13–16], here on a physical inverse-design testbed. As in
Section 3.8, however, this does not establish it as the sole or necessary cause, and this
study cannot rank the commitment rules against each other.


**From-scratch control.** The same architecture was trained from random initialization
rather than from the TMM-pretrained checkpoint (M0, learning rate 10⁻³ from the upstream
from-scratch recipe). It reaches a forward test MAE of 1.89, 2.59 and 1.96 % on B, A and C,
against the transfer-learned 1.72, 1.76 and 2.11 %. It is therefore a surrogate of
comparable accuracy. It also draws the same targets: the split is fixed before the
training subset, so M0 and the transfer-learned run are paired design by design. Over the
60 paired designs the signed-rank test does not reject no-shift (Wilcoxon *p* = 0.176).
Its estimand, the Hodges–Lehmann shift in Δ, is **+0.22 pp** [−0.09, +0.58]. The median
shift is +0.14 pp [−0.07, +0.44] (bootstrap), the mean is +0.06 pp, and 27 designs move
down against 33 up. The pretender counts go **6 → 4** (discordant 4/2, exact McNemar
*p* = 0.688). Per structure, the median shift is B −0.36 pp (*p* = 0.216), A +0.75 pp
(*p* = 0.0017; Hodges–Lehmann +0.92 pp [+0.32, +1.67]) and C −0.03 pp (*p* = 0.622). The
from-scratch A surrogate, whose forward MAE is 47 % higher than the transfer-learned one,
is measurably worse at the committed geometries, whereas B and C are not.

*Replication at seeds 123 and 777 (post hoc, Section 2.7 amendment 4).* The same control
at the other two training seeds gives Hodges–Lehmann shifts of +0.17 pp [−0.15, +0.50]
(Wilcoxon *p* = 0.283; pretenders 4 → 6) and +0.42 pp [+0.15, +0.73] (*p* = 0.0025;
6 → 7). The from-scratch A surrogate is again the less accurate one at every seed (forward
MAE 2.62 and 2.57 % against 1.78 and 1.67 %). B and C are within 0.2 pp of their
transfer-learned counterparts. Over the 179 pairs of all three seeds the median shift is
**+0.20 pp [+0.03, +0.38]**, with the interval clustered by target (a target's three seeds
resampled together; Hodges–Lehmann +0.27 pp, nominal). The pretender counts are
**16 → 17** (discordant 11/12). The per-structure pattern replicates: A +0.82 pp
[+0.55, +1.32] with pretenders 2 → 7, B +0.02 pp [−0.24, +0.61] with 9 → 5, and C +0.01 pp
[−0.10, +0.14] with 5 → 5 (`control_analysis_v10_3seed.json`).

Transferred weights are therefore not necessary for pretenders. A surrogate trained
without them produces pretenders at a rate this design cannot distinguish from the
transfer-learned one (6 against 4 at the canonical seed 42, 16 against 17 over three).
The pooled shift in Δ is small and positive: about a fifth of a percentage point over
three seeds. It is carried entirely by the structure on which the from-scratch surrogate
is less accurate. The interval is what carries that statement. The binary comparison on
its own could not distinguish 6 from 4. At this base rate it would only reach 80 % power
against a control rate of 23–35 %, depending on how the two conditions co-occur. We
therefore report the bound rather than a bare null. Section 2.7 records that this
analysis was fixed after the main-arm rates were known and before any control output
existed.



> **Figure 6.** Pretenders per surrogate-claimed design at training seed 42 under
> three rules for choosing which geometry to commit: random search over 6400
> uniform draws from the design box — *budget-matched*, i.e. given the same number
> of surrogate evaluations as the gradient loop's 8 restarts × 800 iterations —
> gradient descent with the best of 8 restarts, and gradient descent from the
> single mid-box start. Wilson 95 % intervals; k/n labels give pretenders over
> surrogate-claimed designs, so the denominators differ between rules (C random
> search: 1/16). Pretenders occurred under all three commitment rules, without isolating which of the three commitment mechanisms, if any,
> is the larger contributor; no pairwise difference between the rules is significant at
> these counts.

### 3.7. Surrogate-side diagnostics

No cheap, oracle-free diagnostic flags the pretenders on every structure:

- **T4 surrogate-local smoothness** (measured, K = 100): **0/179 flags**. The mean perturbation MAE, the MAE between the surrogate spectrum at each perturbed geometry and at the committed one, averages 0.79–1.12 % per run and never exceeds 2.97 %. The
  pre-specified finite-scale perturbation threshold is therefore not exceeded at any
  committed geometry. This does not rule out roughness at other scales or directions.
- **Multi-start dispersion** (Tier 1C): the spread of the eight restarts' final losses
  is small (standard deviation 5 × 10⁻⁴ to 2.4 × 10⁻³ in MSE units per run, against
  committed losses of order 10⁻⁴–10⁻³). The *endpoint geometric* spread, by contrast, is
  large (1.36–1.88 in u-space). This pattern indicates distinct basins with comparably low
  claimed loss; it does not indicate which basin is real. The loss-tail RSD
  of the committed run (0.002–0.012) indicates stable terminal loss traces. It does not
  establish that the geometry or gradient had stopped changing.
- **T2 box-edge** flags remain abundant (9–18/20 per run). The association with
  pretender status that the as-submitted release showed, however,
  **does not reproduce here**. The Cochran–Mantel–Haenszel (CMH) odds ratio (OR) is 2.18 with
  p = 0.246, and no per-run Fisher test is significant. As a detector it is worse than
  before in the way that matters. Sensitivity 0.75 and specificity 0.37 give a positive predictive value (PPV) of
  0.10 against a base rate of 0.089. Flagging every box-edge design would therefore
  reject nearly two thirds of the good designs to catch three quarters of a much smaller
  bad set. The flag also partly proxies geometric infeasibility (Section 3.8). T3 counts
  at seed 42 are 7 of 20 on B, 5 on C and 0 on A (2 on A at seed 123). The *N* = 20
  uniform null of Table 3, however, expects 4, 5 and 2, with 95 % bands reaching 9, 10
  and 6. No seed-42 run therefore exceeds its null. The one run that does is B at seed
  123 (11 of 20). T2 rates likewise sit at their *d*-dimensional uniform nulls (13, 11 and
  10 of 20 expected on A, B and C).

None of them is usable as a detector on its own across structures. Pooling the three
seeds within each structure gives the following areas under the ROC curve for separating
pretenders from oracle-confirmed designs (0.5 = chance):

**Table 10. Discrimination of the oracle-free diagnostics** (area under the ROC
curve for separating pretenders from oracle-confirmed designs; the pooled value is the
per-run AUROC averaged over the three seeds with weights n_pretender × n_confirmed, so
that seeds are never compared across runs, and the interval is a 2000-draw cluster
bootstrap over targets in which a target's three seeds are resampled together; draws without a pretender are discarded, so the intervals are conditional on that: on A, with two pretenders, 1716–1779 of the 2000 draws were usable and its intervals are exploratory (C 1744–1780, B 1998–2000); 36 cells
are screened, so an interval excluding 0.5 is a selection, not a pointwise confirmation;
0.5 = chance; `detector_bench_v8.json`).

| oracle-free quantity | B AUROC [95 % CI] | A AUROC [95 % CI] | C AUROC [95 % CI] |
|---|---|---|---|
| 3-seed ensemble disagreement | 0.80 [0.63, 0.94] | 0.73 [0.42, 1.00] | 0.95 [0.88, 1.00] |
| held-out ensemble MAE | 0.75 [0.53, 0.91] | 0.51 [0.26, 0.79] | 0.98 [0.89, 1.00] |
| ensemble-mean MAE | 0.73 [0.52, 0.89] | 0.38 [0.16, 0.60] | 0.98 [0.84, 1.00] |
| distance to the fine-tune set | 0.80 [0.63, 0.94] | 0.73 [0.50, 0.94] | 0.42 [0.16, 0.61] |
| distance to the dataset pool | 0.79 [0.62, 0.92] | 0.73 [0.50, 1.00] | 0.49 [0.30, 0.78] |
| TMM–surrogate disagreement | 0.42 [0.23, 0.63] | 0.62 [0.39, 0.84] | 0.23 [0.00, 0.46] |
| TMM–target disagreement | 0.43 [0.24, 0.63] | 0.62 [0.38, 0.84] | 0.22 [0.00, 0.44] |
| T4 local-smoothness value | 0.48 [0.20, 0.77] | 0.92 [0.79, 1.00] | 0.66 [0.44, 0.87] |
| T2 box-edge flag | 0.62 [0.49, 0.75] | 0.66 [0.57, 0.77] | 0.46 [0.35, 0.60] |
| multi-start final-loss spread (std over the 8 restarts) | 0.50 [0.29, 0.68] | 0.27 [0.00, 0.63] | 0.74 [0.54, 0.92] |
| the surrogate's own claimed MAE | 0.40 [0.19, 0.67] | 0.27 [0.09, 0.49] | 0.98 [0.89, 1.00] |
| generator-infeasibility flag | 0.76 [0.68, 0.86] | 0.74 [0.63, 0.87] | 0.37 [0.28, 0.46] |

The picture is structure-specific rather than uniformly null. On **C**, every
ensemble-based quantity and the surrogate's own claimed MAE reach 0.95–0.98, with
intervals excluding 0.5. The C surrogate thus knows, in rank order, which of its designs
are wrong. In every C run the pretenders sit among the four designs with the largest
claimed MAE, so re-simulating the worst-ranked fifth of each run would catch all of them.
On **B**, the distance to the nearest dataset point (0.79–0.80), the ensemble quantities
(0.73–0.80) and the infeasibility flag (0.76) exclude 0.5, but none exceeds 0.81. The
claimed MAE on B is *below* chance (0.40). On **A**, with only two pretenders in 59
designs, only the T4 perturbation value (0.92), the two binary flags (0.66, 0.74) and the
claimed MAE have intervals excluding 0.5. The claimed MAE on A is again below chance
(0.27). No single quantity separates the two classes on all three structures. The one that
works best on C, the surrogate's own error, is the one that ranks pretenders as the *most*
trustworthy designs on A and B.


Reference-solver evaluation determines the audit's labels by definition, and no tested
diagnostic perfectly separated pretenders from confirmed designs on every structure. The
evaluation is one oracle call per design (103–4595 s at the adaptive order, median
1655 s). It applies the decision rule of Section 2.3 (surrogate-claimed, Surr MAE ≤ τ,
**and** reference-solver MAE > τ). Δ, the gap between the two MAEs, is not that decision
rule. It is reported separately, as severity (Section 2.3, Tier 1B).

### 3.8. Geometric feasibility of the committed designs (post hoc)

The datasets are generated under structure-specific fabrication constraints ([1],
Supplementary S17, which orders the radii of B) as implemented by the generator. For A
and C the constraint is a patch width of at most 0.9 P. For B it is an outer ring radius
of at most 0.45 P, with each inner radius at least 10 nm smaller than the one enclosing
it. The 10-nm margin is the generator's, not stated in [1]. The optimizer, however, searches
the raw sampling box, which does not enforce these constraints. It does leave the feasible
region: over the three seeds, 33/60 committed B geometries, 33/59 A and 13/60 C violate a
constraint that every training sample satisfies. This is not a surprise: a uniform draw
from the box is infeasible with probability 0.77 on B, 0.56 on A and 0.52 on C. On B,
however, it matters, because there every pretender is infeasible. Among the 33 infeasible
B designs, 9 are pretenders, against 0 of the 27 feasible ones (Fisher p = 0.0029). That
p is nominal, the three runs sharing one target set (run-stratified CMH p = 0.0036). On A
the two pretenders are also infeasible, but the count is too small to test (2/33 against
0/26, p = 0.50). On C the association runs the other way (0/13 against 5/47, p = 0.58):
the C pretenders are generator-feasible. A feasibility check is free and requires no
solver call. On B it is an oracle-free signal that both carries information (Table 10, AUROC 0.76) and needs no ensemble. This is why Section 3.7 is stated per
structure rather than as a blanket negative. Figure S1 shows where in the design box the
committed geometries sit.

**Constrained rerun.** The seed-42 inverse design was repeated over a nested
parametrization whose image is exactly the generator's feasible region. Every candidate is
then generator-feasible by construction, so the optimizer cannot leave it. The rerun
changes the outcome less than the B result alone would suggest. The distribution of the
change is also asymmetric enough that one number does not describe it. Paired over the 60
designs the two runs share, the signed-rank test rejects no-shift (*p* = 0.0071; 40
designs improve, 20 worsen, sign test *p* = 0.013). The Hodges–Lehmann shift is
**−0.18 pp** [−0.58, −0.00] and the median shift −0.00 pp [−0.21, −0.00] (bootstrap). The
mean is −0.79 pp, because a few designs improve by several points (range −10.7 to
+4.6 pp). Pretenders fall **6 → 3** (discordant 4/1, exact McNemar *p* = 0.375). By
structure, the median shift is B −0.55 pp (*p* = 0.012; Hodges–Lehmann −1.39 pp
[−2.91, −0.16], mean −1.71 pp), A −0.15 pp (*p* = 0.154; mean −0.71 pp) and C −0.00 pp
(*p* = 0.841; mean +0.04 pp). The effect is largest on B, the structure that commits the
most infeasible geometries and whose pretenders are all infeasible. Structure A
contributes the remainder of the pooled mean. Treated as one family with the other nine
control tests, the pooled feasibility *p* becomes 0.063 after Holm correction
(`control_analysis_v10.json`). The two pooled primaries as a family of two give 0.014.

*Replication at seeds 123 and 777 (post hoc, Section 2.7 amendment 4).* The
feasibility-constrained rerun at the other two seeds takes the pretender count 4 → 1 and
6 → 1. The Hodges–Lehmann shifts are −0.02 pp [−0.26, +0.00] (Wilcoxon *p* = 0.298) and
−0.09 pp [−0.51, −0.00] (*p* = 0.019). Over all three seeds the count goes **16 → 5**
(discordant 12/1; exact McNemar *p* = 0.003, nominal). The median shift stays negligible
at −0.00 pp [−0.02, −0.00] (target-clustered), while the mean is −0.60 pp. The structure
pattern is the seed-42 one: B −0.24 pp [−0.89, −0.01] with pretenders 9 → 1, A −0.03 pp
[−0.50, +0.01] with 2 → 0, and C −0.00 pp with 5 → 4 (`control_analysis_v10_3seed.json`).
The five that survive confinement are four C designs and one B design.

Confinement to the generator-feasible region therefore reduces but does not remove the
problem. The median shift is negligible. B, the structure whose pretenders are all
infeasible, improves the most. Pretenders persist when the optimizer cannot leave the
feasible region at all. Together with the from-scratch control of Section 3.6 and the
commitment rules of Table 9, the three controls show that **none** of transferred weights,
the gradient loop or geometric infeasibility is *necessary* for pretenders. Each can be
removed and pretenders remain. They do not show that any of the three contributes nothing;
the feasible rerun takes B from 3 pretenders to 1 and the pooled count from 6 to 3. Nor do
they isolate a single cause. What the three controls share is the one step none of them
removes: committing to the argmin of a surrogate that underestimates its own error at the
geometry it selects.


### 3.9. Realizable targets and recoverability (post hoc)

Two further questions bear on how much the loop is worth. The first is whether the
practitioner could have skipped it. Every target is a held-out dataset spectrum, so a
nearest-neighbour lookup in the fine-tune set is a legitimate baseline. At seed 42 it
already lands within τ for 20/20 targets on B, 19/20 on A and 11/20 on C, with median
errors of 2.5, 3.2 and 4.8 %. Against that, the optimizer's oracle-confirmed successes are
17/20, 19/20 and 18/20 (Table 6). On B the lookup does at least as well as the loop under
the protocol's threshold. On A the two tie. On C, where the surrogate ranks its own designs
correctly, the loop beats the lookup by seven targets. The optimizer's *claimed* errors
(median 0.4, 0.4 and 1.1 %) are far below either. The second question is whether the
surrogate is able to recognise the right answer. Evaluated at the true generating
geometry, it reports a median error of 1.6 % on B, 1.7 % on A and 1.8 % on C. That is
within τ on 20, 20 and 18 of the 20 targets. It is, however, worse than what it claims at
its own committed point on 100 % of targets on all three structures. The committed
geometries are far from the true ones (median ‖u* − u_true‖ = 0.64, 0.78 and 0.31 in box
units). This is expected: these inverse problems admit many solutions by construction. A
different geometry is therefore not itself a failure, and preferring one's own solution to
the generating one is not either. The failure is the 16 cases where the preferred
solution is wrong.


### 3.10. The same protocol on two versions of one pipeline

The audit was also run end to end on the release this pipeline superseded. That release is
the 2026-03 as-submitted version of [1], with its fixed Fourier order 5, complex128
arithmetic, and the data pools and checkpoints of that release. The protocol, thresholds
and target-drawing rule were identical. Nine runs finished in both arms. The counts are
read from `pub_vs_legacy_v10.json`:

**Table 11. The identical protocol on the two versions of the pipeline** (nine runs per
arm; intervals nominal in the sense of Section 3.4).

| | as-submitted release | published pipeline |
|---|---|---|
| forward test MAE | 3.19–4.84 % | 1.58–2.11 % |
| surrogate-claimed (τ = 5 %) | 179/179 | 179/179 |
| oracle-confirmed | 84 | 163 |
| **pretenders** [Wilson 95 %] | **95/179 = 53.1 %** [45.8, 60.2] | **16/179 = 8.9 %** [5.6, 14.0] |
| per structure | B 30/60 · A 27/60 · C 38/59 | B 9/60 · A 2/59 · C 5/60 |

The difference is 44.1 pp (Newcombe 95 % CI [35.2, 52.1], Fisher exact *p* = 2.3 × 10⁻²⁰).
It holds in every structure (Fisher *p* = 7.4 × 10⁻⁵, 5.8 × 10⁻⁸, 6.9 × 10⁻¹¹ for
B, A, C). Resampling targets within each arm and structure, with a target's three seeds
kept together, leaves the interval at [33.0, 54.8] pp. That resampling respects the
clustering of Section 3.4.

**Interpretation of the pipeline comparison.** The two arms differ in their checkpoints, their
training pools, structure A's wavelength band, structure C's feature set and bounds, and
the reference solver's truncation order, precision and material model. All of these
differences are present at once. Their target sets also differ, because the pools do. This
is therefore a comparison of two complete pipelines, not an ablation. It does not
attribute the change to any one of those factors, and the intervals above are nominal in
the sense of Section 3.4. The comparison does establish that **a reliability statement of this kind is a property of a specific release, not of a structure or of a method**. The same
protocol, the same three geometries and the same thresholds move the pretender rate by a
factor of six between two versions of one pipeline. Section 3.11 tests truncation-order sensitivity on the published-pipeline committed geometries.




---

### 3.11. Fixed order 5 on the same committed geometries

The two-arm contrast of Section 3.10 bundles every difference between the releases: data,
hygiene, checkpoints, bounds and the reference solver's truncation. It therefore cannot
say how much of the sixfold gap is the solver. This probe isolates that one factor. The 60
seed-42 geometries the published-pipeline surrogates committed were re-solved with the
**same** module, precision, materials and grid. The Fourier order was instead fixed at 5,
the as-submitted release's setting, rather than chosen adaptively from {9, 13, 17}. Each
design is therefore compared with itself on the same raster, and only the order changes
(`cross_solver_v11.json`).

**Result.** At order 5 the reference MAE is higher on 46 of the 60 designs and lower on 14
(median shift +0.61 pp, median |shift| 0.90 pp, Wilcoxon *p* = 1.4 × 10⁻⁴). Design-level
excursions reach up to 4.2, 4.4 and 6.1 pp on B, A and C. Those excursions sit where they
matter. The pretender count on the identical designs goes **6 → 13 of 60** (B 3 → 6,
A 1 → 4, C 2 → 3). Every flip is in the same direction: seven designs the adaptive solver
confirms are rejected at order 5, and none flip the other way (exact McNemar *p* = 0.016).
The two solvers agree on the ranking only moderately (Pearson *r* = 0.74, 0.73, 0.56). The
order-5 solve is 10–120 times faster (16–29 s per design against 156–3526 s).

**Effect of truncation order.** On the same surrogates and the same
committed geometries, those of the published pipeline, truncating the reference solver to
order 5 alone more than doubles the pretender count from 6 to 13 of 60. That is a clean, controlled
result. A reference solver truncated at order 5 rejects designs that the same solver at
its adaptive order accepts. It does **not** decompose the as-submitted arm's own seed-42
count of 31 of 60 (Supplementary Section S8). That arm's committed geometries are
different designs, generated by different checkpoints and data pools, so this probe was
never run on them. How much of its 31 owes to truncation, versus to the narrower bands,
looser hygiene, the Structure-C bound mismatch or the surrogates trained on them, is
not established by this or any other analysis in this paper. For Section 3.10 the probe does show that the published pipeline's own reported rate is sensitive to the audit's choice of reference fidelity. Order 5 would have reported 13
rather than 6 on identical designs. The 8.9 % is therefore not immune to the same kind of
truncation dependence, whatever the as-submitted arm's own truncation dependence turns out
to be. Which verdict is physically right is not established by either. The adaptive solver
is the reference fidelity of Section 3.5, not a convergence proof. What the higher order
does not change is the rest: 6 of these 60 designs fail at the reference fidelity, and no
truncation setting makes the surrogate's self-check informative.


### 3.12. Out-of-sample test on a fourth structure

The three structures of the protocol were chosen because the pilot *r* ordered them. A
fourth structure that did not exist when the protocol was frozen is a cleaner test of
whether the diagnostic says anything about inverse reliability, because nothing about the
audit was tuned on it. Structure D is a cross-shaped Cr patch (four-fold symmetric and
polarization-independent at normal incidence; six parameters P, L, w, t_Cr, d_SiO₂, θ; 11
physics features). The present audit uses its TE response at oblique incidence. It was
generated on 2026-07-08 as a contingency structure for the companion's re-review. The
generation used the same published-pipeline settings as the main arm (400–1800 nm,
adaptive order, complex64, Johnson–Christy; 494 of 500 samples reliable; dataset
`beb3c107…`, regenerated TMM checkpoint `4ab5babf…`). Its pilot diagnostic, computed with
the companion's printed estimator, is *r* = +0.905 with an operating-band TMM MAE of
10.51 %. That places it between B and A on *r* and between B and C on amplitude. The
fine-tuned surrogate reaches a forward test MAE of 1.30 %, the lowest of the four. The
audit ran on the full 50-design held-out split at seed 42.

**The recorded prediction.** Before any D inverse design or oracle call, we wrote down the
following prediction (2026-09-10, `structure_D_prespecification.md` in the repository). If
the diagnostic carries information, D's pretender rate should fall between its
neighbours', **5–15 %**. A rate below A's 3.4 % or above B's 15.0 % would falsify that
reading. The band was written on 2026-09-10 as 3 to 8 of 50, with the upper falsifying
bound labelled as C's 16.9 %. It was corrected on 2026-09-12 to 3 to 7 of 50 and to B's
15.0 %. The same file records that its arithmetic and the mislabelled bound were corrected
on 2026-09-12 while 42 of the 50 oracle results already existed on disk and were not
inspected. This is therefore an amendment made with partial results in existence, not a
blinded pre-specification, and it should be read as such.

**Result.** All 50 committed designs pass the surrogate's own check. The reference solver
rejects **4 of 50 (8.0 %, Wilson [3.2, 18.8])**. That is inside the recorded band and not
falsifying. Two of the four are mild (5.3 and 6.4 %) and two severe (12.3 and 15.2 %). The
worst commits an arm width w = 278 nm larger than the arm length L = 181 nm, a cross that
the generator cannot produce. That is the pattern of Section 3.8 again. In D, 20 of the 50
committed geometries violate the generator's rule (L ≤ 0.9 P and w ≤ L). **All four
pretenders are among them, and none of the 30 feasible designs is a pretender** (Fisher
*p* = 0.021, post hoc). The Δ-flag fires on 7 of 50 at a threshold of 2.61 %
(k = 2 × 1.30 %). The surrogate's continuous self-report is informative here (ρ = +0.51,
*p* = 0.0002, between A's and C's values). T4's finite-scale perturbation threshold is not
exceeded at any design (0 flags; mean perturbation MAE 0.92 %, maximum 2.27 %). The solver
ran at N = 17 on 45 of the 50 designs, a median of 1505 s per design.

**Interpretation of the out-of-sample result.** Placed on the four-point ordering by the printed *r*
(B > D > A > C), the rates are 15.0, 8.0, 3.4 and 8.3 %. That sequence is not monotone
(Cochran–Armitage *p* = 0.32). Ordered by operating-band amplitude (A > B > D > C), they
are 3.4, 15.0, 8.0 and 8.3 %, likewise not monotone (*p* = 0.97). D therefore lands where
the band said it would. However, the non-falsification range spans A's rate to B's, and a single structure at 50 designs has an interval wide enough to contain all three
neighbours. The out-of-sample test confirms the two findings that do not depend on *r* at
all: the threshold self-check admits every design, and the pretenders are the geometries
committed outside the generator's domain. It leaves the *r* ordering where Section 3.3
left it: consistent with, not established.


## 4. Discussion

### 4.1. Forward scores and downstream use

The transferability diagnostic of [1] is the shape correlation *r* read jointly with the
operating-band TMM MAE. It answers a forward question: whether TMM-pretraining helps the
surrogate interpolate RCWA. The audit answers a
different question: whether the optimizer can be trusted to commit to the surrogate's
argmin. The two come apart because the inverse loop commits to the surrogate's
**pointwise** error at the geometry it selects, not to its average error. That geometry is
the argmin, whether reached by gradients or by random search (Section 3.6). A surrogate
can have excellent forward MAE, 1.7 to 2.1 % here, and still possess shallow basins that
the reference solver rejects. The optimizer is built to find such basins. An average test
error need not reveal those basins, and *r* cannot see them, because *r* is a correlation
of *spectra* rather than of error landscapes.

The conclusion supported by the data is narrower than a blanket separation between forward and inverse reliability, and it should be stated precisely. The **threshold verdict** screened nothing out. All 179 committed designs pass
the surrogate's own 5 % check, and 16 do not survive the solver. A pass/fail self-report
therefore cannot rank anything, on any structure, at any *r*. The **continuous**
self-report is a different object. On Structure C it correlates with reference-solver
error at ρ = +0.72 to +0.83 across seeds (Sections 3.4 and 3.7). That correlation is
exactly the information the thresholded version discards. The diagnostic itself is not
silent on inverse reliability: pretender incidence follows the pre-specified ordering
(Section 3.3). However, it orders incidence across three structures at 16 events, which is
a direction, not a calibration.

These results indicate that **neither the forward score nor the thresholded self-report can substitute for reference-solver evaluation of the committed geometry.** The audit requires one solver call per design and reduces the set of 179 surrogate-certified designs to the 163 that meet the reference-solver tolerance.


### 4.2. Incidence, severity and the *r* ordering at this event count

The only quantity that orders with the preliminary *r* is incidence. This is seen in the
oracle success of 19 / 18 / 17 along A, C, B at seed 42 and in the pooled trend of
Section 3.3. Worst-case severity does not order with the preliminary *r*. The largest gap
is on B (12.1 pp), then A (9.9 pp), then C (4.7 pp), which is monotone under neither
convention. Nor does the Δ-flag count (2 / 1 / 7). B carries the most flags because its
threshold is the lowest. The contrast survives a common threshold: A 2, B 7, C 1 at either
A's 3.51 % or B's 3.44 % threshold, with B-vs-C Fisher p = 0.044 nominal and 0.39 after
Holm. At C's 4.21 % threshold the counts are A 1, B 4, C 1, with p = 0.34
(`stats_supplement_v9.json`). The excess on B is therefore a nominal excess of moderate
gaps rather than an artefact of the scaling alone. A practitioner thus cannot use *r* to
bound the worst case. The practitioner can use *r* to order incidence, but only in the
weak sense that Section 3.3 establishes at 16 events.


### 4.3. The pretender phenomenon and offline model-based optimization

Random search produces pretenders as readily as gradient descent (Section 3.6). That
result places this work squarely beside the offline model-based optimization (MBO) and surrogate-gaming literature [13–16]. Optimizing against a fixed learned objective, however obtained, drifts into
regions where the objective is over-optimistic. It is the same structure that Goodhart's
law [28], reward hacking [29] and reward-model overoptimization [30] describe in other
settings. Autofocused oracles [16] and conservative objective models [14] address it by
re-training or regularizing the surrogate. Design-Bench [15] standardises offline-MBO
tasks and their evaluation as a benchmark suite. Robust model adaptation [26] instead
localises the surrogate around the query point and penalises the local sensitivity of its
prediction. Our T4 test measures a related finite-scale spectral sensitivity (K = 100
Gaussian perturbations at σ = 1 % of the box width), not RoMA's robustness objective. It
finds this sensitivity small at every pretender: 0/179 designs flagged, mean perturbation
MAE 0.79–1.12 % per run, global maximum 2.97 %. The pretenders therefore sit in *smooth*
regions of the surrogate at this perturbation scale. That does not show that training with
a smoothness regularizer would leave the same optima in place. Our contribution is
accordingly orthogonal and diagnostic. It is a pre-specified accounting of how often the
un-regularized, physics-transferred surrogate that practitioners actually ship is wrong.
It is also a demonstration that, within this audit, no tested diagnostic perfectly
separated pretenders from confirmed designs on every structure, so the reference solver's
own call remains the deciding step.

### 4.4. Partial signal in the cheap detectors and the solver call as safeguard


The result in Section 3.7 is practically important. The diagnostics a practitioner would
reach for first are local smoothness, multi-start agreement and box-edge checks. These
either fail outright or, like the box-edge flag, carry at most a weak and unusable
association (CMH OR 2.18, p = 0.25, specificity 0.37). The reason is that a pretender is
precisely a point where the surrogate is *internally* self-consistent and confident yet
*externally* wrong. Internal consistency alone does not certify external accuracy. On
Structure C, however, the surrogate's own continuous error and the ensemble quantities do
rank the pretenders (Section 3.7). Even there, the pass/fail verdict admits every one of
them. Two uses of a simulation budget must then be separated (Section 1.3).
(i) *Surrogate-improving* calls retrain or adapt the model where the search is heading, by
active acquisition [32] or by solver gradients that train a generator [33].
(ii) *Certification-only* calls leave the model untouched and merely test the committed
answer. This paper's protocol is of the second kind. It differs from the iterative model
management of [17, 18] in consulting the expensive model once, at the end. This audit
quantifies the second. The safeguard this audit supports is the direct one: **put the
oracle in the loop at the committed geometry.**
One reference-solver call per design converts a silent-failure rate of 8.9 % on this pipeline, and of 53.1 % on the as-submitted one, into a detected-failure rate.
The detected failure is Surr MAE ≤ τ **and** reference-solver MAE > τ, the pretender rule
of Section 2.3. It is not the Δ gap, which is reported separately as severity.

**Uncertainty quantification is the obvious missing ingredient, and we tested a piece of
it.** The standard toolkit addresses a model asked about inputs unlike its training data.
That toolkit comprises deep ensembles [47] and MC dropout [48] for predictive uncertainty,
post-hoc calibration of classifier confidence [49], and the benchmark evidence that
uncertainty quality often deteriorates under dataset shift and that in-distribution
calibration does not ensure it [50]. The toolkit is surveyed in [51]. The committed
geometries here tend to be such inputs: the optimizer searches for them, and
Sections 3.8 and 3.12 find a fifth to over half of them outside the generator's domain.
Our detector benchmark (Section 3.7) includes the cheapest usable member of that family:
the disagreement of a three-member deep ensemble formed from the three training seeds. It
also includes a held-out-ensemble error and a *k*-NN distance to the training set. Their
AUROCs are reported there. The ensemble quantities reach 0.95–0.98 on C and 0.73–0.80 on
B. The AUROC of the training-set distance is 0.80 on B but 0.42 on C. On A, none of them
is distinguishable from chance. What we have *not* tested is a calibrated Bayesian
surrogate trained for the purpose, or a conformal wrapper for optimizer-induced shift with
the feedback-covariate-shift correction of [59]. The coverage guarantee of such a wrapper
would itself have to be established for designs selected adaptively by the optimizer. This
untested approach is the natural next experiment and the main qualification on the
negative result (W9).

> **Box 1. Oracle-in-the-loop reliability audit for a learned surrogate in inverse design**
>
> 1. **Freeze the accounting first.** Write down the threshold τ, the gap factor k, the
>    restart count R, the target count N and the seeds before the main runs. Here
>    τ = 5 %, k = 2, R = 8, N = 20 and the canonical seed is 42; seeds 123 and 777
>    were extensions whose recording time relative to their runs is not established
>    (Section 2.7).
> 2. **Draw the targets from held-out data by one mechanism for every structure.**
>    *(2b: enforce the generator's geometric feasibility constraints where they are known.
>    Section 3.8 shows this reduces but does not remove pretenders, and the unconstrained arm
>    quantifies the cost of omitting them.)*
> 3. **Run R restarts and commit the lowest final surrogate loss.** This is the rule a
>    practitioner would actually use.
> 4. **Call the reference solver once, at the committed geometry.** Once per design, not
>    per iteration.
> 5. **Report the τ-anchored pretender count with a Wilson interval as the primary metric,** and the gap Δ as severity; when repeated seeds share targets, use a target-clustered interval instead. Do not scale the primary threshold by each
>    surrogate's own forward error.
> 6. **Treat surrogate-side diagnostics as triage, not as a substitute.**
>
>
> Cost: one reference-solver call per committed geometry.

### 4.5. Protocol refinement: τ-anchored accounting over MAE-scaled flags

The k × (forward test MAE)-scaled Δ-flag measures something other than the pretender
count. The published pipeline shows the two coming apart in the opposite direction from
the superseded release. There, Structure A had nine pretenders and no flags, because its
9.69 % threshold sat above its own worst Δ. Relative scaling under-detected exactly where
forward MAE was large. Here the forward MAE has halved, and the thresholds have fallen
to 3.4–4.2 %. The flag now **over**-counts. Structure B carries 7 flags against 3
pretenders at seed 42, and A carries 2 flags against 1. The reason is that a Δ of 4 pp
trips a 3.44 % threshold while the design still passes the 5 % success check. A threshold
tied to the surrogate's own accuracy moves with it, so the same design can be flagged in
one release and not the other. The τ-anchored pretender count (Surr ≤ τ < RCWA) is
anchored to the design tolerance instead, and it is the right primary reliability metric.
The Δ-flag is retained as a severity descriptor, not a detector.


### 4.6. Relation to prior work

**Table 12. Relation to prior work** (setting and failure-mode treatment of the
closest studies, with this audit in the final row).

| Reference | Setting | Failure-mode treatment |
|---|---|---|
| Liu et al. 2018 [52] | NN inverse design, single-fidelity | tandem architecture |
| Peurifoy et al. 2018 [53] | NN inverse design, single-fidelity | forward + back-prop inverse |
| Unni et al. 2020 [40] | mixture-density inverse design | multi-modal handling |
| Jiang & Fan 2020 [54] | simulator/adjoint-in-the-loop generative | simulator and adjoint evaluations inside training |
| Cheng et al. 2024 [55] | TL × MDN inverse, same fidelity | curriculum within one solver |
| Ren et al. 2020 [56] | neural-adjoint inverse benchmark | boundary loss keeps the search inside the training domain; top-*k* candidates re-simulated |
| Jiang & Fan 2019 [33] | simulator-in-the-loop generative design | solver inside the training objective |
| Ovadia et al. 2019 [50]; Abdar et al. 2021 [51] | UQ under dataset shift | predictive uncertainty and its limits under dataset shift |
| Setinek et al. 2025 [58] | neural-surrogate benchmark under distribution shift, **forward** prediction (four industrial simulation tasks) | shift fixed by the benchmark's train/test split; unsupervised domain adaptation as the remedy |
| Kumar & Levine 2020 [13] | offline model-based optimization | direct model inversion (objective value to design) |
| Trabucco et al. 2021 [14] | offline model-based optimization | conservative objective models |
| Fannjiang & Listgarten 2020 [16] | offline model-based optimization | autofocused oracles |
| The forward study [1] | cross-fidelity TL, **forward** | joint *r*/operating-band-MAE diagnostic; controlled negative-transfer tests; a preliminary inverse-failure probe (its Supplementary S13) |
| **This audit** | cross-fidelity TL, **inverse**, 3 structures × 3 seeds | pre-specified Δ audit + measured 4-type taxonomy + uniform-random nulls + *r* non-certification + mechanism control |

The surrogate-gaming *mechanism* itself (committing to the argmin of an imperfect learned
objective) is not new. It is well documented in the offline MBO literature [13–16]. Our
novelty is the **pre-specified, multi-structure, multi-seed reliability audit of a
physics-transferred surrogate in inverse design, with a reference solver on every valid
committed geometry**. The novelty also lies in the resulting finding that the forward
transferability diagnostic does not supply, by itself, a demonstrated inverse guarantee.
The closest recent treatment of the same underlying failure, a neural surrogate losing
accuracy outside its training distribution, is the SIMSHIFT benchmark [58]. That benchmark
measures forward-prediction degradation across four industrial simulation tasks under
shifts that the benchmark itself specifies, and it tests unsupervised domain adaptation as
the remedy. This audit asks the downstream version of that question. Here the shift is not
chosen by the experimenter but produced by the optimizer, which commits to the geometry
the surrogate is most optimistic about (Sections 3.8 and 3.12). The quantity at stake is
the design decision rather than the field error. The two are complementary: the pretenders
found here are the natural test cases for the adaptation methods that benchmark evaluates.

Two mappings between that row set (Table 12) and our taxonomy follow. First,
the neural-adjoint *boundary loss* [56] penalises candidates that leave the
training-domain box. Our **T2** box-edge flag marks a committed geometry within 0.05 of a
box face, which is a weaker, interior condition. A design at u = 0.02 trips T2 and incurs
no boundary penalty. T2 is therefore a diagnostic neighbour of that regulariser, not its
equivalent. Second, re-simulating the top-*k* neural-adjoint candidates and our
**Δ audit** share one operation, a reference-solver evaluation of a surrogate-chosen
design, but in different roles. There the solver picks the winner among *k*. Here it
audits the single winner the surrogate already picked, which is what makes the cost
exactly one solver call per design.


### 4.7. Limitations

- **W1.** Three orderings of the same three structures appear in this literature. The
  first is the *preliminary* pilot *r* (A +0.72, C +0.34, B −0.07), which the protocol
  used and which is the pre-specified convention here. The second is the printed *r* of
  [1] (B +0.96, A +0.83, C +0.65). The third is the printed operating-band TMM MAE (A
  7.9 %, B 8.9 %, C 16.9 %), which is the component the companion says the forward benefit
  actually follows. The published values come from the regenerated data-generation
  pipeline audited here, whereas the preliminary values came from the pilot pools of the
  as-submitted release. The conventions are therefore not three readings of one dataset.
  All three are reported (Section 3.3). None is confirmed at the pre-specified seed, and
  only the preliminary ordering reaches nominal significance in the exploratory pooled
  trend.
- **W2.** All three structures sit on the same 400–1800 nm grid in the published pipeline.
  The band mismatch that constrained cross-structure comparison in the as-submitted
  release (A on 380–780 nm against B and C on 400–1800 nm) therefore does not apply here.
  Section 3.10's legacy arm still carries it, and the comparisons there stay dimensionless
  for that reason. The design boxes still reach P/λ_min = 1.5 (A) and 2.0 (B, C), so
  several diffraction orders are open at the short-wavelength end on every structure.
- **W3.** N = 20/structure/seed (D: 50, one seed). Counts have wide CIs, so we use "not
  supported" rather than "refuted," and we report the mode count beside every pretender
  count (Tables 6 and S1). The three seeds of a structure share the same 20 targets.
  Pooled counts (179) are therefore correlated, and their nominal Wilson intervals are
  optimistic (Section 3.4).
- **W4.** Single architecture, single physics family (MIM, Cr). The scope is the release
  audited here, and the paper is positioned as a companion to [1].
- **W5.** The optimizer compresses self-reported MAE into 0.06–4.6 %, which attenuates
  rank correlations. This range restriction is part of the phenomenon, not a nuisance to
  be corrected away.
- **W6.** The as-submitted release had two Structure-C quirks. One was a fine-tune pool of
  n_train = 300 labelled "n = 350". The other was a checkpoint normalization box of Wx,
  Wy ∈ [50, 400] nm against a dataset sampling those parameters to ~720 nm. That box put a
  third of the committed C geometries outside [0, 1]. Both are **absent from the published
  pipeline audited here**: C trains on 350 of 487 reliable samples with the corrected
  bounds and the 18-feature set. They remain properties of the legacy arm of
  Section 3.10 and are one of the several differences that arm bundles together.
- **W7.** The from-scratch control (Section 3.6) bounds, rather than removes, this
  concern. Over the 60 designs it shares with the transfer-learned run at the canonical seed 42, the paired shift in Δ is +0.14 pp [−0.07, +0.44] and the pretender
  counts are 6 against 4. Over the 179 designs of all three seeds, the target-clustered
  shift is +0.20 pp [+0.03, +0.38] and the counts are 16 against 17. A surrogate trained
  without transferred weights therefore produces pretenders at a rate this design cannot
  distinguish from the transfer-learned one. The control is not identical in accuracy. On
  A its forward MAE is 2.57–2.62 % against 1.67–1.78 % at every seed, and its Δ shifts by
  +0.82 pp [+0.55, +1.32] pooled. On B and C the pooled shifts are +0.02 and +0.01 pp. The
  control therefore bounds the pooled contribution of transfer to a few tenths of a
  percentage point of Δ, and A's own to about +0.5 to +1.3 pp, rather than showing either
  to be zero. It does not establish the behaviour of a much *less* accurate surrogate, or of one carrying a different kind of certificate.
- **W8.** The reference solver is the same coupled-wave code that generated the training
  labels, so surrogate and oracle share a systematic error. This is a reference-fidelity
  statement rather than a convergence-certified one. The published pipeline chooses the
  Fourier order per wavelength from {9, 13, 17} instead of fixing it at 5. This reduces
  the discrepancy relative to that fixed setting. The companion's convergence study ([1],
  Supplementary S16) measured that discrepancy at up to ≈ 6 pp on A and ≈ 15 pp on C at
  long wavelengths. The order-7 revalidation that mitigated it in the legacy arm is
  therefore superseded rather than repeated here. The 64 × 64 real-space raster used to
  sample every geometry is a second fidelity setting. Its Fourier couplings are free of
  index wrap-around only up to N = 15, so at N = 17 the outermost couplings of the
  truncated operator (≈ 2 % of its entries) fold. Re-solving 54 committed designs on a
  256 × 256 raster at the pinned order changed no pretender verdict (mean |ΔA|
  0.01–1.20 pp per design, worst single wavelength 2.2 pp, against τ = 5 %). The re-solved
  set comprised every design within 2 pp of τ (42), the three with the largest predicted
  raster sensitivity, and ten drawn at random (one of them also within 2 pp). The set
  spans all four structures and every training seed, and all were re-solved at 100
  wavelengths. This result bounds the sampled designs, which include every design within
  2 pp of τ, not the population. The companion justified this raster for feature
  sizes ≥ 50 nm. Structure B's design space extends to 10 nm, and 471 of the 500 samples
  behind its printed tables (94.2 %) contain a feature below 50 nm, with 309 (61.8 %)
  below two pixels (`paper1_b_feature_census_v11.json`). For that structure the premise
  therefore does not hold. Re-solving 30 of the companion's Structure-B dataset samples,
  stratified by feature size, on the same 256 raster moves their labels by 0.06–0.90 pp
  (mean |ΔA| per sample; worst single wavelength 2.6 pp). The grid-64 control reproduces
  the archived labels to 0.002 pp (median; maximum 0.05 pp;
  `grid_delta_summary_v11.json`). The oracle also applies no admissibility filter:
  2 of 289 archived committed-design spectra carry one point below the companion's −0.005
  floor (the grazing-order artefact of [1], Supplementary S16). Neither verdict changes if
  the point is clipped or dropped (`oracle_admissibility_v11.json`). The solver still
  describes unconverged cases at its order cap, so nothing here certifies physical truth.
  Section 3.11 re-solves the committed geometries, on the same raster, at a fixed order 5.
  It finds that the order alone more than doubles the pretender count on identical,
  published-pipeline designs (6 → 13 of 60). The verdict on *these* designs is sensitive
  to the audit's own truncation, though not, on this evidence, to its raster.
  The probe was not run on the as-submitted arm's own (different) committed geometries, so how much of its 53.1 % owes to truncation versus to its other differences (Section 3.10) is not established.

- **W9.** The surrogate-side diagnostics benchmarked here are the cheap ones a
  practitioner already has: a three-seed ensemble, held-out ensemble error, *k*-NN
  distance to the training set, TMM disagreement and the four taxonomy tests. We did
  **not** train a calibrated Bayesian or conformal surrogate for the purpose. The negative
  result is therefore scoped to the diagnostics tested rather than to uncertainty
  quantification as a whole.

## 5. Conclusion

We pre-specified a reliability accounting protocol and used it to ask how well a forward
transferability diagnostic predicts the inverse reliability of a physics-transferred neural
surrogate. We ran it on the published pipeline of [1]: three MIM structures spanning the
diagnostic's range, three training seeds each, and the reference solver called at every
valid committed geometry.

On the question asked, the diagnostic's predictive power is not established: pretender incidence follows its preliminary ordering across these three structures, but with 16 events the pairwise tests do not reach significance, and the ordering that tracks reliability differs between the two releases audited (Section 3.3). What the audit does establish is a separate observation: the surrogate's own **pass/fail check screened out none of the designs the solver rejects**. All 179 valid committed designs pass it, and the solver rejects
**16 (8.9 %; [4.0, 14.5] target-clustered, [5.6, 14.0] nominal)**; of 180 committed designs,
163 were confirmed, 16 rejected and 1 unresolved by the solver. The count rises to 31 % of
the designs still certified if the tolerance is tightened to 2.5 %, and a fourth structure
adds 4 of 50 to the same pattern.

Three controls show what is not necessary. A surrogate trained from scratch produces
pretenders at a rate this study cannot distinguish from the transfer-learned one. Confining
the optimizer to the generator-feasible region reduces the count without removing it.
Budget-matched random search produces them as well, at counts the study cannot rank against
gradient descent. None of the three isolates a single cause; each still commits to the argmin of a surrogate that underestimates its own error at the geometry it selects. Running the identical protocol on the release this pipeline superseded gives 95 of 179 (53.1 %): the same structures,
thresholds and target-selection rule, a sixfold difference. A reliability claim of this
kind therefore describes a release rather than a method, and the release to audit is the one being shipped.

Two additional findings, the ordering just mentioned and, on one structure, a continuous error estimate that separates pretenders from confirmed designs almost perfectly, should be interpreted cautiously because of the small number of events. At these event rates, the present sample size can bound the effect but is insufficient to calibrate a threshold. The fourth structure, audited out of sample, lands inside the
rate band recorded for it (4 of 50) without making the ordering monotone. It repeats the
regularity of B and A, that pretenders are designs committed outside the generator's
feasible region; C remains the counterexample, with all five of its pretenders feasible, so the regularity holds for three of the four structures only.

The resulting protocol requires one reference-solver call per committed geometry: 103 to 4595 s here, against a per-run median of 1.2–2.3 min of optimization per design (wall clock from the run logs, `logs_pub/inverse_*_pub.log`). Box 1 states the procedure as six steps.


---

## Code, Data, and Compute

A single structure-agnostic codebase (PyTorch [57]) runs every stage for all four structures:
fine-tuning (strict load of TMM-pretrained checkpoints regenerated with the recipe of [1]), inverse
design (normalized-variable Adam with eight restarts), reference-solver
validation (torcwa 0.1.4.2 [46]), reliability accounting, the cross-structure
synthesis, the two mechanism controls (budget-matched random search and
single-start descent), the from-scratch and feasibility-constrained controls, the
fixed-order cross-solver probe and the Structure-D audit. Per-run artifacts for every structure × seed, the
aggregated evidence files behind every number in Section 3, and the figure
script that regenerates all figures from those files without recomputation are
in the repository (Data availability). Compute: one NVIDIA RTX 4070 Ti
SUPER; reference-solver time per committed geometry 103–4595 s at the printed
pipeline's per-wavelength adaptive order (median 1655 s; B 158 s, C 1662 s and A 3485 s
at the median), against 16–39 s for the as-submitted release's fixed order 5. All
simulation code is vendored at upstream commit `cb486b5` and the exact
input hashes are listed in the repository's manifests (Data availability). The raw data of [1]
are at `github.com/BigMountain87/PBTL-MIM`; that release, at the audited commit, shipped no checkpoints, so the
regenerated ones are in this paper's archive.

## Acknowledgements

This research did not receive any specific grant from funding agencies in the public,
commercial, or not-for-profit sectors.

Generative AI tools were used in preparing this work: Claude (Anthropic; Opus 4.8, Opus 5
and Fable 5.1) assisted with drafting, code refactoring and analysis scripts, and Codex
(OpenAI; gpt-6-astra) assisted with checking code, artifacts and manuscript claims. The
authors take responsibility for the experiments, the protocol, the claims and their
interpretation; the review transcripts are in the repository.

## Data availability

The code and artifacts behind this study are openly available at
`github.com/BigMountain87/PBTL-MIM-audit` (tag `v1-preprint`); a Zenodo archival deposit of
the same release is in preparation. The repository contains the code, the pre-specified protocol,
the per-run artifacts for every structure and training seed, the aggregated evidence
files behind every number reported here, and the figure scripts. It also contains the
TMM-pretrained checkpoints the surrogates are fine-tuned from (regenerated for A, B, C
and D with the recipe of [1], whose public release at the audited commit shipped no checkpoints), the as-submitted
arm's inputs and artifacts, and the Structure-D dataset. The raw datasets of [1] are
additionally available at `github.com/BigMountain87/PBTL-MIM`.

## Conflict of interest

The authors declare no competing interests.

## Author contributions

S.-B. Choi: conceptualization, methodology, software, formal analysis, investigation, data
curation, writing (original draft), visualization. J.-M. Choi: validation, writing (review
and editing). J. Kim: methodology, validation, writing (review and editing). C.-M. Kang: supervision, resources, project administration, writing (review and
editing).

## ORCID iDs

To be added at publication.

## References

[1] S.-B. Choi, J. Kim, C.-M. Kang, "Physics-Based Transfer Learning for Neural
Network Surrogates of Metal–Insulator–Metal Absorbers," *Photonics and
Nanostructures – Fundamentals and Applications* **72**(Part B), 101617 (2026). doi:10.1016/j.photonics.2026.101617

[2] N. I. Landy, S. Sajuyigbe, J. J. Mock, D. R. Smith, W. J. Padilla, "Perfect metamaterial absorber," *Phys. Rev. Lett.* **100**, 207402 (2008). doi:10.1103/PhysRevLett.100.207402

[3] W. Ma, Z. Liu, Z. A. Kudyshev, A. Boltasseva, W. Cai, Y. Liu, "Deep learning for the design of photonic structures," *Nat. Photonics* **15**, 77–90 (2021). doi:10.1038/s41566-020-0685-y

[4] J. Jiang, M. Chen, J. A. Fan, "Deep neural networks for the evaluation and design of photonic devices," *Nat. Rev. Mater.* **6**, 679–700 (2021). doi:10.1038/s41578-020-00260-1

[5] P. R. Wiecha, A. Arbouet, C. Girard, O. L. Muskens, "Deep learning in nano-photonics: inverse design and beyond," *Photonics Res.* **9**, B182 (2021). doi:10.1364/PRJ.415960

[6] S. So, T. Badloe, J. Noh, J. Bravo-Abad, J. Rho, "Deep learning enabled inverse design in nanophotonics," *Nanophotonics* **9**, 1041–1057 (2020). doi:10.1515/nanoph-2019-0474

[7] A. Khaireh-Walieh, D. Langevin, P. Bennet, O. Teytaud, A. Moreau, P. R. Wiecha, "A newcomer's guide to deep learning for inverse design in nano-photonics," *Nanophotonics* **12**, 4387–4414 (2023). doi:10.1515/nanoph-2023-0527

[8] Y. Qu, L. Jing, Y. Shen, M. Qiu, M. Soljačić, "Migrating knowledge between physical scenarios based on artificial neural networks," *ACS Photonics* **6**, 1168–1174 (2019). doi:10.1021/acsphotonics.8b01526

[9] R. Peng, S. Ren, J. Malof, W. J. Padilla, "Transfer learning for metamaterial design and simulation," *Nanophotonics* **13**, 2323–2334 (2024). doi:10.1515/nanoph-2023-0691

[10] L. Zhu, C. Lv, W. Hua, D. Huang, Y. Liu, "PTLOR-Net: physical transfer learning based optical response prediction network of metasurfaces," *ACS Photonics* **12**, 2624–2636 (2025). doi:10.1021/acsphotonics.5c00104

[11] G. Kim, J. Kim, "Efficient nanophotonic devices optimization using deep neural network trained with physics-based transfer learning methodology," *Sci. Rep.* **15**, 39854 (2025). doi:10.1038/s41598-025-23519-5

[12] W. Zhang, L. Deng, L. Zhang, D. Wu, "A survey on negative transfer," *IEEE/CAA J. Autom. Sinica* **10**, 305–329 (2023). doi:10.1109/JAS.2022.106004

[13] A. Kumar, S. Levine, "Model inversion networks for model-based optimization," *NeurIPS* **33** (2020). arXiv:1912.13464

[14] B. Trabucco, A. Kumar, X. Geng, S. Levine, "Conservative objective models for effective offline model-based optimization," *ICML*, PMLR **139** (2021). arXiv:2107.06882

[15] B. Trabucco, X. Geng, A. Kumar, S. Levine, "Design-Bench: benchmarks for data-driven offline model-based optimization," *ICML*, PMLR **162** (2022). arXiv:2202.08450

[16] C. Fannjiang, J. Listgarten, "Autofocused oracles for model-based design," *NeurIPS* **33** (2020). arXiv:2006.08052

[17] N. M. Alexandrov, J. E. Dennis, R. M. Lewis, V. Torczon, "A trust-region framework for managing the use of approximation models in optimization," *Struct. Optim.* **15**, 16–23 (1998). doi:10.1007/BF01197433

[18] A. J. Booker, J. E. Dennis, P. D. Frank, D. B. Serafini, V. Torczon, M. W. Trosset, "A rigorous framework for optimization of expensive functions by surrogates," *Struct. Optim.* **17**, 1–13 (1999). doi:10.1007/BF01197708

[19] A. I. J. Forrester, A. J. Keane, "Recent advances in surrogate-based optimization," *Prog. Aerosp. Sci.* **45**, 50–79 (2009). doi:10.1016/j.paerosci.2008.11.001

[20] Y. Jin, "Surrogate-assisted evolutionary computation: recent advances and future challenges," *Swarm Evol. Comput.* **1**, 61–70 (2011). doi:10.1016/j.swevo.2011.05.001

[21] M. C. Kennedy, A. O'Hagan, "Predicting the output from a complex computer code when fast approximations are available," *Biometrika* **87**, 1–13 (2000). doi:10.1093/biomet/87.1.1

[22] B. Peherstorfer, K. Willcox, M. Gunzburger, "Survey of multifidelity methods in uncertainty propagation, inference, and optimization," *SIAM Rev.* **60**, 550–591 (2018). doi:10.1137/16M1082469

[23] X. Meng, G. E. Karniadakis, "A composite neural network that learns from multi-fidelity data: application to function approximation and inverse PDE problems," *J. Comput. Phys.* **401**, 109020 (2020). doi:10.1016/j.jcp.2019.109020

[24] D. H. Brookes, H. Park, J. Listgarten, "Conditioning by adaptive sampling for robust design," *ICML*, PMLR **97** (2019). arXiv:1901.10060

[25] A. Tripp, E. Daxberger, J. M. Hernández-Lobato, "Sample-efficient optimization in the latent space of deep generative models via weighted retraining," *NeurIPS* **33** (2020). arXiv:2006.09191

[26] S. Yu, S. Ahn, L. Song, J. Shin, "RoMA: robust model adaptation for offline model-based optimization," *NeurIPS* **34** (2021). arXiv:2110.14188

[27] M. Kim, J. Gu, Y. Yuan, T. Yun et al., "Offline model-based optimization: comprehensive review," *Trans. Mach. Learn. Res.* (2026); arXiv:2503.17286 (2025).

[28] D. Manheim, S. Garrabrant, "Categorizing variants of Goodhart's law" (2018). arXiv:1803.04585

[29] J. Skalse, N. H. R. Howe, D. Krasheninnikov, D. Krueger, "Defining and characterizing reward hacking," *NeurIPS* **35** (2022). arXiv:2209.13085

[30] L. Gao, J. Schulman, J. Hilton, "Scaling laws for reward model overoptimization," *ICML*, PMLR **202** (2023). arXiv:2210.10760

[31] S. Surana, N. Grinsztajn, T. Atkinson, P. Duckworth, T. D. Barrett, "Overconfident oracles: limitations of in silico sequence design benchmarking," *ICML 2024 AI for Science Workshop* (2024). arXiv:2502.17246

[32] R. Pestourie, Y. Mroueh, T. V. Nguyen, P. Das, S. G. Johnson, "Active learning of deep surrogates for PDEs: application to metasurface design," *npj Comput. Mater.* **6**, 164 (2020). doi:10.1038/s41524-020-00431-2

[33] J. Jiang, J. A. Fan, "Global optimization of dielectric metasurfaces using a physics-driven neural network," *Nano Lett.* **19**, 5366–5372 (2019). doi:10.1021/acs.nanolett.9b01857

[34] B. A. Nosek, C. R. Ebersole, A. C. DeHaven, D. T. Mellor, "The preregistration revolution," *PNAS* **115**, 2600–2606 (2018). doi:10.1073/pnas.1708274114

[35] J. M. Hofman et al., "Pre-registration for predictive modeling" (2023). arXiv:2311.18807

[36] J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Larivière, A. Beygelzimer, F. d'Alché-Buc, E. Fox, H. Larochelle, "Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program)," *JMLR* **22**(164), 1–20 (2021). arXiv:2003.12206

[37] S. Kapoor, A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns* **4**, 100804 (2023). doi:10.1016/j.patter.2023.100804

[38] N. McGreivy, A. Hakim, "Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations," *Nat. Mach. Intell.* **6**, 1256–1269 (2024). doi:10.1038/s42256-024-00897-5

[39] I. Loshchilov, F. Hutter, "Decoupled weight decay regularization," *ICLR* (2019). arXiv:1711.05101

[40] R. Unni, K. Yao, Y. Zheng, "Deep convolutional mixture density network for inverse design of layered photonic structures," *ACS Photonics* **7**(10), 2703–2712 (2020). doi:10.1021/acsphotonics.0c00630

[41] D. P. Kingma, J. Ba, "Adam: a method for stochastic optimization," *ICLR* (2015). arXiv:1412.6980

[42] E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Am. Stat. Assoc.* **22**, 209–212 (1927). doi:10.1080/01621459.1927.10502953

[43] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," *Nat. Methods* **17**, 261–272 (2020). doi:10.1038/s41592-019-0686-2

[44] D. G. Bonett, T. A. Wright, "Sample size requirements for estimating Pearson, Kendall and Spearman correlations," *Psychometrika* **65**, 23–28 (2000). doi:10.1007/BF02294183

[45] R. A. Fisher, "The logic of inductive inference," *J. R. Stat. Soc.* **98**, 39–82 (1935). doi:10.2307/2342435

[46] C. Kim, B. Lee, "TORCWA: GPU-accelerated Fourier modal method and gradient-based optimization for metasurface design," *Comput. Phys. Commun.* **282**, 108552 (2023). doi:10.1016/j.cpc.2022.108552

[47] B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," *NeurIPS* **30** (2017). arXiv:1612.01474

[48] Y. Gal, Z. Ghahramani, "Dropout as a Bayesian approximation: representing model uncertainty in deep learning," *ICML*, PMLR **48** (2016). arXiv:1506.02142

[49] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern neural networks," *ICML*, PMLR **70** (2017). arXiv:1706.04599

[50] Y. Ovadia et al., "Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift," *NeurIPS* **32** (2019). arXiv:1906.02530

[51] M. Abdar et al., "A review of uncertainty quantification in deep learning: techniques, applications and challenges," *Inf. Fusion* **76**, 243–297 (2021). doi:10.1016/j.inffus.2021.05.008

[52] D. Liu, Y. Tan, E. Khoram, Z. Yu, "Training deep neural networks for the inverse design of nanophotonic structures," *ACS Photonics* **5**(4), 1365–1369 (2018). doi:10.1021/acsphotonics.7b01377

[53] J. Peurifoy et al., "Nanophotonic particle simulation and inverse design using artificial neural networks," *Sci. Adv.* **4**, eaar4206 (2018). doi:10.1126/sciadv.aar4206

[54] J. Jiang, J. A. Fan, "Simulator-based training of generative neural networks for the inverse design of metasurfaces," *Nanophotonics* **9**(5), 1059–1069 (2020). doi:10.1515/nanoph-2019-0330

[55] L. Cheng, P. Singh, F. Ferranti, "Transfer learning-assisted inverse modeling in nanophotonics based on mixture density networks," *IEEE Access* **12**, 55218–55224 (2024). doi:10.1109/ACCESS.2024.3383790

[56] S. Ren, W. Padilla, J. Malof, "Benchmarking deep inverse models over time, and the neural-adjoint method," *NeurIPS* **33** (2020). arXiv:2009.12919

[57] A. Paszke et al., "PyTorch: an imperative style, high-performance deep learning library," *NeurIPS* **32** (2019). arXiv:1912.01703

[58] P. Setinek, G. Galletti, T. Gross, D. Schnürer, J. Brandstetter, W. Zellinger, "SIMSHIFT: A benchmark for adapting neural surrogates to distribution shifts," arXiv:2506.12007 (2025).

[59] C. Fannjiang, S. Bates, A. N. Angelopoulos, J. Listgarten, M. I. Jordan, "Conformal prediction under feedback covariate shift for biomolecular design," *Proc. Natl. Acad. Sci. USA* **119** (43), e2204569119 (2022). doi:10.1073/pnas.2204569119

[60] K. Kandasamy, G. Dasarathy, J. Schneider, B. Póczos, "Multi-fidelity Bayesian optimisation with continuous approximations," *Proc. ICML*, PMLR **70** (2017). arXiv:1703.06240
