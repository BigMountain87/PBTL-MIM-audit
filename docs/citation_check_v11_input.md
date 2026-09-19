# Citation attribution check — input (generated)

Sentences tagged `[CHECK]` (paragraph context) and the reference entries they cite.

## Passage 1 (manuscript line ~92)

Managing an approximate model inside an optimization loop is an old problem. Trust-region
model management [17] and the surrogate-management framework
[18] established the principle that a cheap model may drive the search only if
the expensive one periodically re-certifies the iterate, and the surrogate-based and
surrogate-assisted optimization literatures have refined that bargain ever since
[19, 20]. `[CHECK]` Multi-fidelity modelling formalises the same
two-solver structure we have here — an autoregressive or composite model relates a cheap
low-fidelity source to a scarce high-fidelity one [21–23] — so the TMM→RCWA transfer audited in this paper is naturally read as a
multi-fidelity scheme in which the cheap solver enters through pre-trained weights rather
than through an explicit discrepancy term. `[CHECK]` What that literature supplies, and what a
forward transferability score does not, is a re-certification step: the protocol below is
essentially trust-region model management reduced to its minimal form, one high-fidelity call
at the committed point. `[CHECK]`

## Passage 2 (manuscript line ~106)

The failure mode we measure is the one offline model-based optimization has named. Optimizing
against a fixed learned objective drives the solution into regions where that objective is
over-optimistic, and the remedies proposed — conditioning by adaptive sampling
[24], model inversion networks [13], weighted retraining
[25], autofocused oracles [16] and robust model adaptation
[26] — all constrain the search to regions where the model is trustworthy, or repair
the model as the search moves. `[CHECK]` Design-Bench standardises the resulting benchmark
suite [15], and recent reviews survey the field [27]. `[CHECK]`
The same structure appears wherever a proxy is optimized: it is Goodhart's law
[28], reward hacking [29] and reward-model overoptimization
[30]. `[CHECK]` Best-of-*n* selection against a proxy is the sharpest analogy to our
mechanism control — choosing the argmin of an imperfect surrogate over eight restarts, or over
6400 random draws, degrades true quality even though the proxy score improves, which is why
budget-matched random search manufactures as many pretenders as gradient descent
(Section 3.6). `[CHECK]` In silico design benchmarking has independently reported that a
learned oracle used to score generated candidates is itself the weak link
[31]. `[CHECK]`

## Passage 3 (manuscript line ~124)

Two responses to an unreliable surrogate must be distinguished. (i) *Surrogate-improving*
active learning spends its simulation budget on making the model better where the optimizer is
looking [32], and physics-driven or simulator-in-the-loop training folds the
solver into the training objective itself [33]. `[CHECK]` (ii) *Certification-only*
oracle calls spend the budget on checking the committed answer and change the model not at all
[17]. `[CHECK]` The first is the better long-run investment; the second is what
a practitioner who has already trained a surrogate can do today, and it is the only one of the
two whose cost is bounded by one call per design. `[CHECK]` This paper measures how much that
single call buys. `[CHECK]`

## Passage 4 (manuscript line ~134)

Finally, the reliability of the claim itself. Pre-registration separates confirmatory from
exploratory analysis [34] and has recently been argued for predictive modelling
specifically [35]; reproducibility programmes ask for the artifacts that let a
reader re-derive a number [36]; and machine-learning-based science has been shown
to fail systematically through leakage [37] and through weak baselines and
reporting biases [38]. `[CHECK]` We follow that line by freezing the protocol,
its thresholds and its hypotheses before the main runs, labelling every post-hoc analysis
as such, and releasing the artifact behind every number. `[CHECK]`

## Passage 5 (manuscript line ~1188)

**Uncertainty quantification is the obvious missing ingredient, and we tested a
piece of it.** The standard toolkit — deep ensembles [47], MC
dropout [48], calibration correction [49] and the benchmark
evidence that all of them degrade under dataset shift [50], surveyed
in [51] — is designed for exactly this situation: a model asked about
inputs unlike its training data. The committed geometries here are such inputs by
construction, since the optimizer searches for them. Our detector benchmark
(Section 3.7) includes the cheapest usable member of that family, the
disagreement of a three-member deep ensemble formed from the three training
seeds, together with a held-out-ensemble error and a *k*-NN distance to the
training set. Their AUROCs are reported there: the ensemble quantities reach 0.95–0.98
on C and 0.73–0.80 on B, the training-set distance 0.80 on B but 0.42 on C, and on A none
of them is distinguishable from chance. What we have *not* tested is a
calibrated Bayesian surrogate trained for the purpose, or a conformal wrapper
with a coverage guarantee; that is the natural next experiment and the main
qualification on the negative result (W9). `[CHECK]`

## Passage 6 (manuscript line ~1268)

Two mappings between that row set and our taxonomy are worth stating, and are
flagged for author confirmation. TODO-AUTHOR: the neural-adjoint *boundary loss*
[56] penalises exactly the excursion our **T2** box-edge flag detects,
so T2 can be read as a diagnostic version of that regulariser — 9–18 of 20
committed designs per run would have been pushed back by it. `[CHECK]`
TODO-AUTHOR: re-simulating the top-*k* neural-adjoint candidates is the same
operation as our **Δ audit**, differing only in that we re-simulate the single
committed geometry rather than a shortlist, which is what makes the cost exactly
one solver call per design. `[CHECK]`

## Reference entries cited above

[13] A. Kumar, S. Levine, "Model inversion networks for model-based optimization," *NeurIPS* **33** (2020). arXiv:1912.13464

[15] B. Trabucco, X. Geng, A. Kumar, S. Levine, "Design-Bench: benchmarks for data-driven offline model-based optimization," *ICML*, PMLR **162** (2022). arXiv:2202.08450

[16] C. Fannjiang, J. Listgarten, "Autofocused oracles for model-based design," *NeurIPS* **33** (2020). arXiv:2006.08052

[17] N. M. Alexandrov, J. E. Dennis, R. M. Lewis, V. Torczon, "A trust-region framework for managing the use of approximation models in optimization," *Struct. Optim.* **15**, 16–23 (1998). doi:10.1007/BF01197433

[18] A. J. Booker, J. E. Dennis, P. D. Frank, D. B. Serafini, V. Torczon, M. W. Trosset, "A rigorous framework for optimization of expensive functions by surrogates," *Struct. Optim.* **17**, 1–13 (1999). doi:10.1007/BF01197708

[19] A. I. J. Forrester, A. J. Keane, "Recent advances in surrogate-based optimization," *Prog. Aerosp. Sci.* **45**, 50–79 (2009). doi:10.1016/j.paerosci.2008.11.001

[20] Y. Jin, "Surrogate-assisted evolutionary computation: recent advances and future challenges," *Swarm Evol. Comput.* **1**, 61–70 (2011). doi:10.1016/j.swevo.2011.05.001

[21] M. C. Kennedy, A. O'Hagan, "Predicting the output from a complex computer code when fast approximations are available," *Biometrika* **87**, 1–13 (2000). doi:10.1093/biomet/87.1.1

[22] B. Peherstorfer, K. Willcox, M. Gunzburger, "Survey of multifidelity methods in uncertainty propagation, inference, and optimization," *SIAM Rev.* **60**, 550–591 (2018). doi:10.1137/16M1082469

[23] X. Meng, G. E. Karniadakis, "A composite neural network that learns from multi-fidelity data," *J. Comput. Phys.* **401**, 109020 (2020). doi:10.1016/j.jcp.2019.109020

[24] D. H. Brookes, H. Park, J. Listgarten, "Conditioning by adaptive sampling for robust design," *ICML*, PMLR **97** (2019). arXiv:1901.10060

[25] A. Tripp, E. Daxberger, J. M. Hernández-Lobato, "Sample-efficient optimization in the latent space of deep generative models via weighted retraining," *NeurIPS* **33** (2020). arXiv:2006.09191

[26] S. Yu, S. Ahn, L. Song, J. Shin, "RoMA: robust model adaptation for offline model-based optimization," *NeurIPS* **34** (2021). arXiv:2110.14188

[27] M. Kim, J. Gu, Y. Yuan, T. Yun et al., "Offline model-based optimization: comprehensive review" (2025). arXiv:2503.17286

[28] D. Manheim, S. Garrabrant, "Categorizing variants of Goodhart's law" (2018). arXiv:1803.04585

[29] J. Skalse, N. H. R. Howe, D. Krasheninnikov, D. Krueger, "Defining and characterizing reward hacking," *NeurIPS* **35** (2022). arXiv:2209.13085

[30] L. Gao, J. Schulman, J. Hilton, "Scaling laws for reward model overoptimization," *ICML*, PMLR **202** (2023). arXiv:2210.10760

[31] S. Surana, N. Grinsztajn, T. Atkinson, P. Duckworth et al., "Overconfident oracles: limitations of in silico sequence design benchmarking" (2025). arXiv:2502.17246

[32] R. Pestourie, Y. Mroueh, T. V. Nguyen, P. Das, S. G. Johnson, "Active learning of deep surrogates for PDEs: application to metasurface design," *npj Comput. Mater.* **6**, 164 (2020). doi:10.1038/s41524-020-00431-2

[33] J. Jiang, J. A. Fan, "Global optimization of dielectric metasurfaces using a physics-driven neural network," *Nano Lett.* **19**, 5366–5372 (2019). doi:10.1021/acs.nanolett.9b01857

[34] B. A. Nosek, C. R. Ebersole, A. C. DeHaven, D. T. Mellor, "The preregistration revolution," *PNAS* **115**, 2600–2606 (2018). doi:10.1073/pnas.1708274114

[35] J. M. Hofman et al., "Pre-registration for predictive modeling" (2023). arXiv:2311.18807

[36] J. Pineau et al., "Improving reproducibility in machine learning research," *JMLR* **22**(164) (2021). arXiv:2003.12206

[37] S. Kapoor, A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns* **4**, 100804 (2023). doi:10.1016/j.patter.2023.100804

[38] N. McGreivy, A. Hakim, "Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations," *Nat. Mach. Intell.* **6**, 1256–1269 (2024). doi:10.1038/s42256-024-00897-5

[47] B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," *NeurIPS* **30** (2017). arXiv:1612.01474

[48] Y. Gal, Z. Ghahramani, "Dropout as a Bayesian approximation: representing model uncertainty in deep learning," *ICML*, PMLR **48** (2016). arXiv:1506.02142

[49] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern neural networks," *ICML*, PMLR **70** (2017). arXiv:1706.04599

[50] Y. Ovadia et al., "Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift," *NeurIPS* **32** (2019). arXiv:1906.02530

[51] M. Abdar et al., "A review of uncertainty quantification in deep learning: techniques, applications and challenges," *Inf. Fusion* **76**, 243–297 (2021). doi:10.1016/j.inffus.2021.05.008

[56] S. Ren, W. Padilla, J. Malof, "Benchmarking deep inverse models over time, and the neural-adjoint method," *NeurIPS* **33** (2020). arXiv:2009.12919
