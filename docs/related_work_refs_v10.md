# New references added by T32/T33 — one row per work, with the sentence it supports

Checked 2026-09-06 against Crossref (DOIs) and the arXiv API; every identifier returned a
matching record. Full returned metadata is in `paper/related_work_draft_v10.md`.

| key | identifier | supports (sentence / location) |
|---|---|---|
| alexandrov1998 | 10.1007/BF01197433 | §1.3 ¶1 "trust-region model management … established the principle that a cheap model may drive the search only if the expensive one periodically re-certifies the iterate"; §4.4 (i) surrogate-improving calls |
| booker1999 | 10.1007/BF01197708 | §1.3 ¶1 same sentence (surrogate-management framework); §4.4 (ii) certification-only calls |
| forrester2009 | 10.1016/j.paerosci.2008.11.001 | §1.3 ¶1 "the surrogate-based … literatures have refined that bargain ever since" |
| jin2011 | 10.1016/j.swevo.2011.05.001 | §1.3 ¶1 same sentence (surrogate-assisted evolutionary computation) |
| kennedy2000 | 10.1093/biomet/87.1.1 | §1.3 ¶1 "Multi-fidelity modelling formalises the same two-solver structure" (autoregressive co-kriging) |
| peherstorfer2018 | 10.1137/16M1082469 | §1.3 ¶1 same sentence (multifidelity survey) |
| meng2020 | 10.1016/j.jcp.2019.109020 | §1.3 ¶1 same sentence (composite multi-fidelity network) |
| brookes2019 | arXiv:1901.10060 | §1.3 ¶2 "conditioning by adaptive sampling" among the offline-MBO remedies |
| kumar2019 | arXiv:1912.13464 (= ref [8]) | §1.3 ¶2 "model inversion networks" |
| tripp2020 | arXiv:2006.09191 | §1.3 ¶2 "weighted retraining" |
| fannjiang2020 | arXiv:2006.08052 (= ref [11]) | §1.3 ¶2 "autofocused oracles" |
| yu2021 | arXiv:2110.14188 | §1.3 ¶2 "robust model adaptation"; §4.3 the RoMA/T4 sentence (pretenders sit in smooth regions, so a smoothness prior would not remove them) |
| trabucco2022 | arXiv:2202.08450 (= ref [10]) | §1.3 ¶2 "Design-Bench standardises the resulting benchmark suite" |
| omboffline2025 | arXiv:2503.17286 | §1.3 ¶2 "recent reviews survey the field" — **author decision: keep or strike** |
| overconfident2025 | arXiv:2502.17246 | §1.3 ¶2 "a learned oracle used to score generated candidates is itself the weak link" — **author decision: keep or strike** |
| manheim2018 | arXiv:1803.04585 | §1.3 ¶2 and §4.3 "it is Goodhart's law" |
| skalse2022 | arXiv:2209.13085 | §1.3 ¶2 and §4.3 "reward hacking" |
| gao2023 | arXiv:2210.10760 | §1.3 ¶2 and §4.3 "reward-model overoptimization"; the best-of-n parallel to the mechanism control |
| pestourie2020 | 10.1038/s41524-020-00431-2 | §1.3 ¶3 (i) surrogate-improving active learning; §4.4 |
| jiang2019 | 10.1021/acs.nanolett.9b01857 | §1.3 ¶3 physics-driven / simulator-in-the-loop training; §4.4 |
| nosek2018 | 10.1073/pnas.1708274114 | §1.3 ¶4 "Pre-registration separates confirmatory from exploratory analysis" |
| hofman2023 | arXiv:2311.18807 | §1.3 ¶4 "argued for predictive modelling specifically" |
| pineau2021 | arXiv:2003.12206 | §1.3 ¶4 "reproducibility programmes ask for the artifacts" |
| kapoor2023 | 10.1016/j.patter.2023.100804 | §1.3 ¶4 "fail systematically through leakage"; §2.6 row 3 |
| mcgreivy2024 | 10.1038/s42256-024-00897-5 | §1.3 ¶4 "weak baselines and reporting biases"; §2.6 row 7 |

## Already in the list (no new entry needed)

[8] Kumar & Levine (model inversion networks), [9] Trabucco et al. (conservative objective
models), [10] Design-Bench, [11] Fannjiang & Listgarten (autofocused oracles) — the §1.3 draft
cites them by their `[NEW-*]` keys for readability; T34 maps those keys back onto the existing
numbers rather than duplicating the entries.

## Added by T33 (photonics reviews, TL precedents, neural adjoint, UQ)

| key | identifier | supports (sentence / location) |
|---|---|---|
| ma2021 | 10.1038/s41566-020-0685-y | §1.1 "several reviews survey the field and its inverse-design applications" |
| jiangfan2021 | 10.1038/s41578-020-00260-1 | §1.1 same sentence (Nature Reviews Materials survey) |
| wiecha2021 | 10.1364/prj.415960 | §1.1 same sentence |
| so2020 | 10.1515/nanoph-2019-0474 | §1.1 same sentence |
| khaireh2023 | 10.1515/nanoph-2023-0527 | §1.1 same sentence (newcomer's guide) |
| qu2019 | 10.1021/acsphotonics.8b01526 | §1.1 "knowledge has been migrated between physical scenarios" |
| peng2024 | 10.1515/nanoph-2023-0691 | §1.1 "between metamaterial design tasks" |
| zhu2025 | 10.1021/acsphotonics.5c00104 | §1.1 "from physical priors into optical-response prediction" |
| kimkim2025 | 10.1038/s41598-025-23519-5 | §1.1 the PBTL acronym-collision clause (same acronym, different scheme) |
| negtransfer2023 | 10.1109/jas.2022.106004 | §1.1 "negative transfer is a documented failure mode with its own survey literature" |
| ren2020 | arXiv:2009.12919 | §4.6 neural-adjoint row; the two TODO-AUTHOR mapping sentences (boundary loss ≈ T2; top-k re-simulation ≈ Δ audit) |
| lakshmi2017 | arXiv:1612.01474 | §4.4 UQ paragraph, deep ensembles |
| gal2016 | arXiv:1506.02142 | §4.4 UQ paragraph, MC dropout |
| guo2017 | arXiv:1706.04599 | §4.4 UQ paragraph, calibration |
| ovadia2019 | arXiv:1906.02530 | §4.4 UQ paragraph and §4.6 UQ row, degradation under dataset shift |
| abdar2021 | 10.1016/j.inffus.2021.05.008 | §4.4 UQ paragraph and §4.6 UQ row, UQ survey |

Note: `jiang2019` (10.1021/acs.nanolett.9b01857, physics-driven / simulator-in-the-loop) is
already in the T32 list and is reused for the §4.6 simulator-in-the-loop row. The
Structure-C-adjacent [7] (Cheng et al., IEEE Access 2024) is already reference [7].
All identifiers checked 2026-09-06; every one returned a matching Crossref/arXiv record.
