# Full-scope codex review — slice 5_process (2026-09-12)

## 1. Verdict

**Major revision for process integrity and verification tooling.** The headline accounting reproduces. The available chronology supports recording D’s prediction before its inverse audit. However, the D specification is internally inconsistent, some task-completion claims exceed their cited commits, and the manuscript checker demonstrably accepts a grossly false numerical claim.

Reviewed checkout: `030eecb`. I opened no separate earlier review document. Findings below come from the assigned files, their historical versions, and computations performed during this review. No repository files were modified.

## 2. Must-Fix

**M1. The September disclosure rewrites the recorded rationale and overstates what was fixed on September 8.**

The manuscript says the September 7 seed reduction was because of “cost, not statistics” ([manuscript_v11.md:421](paper/manuscript_v11.md:421)). But the contemporaneous amendment at `cf5d26e` explicitly justified it by observing six pretenders and asserting that three seeds would not provide power. That language remains at [PLAN_v10_amendment_pub.md:88](docs/PLAN_v10_amendment_pub.md:88). The cost rationale appears as a subsequent correction at lines 126–128.

The disclosure should distinguish **the rationale recorded when the decision was made** from **the revised justification**. Otherwise, it understates the role of an outcome-dependent statistical argument.

Additionally, [manuscript_v11.md:433](paper/manuscript_v11.md:433) says the direction, incomplete-pair handling, and two-comparison family were fixed on September 8. I read the amendment as committed at `4fe3544`: it specifies pairing, Δ, Wilcoxon, median/bootstrap CI, McNemar, and power reporting, but **does not explicitly specify incomplete-pair handling or the multiplicity family**. The accompanying historical JSON contains empty `conditions` and six `missing` entries; it does not establish those extra commitments.

Supply the exact contemporaneous specification for those details or narrow the claim. This evidence does **not** establish that the endpoint was chosen after control results.

**M2. D’s specification does not define a coherent success/falsification rule.**

At [structure_D_prespecification.md:53](docs/structure_D_prespecification.md:53):

- “Between B and C” means **8.33–15%** using the document’s own observed rates, not 5–15%.
- With 50 designs, 5–15% permits **3–7 pretenders**, not “3 and 8.” I enumerated the admissible integer counts; eight is 16%.
- Line 60 calls **16.9% C’s pretender rate**. C’s recomputed pretender rate is **5/60 = 8.33%**; 16.9% is its operating-band fidelity MAE.

The last error propagates into `results_pub/structure_d_v10.json:prediction.falsified_above_pct = 16.9`. It therefore affects the executable interpretation, not just prose.

Preserve the original specification and append a dated clarification identifying which rule governs. Do not silently replace it once D outcomes become available.

**M3. The checker’s “full consistency audit” claim is false. I executed a counterexample.**

I inserted, in memory immediately before References:

> The published pipeline produced exactly 178 pretenders among 179 valid designs.

Running the unmodified checker against that modified text returned **exit 0, `check_manuscript: PASS`**. The genuine text also returned PASS, with 37/37 checks.

The mechanism is explicit at [check_manuscript_v10.py:33](scripts/check_manuscript_v10.py:33): `want()` tests substring presence anywhere before References. Retaining the correct number elsewhere satisfies the check despite an additional contradictory claim.

Other concrete gaps:

- Optional evidence files are silently omitted by lines 25–28. Missing statistics suppress their checks at lines 88–102.
- Several files are loaded but their substantive contents are never checked, including detector, feasibility, lookup, and recoverability evidence.
- NPZ verification at lines 110–117 checks only whether the maximum stored MAE lies between zero and 100; it does not recompute MAE or accounting.
- The asserted T4 numerator is hard-coded as zero at line 60.

The opening description—“every quoted number is recomputed”—substantially overstates this implementation. Require an explicit evidence inventory and bind checked values to identified claims/tables. Report the remaining text as unchecked.

**M4. Two progress claims are contradicted by their committed artifacts.**

- **T59:** [PROGRESS.md:84](docs/PROGRESS.md:84) says `stats_supplement_v9.json` was generated in `f2b3671`. That file is **absent from that commit’s entire `results_pub` tree**, not merely absent from its diff. It first appears in the inspected subsequent commit `ab67d15`. This matters because the verifier at `f2b3671` consequently runs only **25 numerical checks**, silently omitting the statistics block.
- **T70:** [PROGRESS.md:95](docs/PROGRESS.md:95) says the seed-42 results falsely labelled “three-seed pooled” were corrected. Yet `879fd50:results_pub/structure_d_v10.json`, and the current artifact, retain `per_structure_pooled_3seed` with **n=20 for every structure** and A/B/C pretender counts **1/3/2**. Actual three-seed counts recompute to **2/9/5**, with denominators **59/60/60**.

The second is especially understated: the record presents the defect as repaired while the deliverable still exposes the incorrect sample provenance.

**M5. The prescribed LaTeX conversion path is inconsistent with the pub manuscript.**

[PLAN_v11_rewrite.md:62](docs/PLAN_v11_rewrite.md:62) prescribes `md_to_latex_v10.py --build`. But the converter still selects **`manuscript_v10.md`, `supplementary_v10.md`, and `figures_v9/`** at [md_to_latex_v10.py:19](scripts/md_to_latex_v10.py:19).

Meanwhile, its figure map was changed to the six-figure pub convention, removing key `"7"` and changing Figures 5–6. Executing `figure_tex("7", ...)` raises **`KeyError: '7'`**. The build checks nevertheless still demand seven figures at line 416.

Thus the current converter combines legacy sources with pub numbering; the historical T42 PASS cannot validate this changed configuration. Make manuscript, supplement, figure directory, and expected numbering explicit inputs, then verify the actual pub build.

## 3. Should-Fix

- **The execution ledger is abandoned while still claiming completeness.** [compute_log_v10.md:3](docs/compute_log_v10.md:3) calls itself the ledger of all GPU work and requires updates on starts/finishes. Lines 33–35 still show September 6 partial execution and three-seed controls waiting, despite completed control artifacts and D training. Its git history contains only the September 6 addition. Reconcile actual runs, finishes, failures, and revised scope before using it for compute/provenance statements.

- **Artifact provenance does not identify the audit code or exact inputs adequately.** Across all 13 fine-tuning JSONs, the only commit/hash/date-like top-level field found was `upstream_commit`; checkpoint references are paths, without content hashes. Canonical ABC files also lack the `env` block present in later runs. Add the audit repository revision, dirty-state/content digest, input hashes, and execution timestamps. The upstream revision alone cannot identify the local audit implementation.

- **Build verification overstates its checks.** At [md_to_latex_v10.py:419](scripts/md_to_latex_v10.py:419), “end matter in order” tests membership in a preordered list, not positions in the PDF. I executed that expression with every heading reversed: **True**. Lines 393–425 also do not reject overfull boxes or verify that every expected label exists; they inspect labels that survived conversion.

- **Task status no longer follows the declared dependency rules.** [PROGRESS.md:99](docs/PROGRESS.md:99) requires dependencies to be done, yet T55 is done while T54 remains in progress, and T64 is done while T62 remains todo. Record explicit exceptions or update statuses; the current table cannot establish execution order.

## 4. Checked and OK

**The headline reproduces from spectra, not just JSON summaries.** I loaded all nine core RCWA NPZs, recomputed absorption-channel MAEs from stored reference and target spectra, applied finite-value validity and the two 5% conditions, and compared with the JSON accounting.

| Structure | Valid | Surrogate-claimed | Reference-confirmed | Pretenders |
|---|---:|---:|---:|---:|
| A | 59 | 59 | 57 | 2 |
| B | 60 | 60 | 51 | 9 |
| C | 60 | 60 | 55 | 5 |
| Total | **179** | **179** | **163** | **16** |

The maximum discrepancy between recomputed and stored reference MAE was **8.9 × 10⁻¹⁶ percentage points**. This verifies accounting from stored spectra; it is not a fresh solver run.

**Control calculations reproduce.** I checked exact target-index equality and independently recomputed paired shifts, Wilcoxon p-values, pretender counts, and exact McNemar tests:

| Control | Pairs | Median shift, pp | Wilcoxon p | Pretenders, base→control | McNemar p |
|---|---:|---:|---:|---:|---:|
| M0 | 60 | +0.1352326706 | 0.1755657701 | 6→4 | 0.6875 |
| Feasible | 60 | −0.0011217891 | 0.0070526762 | 6→3 | 0.375 |

These match `control_analysis_v10.json`. I did not independently regenerate its bootstrap intervals.

**D’s timing allegation is not supported by the available evidence.**

- Specification commit `c8a6027`: **September 10, 02:37:46 KST**.
- Local specification birth/mtime: **02:37:45 KST**.
- No D artifact exists in that commit’s `results_pub` tree.
- First tracked D result commit: `12ca2b4`, **September 11, 16:03:37 KST**.
- Local D fine-tuning JSON/statistics birth/mtime: **September 11, 15:02:31 KST**.

The apparent September 9 git date is the same instant expressed in `+02:00`. These records support the claimed sequence. They cannot prove that no untracked or remote artifact ever existed.

**Ten progress rows were spot-checked against their cited commits or, for T70, its recording commit:**

| Row | Commit | Check performed and result |
|---|---|---|
| T41 | `d64cceb` | Executed historical checker against historical manuscript/artifacts: **41/41, exit 0**, matching the claim. |
| T42 | `d50f70b` | Historical converter has the original seven-figure map. Build/overfull claims were **not re-executed**; current regression is M5. |
| T52 | `e81085b` | Historical fine-tuning JSONs contain **1.757402 / 1.722123 / 2.107106%**, matching the recorded rounded results. |
| T55 | `12ca2b4` | Commit tree contains completed control evidence; M0 MAEs reproduce the listed **2.586 / 1.887 / 1.957%**. |
| T57 | `dfa2959` | Historical §3.10 contains the comparison; historical checker returns **37/37, exit 0**. |
| T58 | `f2b3671` | Historical checker returns **9/25, 16 failures**, reproducing the claimed replacement inventory; blanket no-silent-skips claim is unsupported. |
| T59 | `f2b3671` | Nine of ten listed JSONs present; statistics JSON **absent**. See M4. |
| T60 | `ab67d15` | Commit adds the claimed eight figure stems, each PDF and PNG, and the missing statistics JSON. |
| T64 | `12ca2b4` | Commit contains both control disclosures and September amendments; numerical point estimates/tests reproduce above. |
| T70 | `879fd50` | Claimed label correction is contradicted by the committed D JSON. See M4. |

## 5. Files read

The opening table inventories all **eight assigned text/code files and 54 JSON files**. Text/code files were read end to end; every JSON was parsed and its metadata/provenance inspected throughout.

Additional reads were limited to the requested manuscript disclosure, relevant historical manuscript/artifact versions, git records, filesystem timestamps, and NPZ/JSON evidence used for the execution checks. No separate earlier review document was opened or used as evidence.
