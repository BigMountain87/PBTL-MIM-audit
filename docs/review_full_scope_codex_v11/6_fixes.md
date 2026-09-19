# Full-scope codex review — slice 6_fixes (2026-09-12)

## 1. Verdict

**The fixes are incomplete.** Fresh execution found failures in cache validation, OOM persistence, surrogate identity, artifact matching and manuscript verification. The distributed main PDF also predates the final Markdown correction.

The archived numerical headline **does reproduce exactly**. These findings establish defective safeguards; they do not establish that the archived 16 pretenders are incorrect.

I inspected the stats and assigned-file diffs for `496ee19`, `43a65e3` and `60c4cc3`. **`5f3b0b1` is unavailable**: both `git show` and `git cat-file` fail. I did not open earlier review documents.

Execution used real NPZ/JSON artifacts, actual solver-module configuration and the current Python functions. Synthetic failure tests replaced solver outputs and filesystem operations in memory. They were not new electromagnetic simulations.

## 2. Must-Fix

**1. The live fingerprint still does not establish which solver implementation ran.**

At [oracle.py:93](src_v8/oracle.py:93), provenance contains settings, labels and a module filename—not implementation identity. Contract enforcement occurs during configuration, at [oracle.py:81](src_v8/oracle.py:81), but not before subsequent solves.

Executed counterexamples:

- Replacing the actual pub A module’s `simulate_single` with a zero-spectrum function left its fingerprint **unchanged** and still passed `configure_solver`.
- Changing the configured dtype to `torch.complex128` afterwards allowed a solve under the pub profile.
- `/pinned_other/solver.py` passed a contract whose root was `/pinned`, because [oracle.py:124](src_v8/oracle.py:124) uses a string prefix rather than resolved path containment.

The fingerprint does detect changed settings; it does not authenticate solver code. Enforce the contract at consumption, validate actual path containment and bind provenance to verified implementation/material dependencies.

**2. Pre-fingerprint caches bypass solver identity and the new NaN checks.**

The fallback at [oracle.py:205](src_v8/oracle.py:205) checks only wavelengths, geometry and Fourier orders. Cached output is accepted at [oracle.py:216](src_v8/oracle.py:216) without shape/finiteness validation.

Executed results:

- An old-format entry explicitly recording **legacy materials and complex128** was accepted under pub, with **zero solver calls**.
- An old-format entry containing NaN and `failed=False` returned **`cached=True, failed=False, mae=NaN`**.

This defeats both fixes through the compatibility branch. Validate cached spectra too, and reject or explicitly migrate entries lacking sufficient solver identity.

**3. OOM can still become a persisted failure through two paths.**

At [oracle.py:244](src_v8/oracle.py:244), the error is truncated to 200 characters. Retry detection uses the full exception, but the no-cache decision at [oracle.py:254](src_v8/oracle.py:254) uses the truncated string.

I raised:

```python
RuntimeError("x" * 210 + " out of memory")
```

The function attempted the solve four times, then **persisted the failure**. The untruncated OOM marker triggered retries but disappeared before the persistence decision.

Separately, stale cache files remain on disk after rejection. [oracle.py:276](src_v8/oracle.py:276) treats their existence as completion. Consequently, [rcwa_validate.py:64](src_v8/rcwa_validate.py:64) can enter assembly despite unsuccessful recomputation. Executing the assembly path with exhausted OOM results and existing-cache completion produced a final artifact with `failed[0]=True`.

Carry a structured transient-failure flag; require validated completion before writing final artifacts. “Not written to the per-design cache” is insufficient.

**4. Surrogate identity is neither bound to loaded bytes nor checked across all joined artifacts.**

There are three distinct failures:

- **Hash timing:** [inverse.py:31](src_v8/inverse.py:31) loads the model, but [inverse.py:191](src_v8/inverse.py:191) resolves and hashes the checkpoint again after optimization. Random search likewise loads at [random_baseline.py:37](src_v8/random_baseline.py:37) and hashes after oracle validation at line 152. A checkpoint replacement during the run produces metadata for different bytes.
- **Fresh `"missing"` identities:** [common.py:354](src_v8/common.py:354) emits `"missing"` when the file disappears; [common.py:363](src_v8/common.py:363) exempts that value from comparison. After loading the real A model, I simulated checkpoint disappearance during identity collection. Its fresh `"missing"` identity was accepted after restoration.
- **Unvalidated join:** [reliability.py:148](src_v8/reliability.py:148) compares only the inverse hash against the current checkpoint. It ignores the RCWA hash and geometry. Executing reliability with actual saved arrays, a matching inverse hash, and an RCWA artifact carrying both a conflicting hash and shifted geometry still produced the reliability output.

Hash the exact bytes supplied to model loading and retain that identity. Newly generated identities must never silently become `"missing"`. Validate inverse–RCWA geometry, targets and identity before combining them.

**5. Part D accepts false counts and rejects true ones.**

[check_manuscript_v10.py:202](scripts/check_manuscript_v10.py:202) combines unrelated quantities into one permitted-number set. Line 224 accepts any member regardless of what the sentence claims. Line 231 exempts whole sentences through incidental text such as “Table 9.”

I inserted each sentence into the current manuscript and executed the **full checker**, retaining its original content:

| Added sentence | Actual result |
|---|---|
| “The published pipeline produces **163 pretenders** among 179 valid designs.” | **PASS**, exit 0 |
| “The published pipeline produces **178 pretenders** among 179 valid designs; see Table 9.” | **PASS**, exit 0 |
| “Structure B has **9 pretenders** across its three training seeds.” | **FAIL**, exit 1 |
| “Structure C has **5 pretenders** across its three training seeds.” | **FAIL**, exit 1 |

The last two sentences are supported by today’s spectrum recomputation. The first passes because 163 is the *confirmed* count. Bind checks to quantity, structure, arm and threshold; a whitelist of integers cannot provide the claimed protection.

**6. D’s clarification changes the primary prediction after partial results could exist; it does not establish prospective blinding.**

The new rule at [structure_D_prespecification.md:94](docs/structure_D_prespecification.md:94) is mathematically defensible as the **union** of the two diagnostic-neighbour intervals. It is nevertheless broader than the original quantitative primary prediction at line 56.

Recomputed acceptable counts for 50 designs:

- Original quantitative band, 5–15%: **3–7**.
- New governing band, 3.4–15%: **2–7**.
- B–C interval, 8.3–15%: **5–7**.

Thus, **2/50 changes from outside the primary band to inside**. That is a substantive amendment.

The chronology does not support “before any D result existed”:

- D fine-tuning/statistics artifacts already existed and were committed on **September 11**.
- The clarification itself says **42/50 oracle designs were solved** at [line 78](docs/structure_D_prespecification.md:78).
- [oracle.py:263](src_v8/oracle.py:263) persists individual solves, and [rcwa_validate.py:60](src_v8/rcwa_validate.py:60) prints their MAEs before final assembly.

The clarification’s heading says **09:10**, whereas its filesystem modification time is **08:55:30** and the containing commit is **09:01:45**, all September 12, +09:00. These timestamps require reconciliation.

No D oracle cache/log or final RCWA artifact is available here to independently establish the partial outcome or anyone’s exposure to it. Absence of the final NPZ does **not** prove absence of outcome information. Report the new rule as an amendment, preserve the original primary outcome, and qualify the blinding claim.

**7. The distributed main PDF still contains the text that today’s Markdown fix removed.**

Executing the current converter in memory produced main TeX differing from the stored TeX in §4.5. [main.tex:1272](paper/mlst/main.tex:1272) still says:

> Structure A has 9 pretenders but 0 flags

I independently confirmed this on **page 19 of the PDF**. The updated Markdown instead distinguishes the superseded release and describes the pub-arm over-counting behavior.

Filesystem times explain the mismatch:

- Main TeX: **08:57:17**
- Main PDF: **08:57:23**
- Current Markdown: **09:00:36**

The saved build predates the final correction. Rebuild and verify the actual distribution artifacts after the final source edit. :codex-file-citation{path="paper/mlst/main.pdf" purpose="source"}

## 3. Should-Fix

- **Checker crashes for equivalent directory spelling.** `results_pub/` produced `UnboundLocalError: ... 's310'`. Initialization is conditional at [check_manuscript_v10.py:159](scripts/check_manuscript_v10.py:159), but use at line 224 is not. Initialize it independently and normalize paths.

- **The gate can announce completion after failures.** [run_pub_gate.sh:110](scripts/run_pub_gate.sh:110) and subsequent stages append failures and continue; line 134 unconditionally writes `gate_DONE.txt`. The new timestamp guard also covers only feasible outputs, not the M0 chain. Require final artifact validation before publishing completion.

- **Identity is lost again in downstream outputs.** Reliability’s JSON construction at [reliability.py:205](src_v8/reliability.py:205), restart-0 outputs at [restart0_validate.py:108](src_v8/restart0_validate.py:108), and random-baseline JSON at [random_baseline.py:143](src_v8/random_baseline.py:143) do not preserve the new identity. Carry it into every analytical artifact.

- **The converter’s end-matter “order” check does not check order.** [md_to_latex_v10.py:433](scripts/md_to_latex_v10.py:433) iterates the expected list and tests presence. Reversing the headings in the document would still satisfy it. Compare document positions.

## 4. Checked and OK

**Spectra and reported counts reproduced.** I loaded all 18 canonical A/B/C structure–seed inverse/RCWA pairs across both arms, recomputed channel-averaged absolute-error percentages, and compared their counts with reliability and pooled JSONs.

| Arm | Valid | Claimed | Confirmed | Pretenders | Maximum saved/recomputed MAE difference |
|---|---:|---:|---:|---:|---:|
| Pub | 179 | 179 | 163 | 16 | **0** |
| Legacy | 179 | 179 | 84 | 95 | **0** |

Pub structure counts were **A 2/59, B 9/60, C 5/60**.

Other checks:

- Actual pub A–D modules accepted adaptive and fixed-order-5 configuration. Actual legacy A–C modules accepted their default and explicit order 5. **No false rejection occurred in these legitimate configurations.**
- Changing live Fourier order changed the fingerprint. Every assigned consumer’s `simulate_cached` call passes `mod`; the separate `all_cached` completion check remains existence-only.
- A short OOM message caused exactly **four attempts and no cache write**. No infinite oracle retry loop was found.
- Fresh NaN output was marked failed.
- Reduced execution of `inverse.main` tested **all eight restarts**, two targets, A–D, and both parametrizations, using real geometry mappings with a test surrogate/optimizer. Selected endpoints equalled committed geometry exactly; every constrained endpoint passed feasibility.
- All **15 current pub inverse artifacts lack both new fields**. Their absence was checked directly; production artifacts therefore cannot validate the new endpoint/hash-writing behavior.
- Feasible random baselines record `feasible=True`; recomputed pretender counts **2, 0, 2** match their JSONs.
- Re-executed Structure D’s A/B/C aggregation matches the saved JSON after JSON normalization. D remains unavailable.

**Document numbering:** current in-memory conversion produced main Figures **1–6**, Tables **1–11**, supplementary Figures **S1–S2**, and Tables **S1–S7**. These match the Markdown order and existing AUX/PDF numbering. Supplementary TeX reproduced exactly. Existing logs contain zero undefined-reference or overfull-box messages.

**Build limitation:** this workspace is read-only, so I could not write and compile fresh PDFs. Conversion and existing numbering were verified; a fresh two-document PDF build remains unverified.

## 5. Files read

The opening table records all **12 assigned files, 2,786 lines, read end to end**. Additional evidence was limited to their relevant input/output artifacts and scoped Git history. Earlier review documents were not opened. The working tree remains unchanged.
