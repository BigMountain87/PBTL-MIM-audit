# Full-scope codex review — slice 1_solver (2026-09-12)

## 1. Verdict

**The saved headline reproduces, but the oracle does not enforce the solver identity it claims to enforce.** Both cache-validation paths have executable counterexamples. Solver failures also have problematic persistence and validity handling.

I independently loaded all 18 canonical A/B/C RCWA NPZ artifacts across both arms, decoded their available JSON metadata, and recomputed MAEs from spectra:

| Arm | Valid | Surrogate-claimed among valid | Pretenders | Maximum stored/recomputed MAE difference |
|---|---:|---:|---:|---:|
| Pub | 179 | 179 | 16 | 0 |
| Legacy | 179 | 179 | 95 | 0 |

These are arithmetic reproductions from stored spectra, **not fresh electromagnetic solves**. No per-design cache files were available locally; cache counterexamples used in-memory fixtures and the actual assigned functions. No earlier review documents were opened.

Two requested conclusions remain outside the strict assigned-file scope: whether `inverse.py` is onto each feasible region, and whether copied feature definitions independently match upstream training scripts.

## 2. Must-Fix

**1. “Pinning” records settings without enforcing the required tree, dtype, or material model.**

Sources: [common.py:41](src_v8/common.py:41), [oracle.py:57](src_v8/oracle.py:57).

`UPSTREAM_COMMIT` comes from the directory basename, rather than verified source contents. `configure_solver()` checks only the presence/absence of `adaptive_order`; it neither checks the loaded module’s origin nor enforces dtype/material identity.

Executed counterexamples under `pub`:

- Changing the actual module’s dtype to `torch.complex128` was accepted.
- Changing `MATERIAL_MODEL` to `legacy` was accepted.

The returned provenance honestly recorded those changes, but the supposed profile contract did not reject them. Likewise, placing a directory first on `sys.path` does not replace previously imported `src` modules.

**Fix:** verify loaded module origins and source identity; enforce expected settings for ordinary audit runs. Make intentional deviations explicit probe configurations.

**2. Neither cache path establishes the identity of the live solver.**

Sources: [oracle.py:85](src_v8/oracle.py:85), [oracle.py:137](src_v8/oracle.py:137), [oracle.py:165](src_v8/oracle.py:165).

- **Fingerprint path:** hashes a global `_PROVENANCE` snapshot. Changing live settings afterward does not change the fingerprint. I created a matching in-memory cache entry, changed the live solver to fixed order 1, and received `cached=True`, with cached orders `[13, 9, 9]` although current orders were `[1, 1, 1]`.
- **Pre-fingerprint path:** checks geometry, wavelengths, and orders only. An entry with no solver provenance was accepted. Identical Fourier orders cannot establish identical dtype, materials, grid, or source implementation.

**Fix:** derive identity from the solver actually being used, with immutable configuration and verified source identity. Reject unverifiable old entries or migrate them using independently established provenance.

These tests demonstrate acceptance holes; they do not establish that the archived headline used contaminated caches.

**3. Nonfinite solver output is marked successful and cached.**

Sources: [oracle.py:190](src_v8/oracle.py:190), [rcwa_validate.py:79](src_v8/rcwa_validate.py:79).

After `simulate_single()` returns, the code sets `failed=False` without checking channel shape or finiteness.

Executed with a mocked solver returning three NaNs:

```text
failed=False
mae=nan
cache write attempted=True
```

The assembler accepts this as a successful solve, and its reported success count uses `~failed`. This undermines the meaning of “valid reference evaluation.”

**Fix:** validate expected channel shapes and finite values before accepting or caching success. Make validity explicit in the assembled artifact.

None of the 18 canonical artifacts contained an unflagged nonfinite MAE in my recomputation.

**4. Exhausted OOM retries become permanently cached failures.**

Sources: [oracle.py:197](src_v8/oracle.py:197), [oracle.py:209](src_v8/oracle.py:209).

The comment promises “never cache an OOM,” but after the fourth failed attempt execution reaches the unconditional cache write. Subsequent matching reads return that failure without retrying.

Executed with persistent mocked `CUDA out of memory`: **four calls, `failed=True`, cache write attempted**.

A temporary resource shortage can therefore become a persistent exclusion from the valid-design denominator.

**Fix:** leave transient failures retryable, or explicitly distinguish resumable failures from completed reference evaluations.

## 3. Should-Fix

**1. Feasibility depends on unrelated rows and does not check the design box.**

Source: [common.py:368](src_v8/common.py:368).

The tolerance uses the maximum magnitude across the entire batch. Executed D examples:

- `P=500`, `L=450.0004`: accepted despite a strict \(0.0004\) nm violation.
- `P=500`, `L=451`: rejected alone, accepted when another row has `P=1e9`.

The latter row is itself accepted because bounds are not checked. For ordinary bounded geometries, the tolerance admits only a very small excess; **I found zero strict violations admitted solely by this tolerance among the 360 canonical stored geometries**. There is no evidence here that this tolerance changes the headline.

Use per-row, constraint-specific tolerances and explicitly separate geometric constraints from complete domain validity.

**2. The preflight gate can miss NaNs and does not gate its advertised regression metric.**

Sources: [selfcheck.py:33](src_v8/selfcheck.py:33), [selfcheck.py:62](src_v8/selfcheck.py:62), [check_phys_D11.py:42](scripts/check_phys_D11.py:42).

`worst = max(worst, difference)` can swallow NaN: executed `max(0.0, NaN)` returns `0.0`. Explicit finiteness checks are missing. The pre-fine-tune MAE is printed without any regression assertion.

The regular gate compares local NumPy/Torch implementations, which can share a copied-definition error. The separate D comparison tests Torch parity at only three wavelengths and uses a random box that misses parts of D’s registered domain.

Add explicit finite checks, actual regression thresholds, and training-definition fixtures covering boundaries and symmetry cases.

**3. `--materials legacy` does not select legacy materials.**

Source: [rcwa_validate.py:30](src_v8/rcwa_validate.py:30).

Assignment happens only when `materials != "legacy"`. Thus explicit `--materials legacy` leaves an existing `jc` setting untouched. The default also conflates “use the profile setting” with a named material model.

Use a distinct default such as `None`, and apply explicit selections consistently.

## 4. Checked and OK

**Feature-port parity:** executed comparisons across all 100 wavelengths, 64 random geometries plus endpoints; pub also included symmetric C and `w=L` D cases. All tested values and summed-feature gradients were finite.

| Profile / feature set | Maximum absolute NumPy–Torch difference |
|---|---:|
| Pub A17 | \(2.38\times10^{-6}\) |
| Pub B13 | \(1.04\times10^{-6}\) |
| Pub C18 | \(1.19\times10^{-6}\) |
| Pub D11 | \(1.12\times10^{-6}\) |
| Legacy A17 | \(2.86\times10^{-6}\) |
| Legacy B13 | \(1.04\times10^{-6}\) |
| Legacy C13 | \(1.20\times10^{-6}\) |

C18’s first 13 columns matched C13 exactly. No column-order defect appeared in these executions. Independent upstream fidelity and full checkpoint/selfcheck reproduction remain unverified.

**Adaptive-order dispatch:** recomputed per-wavelength orders for all 180 pub A/B/C geometries using the imported solver callbacks. Every recomputed minimum/maximum matched its artifact.

| Structure | Wavelengths at N=9 | N=13 | N=17 |
|---|---:|---:|---:|
| A | 78 | 22 | 5,900 |
| B | 4,996 | 853 | 151 |
| C | 1,170 | 330 | 4,500 |

D dispatch passed exactly `(lam, P, L, w)`. Its actual imported callback executed successfully for the tested geometry. These checks verify dispatch and stored order extrema, not convergence of the electromagnetic solution.

**MAE implementation:** the recomputed absorption-channel MAE, including equal averaging of C’s TE/TM channels, exactly reproduced all valid canonical stored MAEs and both pretender totals.

## 5. Files read

The opening table is the complete source-read inventory: **nine assigned files, 1,322 lines, all read end to end**. Artifact checks used the canonical `rcwa_{A,B,C}{,_s123,_s777}_v8.npz` files in both result directories and their embedded JSON metadata. No excluded review documents were read.
