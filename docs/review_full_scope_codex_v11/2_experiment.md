# Full-scope codex review — slice 2_experiment (2026-09-12)

## 1. Verdict

**The pub headline reproduces, but the experiment orchestration does not reliably enforce experiment identity or successful completion. The assigned protocol also describes a different experimental configuration from the pub runs.**

Recomputing MAEs from archived spectra gives **179 valid, 179 surrogate-claimed, 163 oracle-passing, and 16 pretenders**. I found no arithmetic discrepancy in the checked core reliability metrics.

The strongest defects concern restart-zero reconstruction, stale-artifact acceptance, incomplete identity checks, and drivers reporting completion after failures. These are distinguishable from evidence that existing core counts are wrong.

One requested proof remains unavailable within the assigned scope: `common.py` was not assigned, and no authorization arrived for the narrowly requested helper inspection. Consequently, I can verify existing 20-target selections, but cannot certify `pick_targets`’ 20→50 behavior or the internal fallback predicate. No earlier review documents were read.

## 2. Must-Fix

**1. Restart-zero validation reconstructs the wrong geometries for feasible runs.**

`src_v8/restart0_validate.py:50–51` always applies:

```python
x0 = lo + u0_end * (hi - lo)
```

Feasible inverse runs instead use the nested mapping at `src_v8/inverse.py:80–100`. The validator neither checks `inv["feasible"]` nor applies that mapping.

I reconstructed both mappings from the saved feasible endpoints:

| Structure | Restart-zero geometries changed | Maximum coordinate discrepancy |
|---|---:|---:|
| A | 17/20 | 253.099 nm |
| B | 19/20 | 196.822 nm |
| C | 14/20 | 289.300 nm |

Thus invoking this validator on `_feas` artifacts would label different geometries as the single-start control. The current gate driver does not invoke feasible restart-zero validation, so this is a demonstrated execution-path defect, not a claim that existing untagged restart-zero results are corrupted.

**Fix:** archive physical restart endpoints or share the inverse parametrization with the validator.

**2. The fallback consumers do not preserve surrogate identity.**

Every `input_sfx` consumer in the assigned Python files is listed below. These paths follow the fallback contract explicitly documented in their source; the helper’s implementation itself remains outside scope.

| Consumer | Remaining mislabelling path | Affected output |
|---|---|---|
| `inverse.py:24–34,195–196` | Missing tagged fine-tune inputs can resolve to the base surrogate while output uses the requested tag. A direct `_m0` invocation can therefore produce a treatment result labelled as a scratch control. | `inverse_*_m0_v8.npz` |
| `random_baseline.py:32–38,143–156` | The same fallback can make tagged random-search controls use the wrong surrogate. No scratch-identity check exists here. | Tagged random-baseline NPZ and JSON |
| `restart0_validate.py:28–34,104–107` | Tagged inverse endpoints are combined with a surrogate resolved anew. Missing or subsequently changed tagged fine-tune artifacts can change the surrogate used to classify those endpoints. | Tagged restart-zero NPZ and JSON |
| `reliability.py:143–147,154–173,245–257` | Archived inverse MAEs are combined with the currently resolved fine-tune test MAE and checkpoint. The resulting threshold and T4 measurement can describe a different surrogate from the one that produced the designs. | Tagged reliability JSON and CSV |

For the last path, `tier1b.flag_threshold_pct` depends on the newly loaded fine-tune log, while T4 reloads its checkpoint at `reliability.py:97`. Neither is bound to the originating inverse run by a checkpoint fingerprint.

Two distinctions matter:

- Reusing the base surrogate for `_feas` is intentional. Fine-tune fallback does **not itself** disable `--feasible`.
- A `_feas` label alone does not enable constraints: both optimization programs default to `feasible=False` (`inverse.py:37`; `random_baseline.py:30`). Gate stage 3 skips an existing feasible-random JSON without checking its `feasible` field (`run_pub_gate.sh:102–104`).

**Fix:** record and validate the actual input checkpoint, statistics, fine-tune log, and inverse artifact identities. Require explicit base-surrogate reuse for conditions that intend it.

**3. `check_m0` and `check_feas` fail loudly on their narrow checks, but are insufficient to validate resumed experiments.**

`run_pub_gate.sh:26–39` checks only that the scratch log exists and reports `init="scratch"` and no checkpoint. It does not verify the checkpoint/statistics bundle, seed, profile, learning rate, or downstream provenance.

I executed the exact embedded guard with mocked inputs:

- Missing log: rejected.
- Pretrained log: rejected.
- Scratch log without any checkpoint/statistics verification: accepted.
- Scratch log declaring seed 777, learning rate 0.2, and legacy profile: accepted.

The guard runs before inverse generation, which is appropriate. However, existing inverse, oracle, and reliability files are then accepted solely by existence (`:69`, `:74`, `:78`). A newly corrected scratch log cannot certify older downstream files.

Likewise, `check_feas` (`:42–53`) checks only one Boolean. Executing its exact body accepted an input containing only `{"feasible": True}`—without any geometry. Its placement before oracle validation is appropriate, but existing oracle/reliability artifacts remain unchecked (`:91`, `:95`). No equivalent guard protects feasible random-search outputs.

The `die` handlers do exit nonzero. **Loud rejection is implemented; sufficient identity verification is not.**

**4. Drivers can announce successful completion after failed work.**

Concrete paths:

- Core catches several failures with `|| echo`, continues, and unconditionally writes `core_DONE.txt`: `run_pub_core.sh:23,30,43,67,69`.
- Core control and assembly failures lack reliable propagation: `:48,55–61`.
- Gate catches inverse, oracle, reliability, and random-control failures, then unconditionally writes `gate_DONE.txt`: `run_pub_gate.sh:70,75,79,92,96,104,112`.
- Extension’s cross-solver background jobs have no failure aggregation; it prints completion and writes `extend_DONE.txt`: `run_pub_extend.sh:54–67`.

I executed the shell pattern used here with failed foreground/background commands: it printed `FAIL inverse`, then `ALL STAGES DONE`, and exited **0**. A bare `wait` is not a per-worker success audit.

This also breaks sequencing: gate waits for the core marker (`gate:57`), and extension waits for the gate marker (`extend:17`). A failed predecessor can release its successor.

**Fix:** collect worker exit statuses, validate required outputs, and create completion markers only after successful validation.

**5. Existence-based resumption accepts partial, stale, or incorrectly sized artifacts.**

Writers use direct final-path writes:

- Fine-tune checkpoint → statistics → JSON: `finetune.py:132–159`.
- Inverse NPZ: `inverse.py:195–196`.
- Random NPZ → JSON: `random_baseline.py:152–156`.
- Restart-zero NPZ → JSON: `restart0_validate.py:104–107`.
- Reliability JSON → CSV: `reliability.py:245–257`.

An interrupted write can leave a truncated final file or an inconsistent bundle. The drivers’ `-f` checks then skip it. Reliability’s JSON can also exist while its CSV is missing.

The D extension explicitly requests 50 targets (`run_pub_extend.sh:22`) but skips any existing `inverse_D_v8.npz` without checking its target count (`:33`). A pre-existing 20-target artifact would satisfy that condition.

Core and gate also do not clear inherited target-count settings; none of the three drivers clears an inherited run tag before untagged stages.

**Fix:** validate dimensions, configuration, dependencies, and readable content; publish artifacts atomically and mark complete bundles explicitly.

**6. The assigned protocol is not an accurate specification of the pub experiment.**

This is an observed documentation/configuration mismatch, not evidence that the pub configuration is intrinsically invalid.

| Quantity | Assigned protocol | Recomputed/read pub artifact |
|---|---|---|
| A wavelength grid | 380–780 nm | 400–1800 nm |
| Good samples A/B/C | 479 / 461 / 400 | 499 / 498 / 487 |
| C training samples | 300 | 350 |
| Main oracle order | Fixed `[5,5]` | Adaptive orders; seed-42 archives contain orders 9–17 |

Protocol citations: `docs/v8_protocol.md:46–48,62–72,126–130`.

Artifact evidence: `results_pub/stats_A_v8.npz:wavelengths`; `finetune_{A,B,C}_v8.json:n_good,n_train`; `rcwa_{A,B,C}_v8.npz:adaptive,orders_min,orders_max`.

The assigned document includes the training-seed amendment but no pub-arm configuration amendment. Calling these pub runs executions of this exact frozen specification is therefore unsupported by the assigned files.

## 3. Should-Fix

- **Enforce inverse/oracle alignment in reliability.** `reliability.py:144–152` combines arrays positionally without checking geometry, wavelengths, target spectra, or surrogate identity. I verified geometry alignment in the 24 available core/control pairs, but the program itself would accept mismatched same-length inputs.

- **Reject nonfinite metrics explicitly.** Reliability uses only `~failed` for validity (`:152`) and defines pretenders through the complement of oracle pass (`:156–158`). An unflagged NaN oracle MAE therefore becomes a pretender when its surrogate passes, despite not satisfying the protocol’s explicit `RCWA MAE > 5%` definition. Random/restart-zero validation excludes NaN but not infinity (`random_baseline.py:139`; `restart0_validate.py:91`).

- **Correct the “exact p” statement.** `reliability.py:5` says “exact p”; `:40` calls SciPy Spearman, and `docs/v8_protocol.md:154–156` correctly specifies asymptotic inference. The implementation and protocol agree; the module description does not.

- **Supply the specified within-cluster spread.** Protocol `:166–168` requests mode count and within-cluster spread alongside count intervals. Reliability emits mode count, but no within-cluster spread (`reliability.py:228–236`). Tier-1C restart dispersion measures a different object.

- **Fix core control concurrency accounting.** One capacity check launches two workers (`run_pub_core.sh:54–56`). With `RPAR=2`, one existing worker permits two additional launches, exceeding the stated limit.

- **Correct generalized-run reporting.** Random-baseline console summaries hard-code `/20` (`random_baseline.py:157–158`) although allocation uses `N_TARGETS`. Its “equal surrogate-call budget” description also omits inverse’s two additional final prediction evaluations (`inverse.py:155,158`): 6,416 restart-spectrum evaluations versus 6,400 random candidates, excluding backward-pass cost.

## 4. Checked and OK

**Executed artifact checks—not agreement with stored summaries:**

| Arm | Valid | Surrogate passes | Oracle passes | Pretenders |
|---|---:|---:|---:|---:|
| Pub, nine core runs | 179 | 179 | 163 | 16 |
| Legacy, nine core runs | 179 | 179 | 84 | 95 |
| Pub scratch controls | 60 | 60 | 56 | 4 |
| Pub feasible controls | 60 | 60 | 57 | 3 |

Across **24 inverse/oracle/reliability bundles**, the following reproduced without discrepancies at the numerical comparison tolerance:

- MAEs recomputed from saved surrogate, target, and oracle spectra.
- Tier 1A counts and Wilson intervals.
- Tier 1B threshold, flag counts/intervals, maximum gap, and maximum amplification.
- Tier 1C final-loss standard deviation, endpoint spread, and committed-restart tail RSD.
- T1–T3 classifications, T2/T3 null calculations, and mode counts.
- Spearman correlations, asymptotic p-values, and approximate intervals.
- Committed restart equals `argmin(final_losses)`.
- Inverse and oracle committed geometries agree.

**Seeds and existing targets:** reconstruction of the fixed seed-42 split, seed-dependent training subset, and `rng(42).choice(test_idx,20,replace=False)` matched the checked artifacts. Original-dataset target mappings also matched. The assigned inverse code resets restart RNG and Adam separately for every target (`inverse.py:133–141`), so it introduces no optimizer-state dependence on earlier targets.

**Controls:** all 15 available random/restart-zero NPZ/JSON pairs reproduced their saved valid/pass/pretender counts. Pub scratch logs report scratch initialization, no loaded checkpoint, seed 42, and learning rate 0.001; feasible inverse artifacts report `feasible=True`.

**Limits:** T4 flags reproduce from archived perturbation-MAE values, but I did not independently regenerate those spectra. I did not rerun training or the reference solver. There is no D inverse artifact in the inspected outputs—only fine-tune/statistics artifacts—so D’s 50-target execution is not yet verifiable. Existing 20-target agreement does **not** prove that `pick_targets` preserves the prefix when extended to 50.

## 5. Files read

The opening table records complete coverage of all **nine assigned files, 1,406 lines**.

Artifact reads were limited to relevant fine-tune/statistics, inverse, oracle, reliability, random-baseline, and restart-zero outputs in `results_pub/` and `results_v8/`. No earlier review documents or aggregate review/synthesis reports supplied findings.
