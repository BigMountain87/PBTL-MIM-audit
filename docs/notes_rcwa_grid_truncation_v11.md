# Reference-solver fidelity: the 64×64 raster and the Fourier order (2026-09-17 to 09-19)

Record of a question chain during the T76 wait: what "N = 17" does and does not certify, and
whether the 64 × 64 real-space raster is a second, undisclosed fidelity limit. **Status:
measured on 31 designs (all 18 within 1 pp of τ), 0 verdict changes; reviewed by codex
(2026-09-19, `docs/followups/codex_review_grid_truncation_T78.md`), whose corrections are
applied below.** §6 corrects three claims an
adversarial review (18 agents, 2026-09-17) found wrong or overstated in the first draft of
this note; §7 reports the 31-design spectral measurement that answers the open question
the first draft left (§6 of the original, now closed). The manuscript changes made on the
strength of this note are listed in §8; nothing else in either paper was changed.

## 1. Trigger

T76 (control seeds 123/777) was spending ~1 h per structure-A design because every design
in those runs solved at `N = 17-17` (`logs_pub/rcwa_A_s123_m0_pub.log`,
`rcwa_A_s777_m0_pub.log`). The author asked whether that means the solve converged.

## 2. What the adaptive order is (corrected: the rule differs by structure)

A fixed rule per structure, evaluated once per wavelength, not a per-solve convergence
test:

- **A** (`rcwa_struct_a.py:49-62`) and **C** (`rcwa_struct_c.py:49-80`): `hard = (aspect
  ratio ≥ 2) or (smallest metal/gap feature < 150 nm)` → N = 17; else N = 13 for λ <
  700 nm, N = 9 otherwise. (A's smallest-feature term includes the bottom patch W2 and
  P − W2; C's does not have a bottom patch.)
- **B** (`rcwa_struct_b.py:46-61`): driven by **P/λ**, not feature size or anisotropy —
  P/λ ≥ 1.3 → N = 17, ≥ 0.8 → N = 13, else N = 9. B's own docstring: "Convergence is
  driven by the number of diffraction channels (P/λ) and the staircased curved edges, NOT
  by anisotropy."
- **D** (`rcwa_struct_d.py:50-71`): `w < 150 nm or (P−L)/2 < 150 nm` → N = 17, with **no
  aspect-ratio test**.

The rule comes from Paper 1's revision (§3). Its *published*
form, Supplementary S16 ("RCWA Fourier-Truncation Convergence"), tabulates only N = 5 vs
N = 17 and states N = 17 is "the highest order reached before the memory ceiling … at
which a small residual truncation can persist"; where 17 was infeasible, 13 was used as
the reference. **This describes what S16 prints, not all the authors measured** — see [1], S16. Paper 3 already states the printed rule correctly (§2.1: "a heuristic, not a
per-solve convergence test"; W8: "reference-fidelity statement rather than a
convergence-certified one").

## 3. Provenance of the convergence request

S16 answered a referee's request during Paper 1's peer review, not an AI review. S16 itself
describes N = 17 as "converged" (supplementary.tex:534, 562-563), so the looseness is
internal to S16.

## 4. What the vendored solver actually looks like (corrected, see §6)

`_material_conv` (`torcwa/rcwa.py:1183`) builds the Toeplitz coupling matrix by indexing
the FFT of the raster at harmonic differences up to ±2N (`rcwa.py:1199-1200`); an
nx-point FFT holds only −nx/2 … nx/2−1, so entries with |m−n| beyond that wrap. On a
64-point grid this requires N ≤ 15 for a clean matrix; **N = 16 is already aliased**
(4,352 of its entries wrap, not just N = 17's). At N = 17, 1.95 % of the matrix entries
(the outermost couplings) alias — confirmed bit-for-bit (`C64[+33] ≡ C64[−31]`,
`C64[±32]` shares the folded Nyquist bin) — and contribute ≈ 1 % of the matrix's Frobenius
norm.

torcwa rasterises the A/C rectangles with a **soft** sigmoid level set,
`rcwa_geo.edge_sharpness = 100` (`geometry.py:156`, the class API the pipeline and this
note's script both call — **not** 1000, which is an unused instance class's default,
`geometry.py:10`). At sharpness 100 the 10–90 % edge transition spans ≈ 0.022·W, on the
order of one pixel — a smoothed staircase, not a hard threshold: for a 120 nm feature on a
9.4 nm pixel the effective width tracks the nominal within a few nm; for a ~400 nm feature
it tracks within ≈ 1 nm per nm of nominal change. **B rasterises with a genuine boolean
mask** (`(r ≥ R_in) & (r ≤ R_out)`, `rcwa_struct_b.py:64-76`) — no smoothing — so B is the
structure where sub-pixel geometry changes can leave the raster unchanged.

The 64-vs-256 grid comparison of the raw convolution matrices (`scripts/check_torcwa_grid_alias.py`)
showed 13.6 % (N=17) / 12.9 % (N=15) Frobenius difference, and a 3–16 % row-by-row pattern
at low harmonic differences. **Most of that is not raster error**: torcwa samples at pixel
*centres*, `(n+0.5)·P/nx` (`geometry.py:173-174`), so the 64- and 256-point FFTs differ by
a rigid sub-pixel translation phase, a similarity transform that leaves the eigenvalues
(and hence R, T and A) unchanged to 1.7 × 10⁻¹³ — spatial field phases are not invariant,
but no audited quantity is. Removing that phase drops the
low-harmonic rows to 3–4 % and the Frobenius difference to 7.6 % (N=17) / 7.0 % (N=15) —
essentially unchanged from N=15 to N=17, i.e. genuinely raster-driven, not an aliasing
artefact, but roughly half the size first reported.

## 5. Zero-cost census (all 230 committed designs, 2026-09-18)

Before spending compute, every committed design (A/B/C seed 42/123/777 + D seed 42, 230
total) was ranked by three proxies for raster sensitivity: smallest feature in pixels,
raster fill-fraction change (64 vs 256), and incidence angle. Findings:

- 18 of 230 surrogate-claimed designs sit within 1 pp of τ = 5 %, 24 more within 1–2 pp
  (the populations a solver perturbation of the sizes later measured, ≤ 0.8 pp mean,
  could plausibly flip; the first was re-solved in full, the second was not).
- The largest predicted raster effect is **B, seed 42, index 16**: ring width 0.7 px,
  fill-fraction change 15.3 %, θ = 60°.
- Several A designs have Wx > P (patch wider than the period) — a parameter-bounds
  artefact (§3.8's scope), not a raster question; excluded from the ranking.

This census, not a uniform sample, chose which designs to actually re-solve (§7).

## 6. Corrections to the first draft of this note (adversarial review, 18 agents, xhigh, 2026-09-17)

The claims below were checked against the sources by three independent lenses each
(numerics / referee / author-defence) before any of the §7 measurements existed. All were
about the *matrix-level* evidence; none change the conclusion that the audit's headline
claims survive (§8), but three were wrong or materially overstated:

1. **`edge_sharpness`** — the first draft said 1000 (hard threshold, −7.7 % fill on the
   worked example). Corrected in §4: it is 100 (soft threshold, ≈ −3.6 % fill on the same
   example). The mechanism ("raster, not aliasing, dominates") survives; the size of the
   effect was overstated by roughly 2×.
2. **The 13.6 %/12.9 % Frobenius figures and the 5/8/16 % table rows** were inflated by a
   physically inert translation phase (§4). Corrected values: 7.6 %/7.0 %, 3/3/4 %.
3. **"The adaptive order" treated as one rule.** It is four rules (§2); B's is unrelated to
   feature size, D has no aspect-ratio test. A claim scoped to A's rule does not
   generalise to B or D.

Two further, smaller corrections: the aliasing wrap is stated for both signs of the
harmonic difference (§4), and N = 16, not only N = 17, is already aliased on this grid.

One objection the review raised was **not** a numerics correction and is worth recording
rather than resolving by assertion: *common-mode error between training labels and the
oracle does not, by itself, prove no raster-induced pretenders*, because a pretender
compares the oracle to the **surrogate** (a smooth regressor), not to the labels. This is
a real gap in the argument as originally stated. §7 closes it empirically rather than
logically: the pretender verdict was re-checked on a fine raster directly, on designs
chosen for exactly this exposure, rather than inferred from the common-mode argument.

## 7. Measurement: does the raster move the pretender verdict? (2026-09-17 to 09-19)

**Method.** Re-solve a committed design with the pinned solver (same module, order rule,
materials, precision) at grid 64 (control) and grid 256, and compare the RCWA MAE against
target at each grid. `scripts/grid_delta_resolve.py`; queue in
`scripts/grid_delta_queue.txt`; driver `scripts/run_grid_delta_server.sh` (server, waits
for T76's completion marker, then runs on GPU, 2 workers).

**Selection (31 of 230 committed designs, not a uniform sample — see §5):**
- 7 within 1 pp of τ = 5 % (the only population a raster perturbation could flip)
- 3 with the largest predicted raster sensitivity from the §5 census (C, seed-independent
  canonical-feature designs)
- 1 (A s123 i3) run first as a pilot, before the census existed
- 10 drawn at random (seed 0), for an unselected baseline
- 10 more from an earlier overnight batch on the Mac (8 B chosen for small ring width /
  infeasible ring / near-τ, plus 2 C near-τ, before the full census), each at 20 sampled
  wavelengths rather than the full 100

(7 + 3 + 1 + 10 + 10 = 31. All 18 designs within 1 pp of τ are among them — 7 at 100
wavelengths, 10 at 20, the pilot at 10. Structure D contributes 7 designs, one near τ and
six random. Designs 1–2 pp from τ — 24 of them — were **not** re-solved, nor was the
legacy arm; the result bounds the sampled designs, not the population.)

**Result — 0 of 31 pretender verdicts changed.**

| batch | n | wavelengths | mean \|ΔA\| range (pp) | max \|ΔA\| range (pp) | flips |
|---|---:|---|---|---|---:|
| Mac, overnight, structure B/C | 10 | 20 | 0.04 – 0.82 | 0.10 – 2.12 | 0 |
| Mac, pilot (A s123 i3, N=17)† | 1 | 10 | 0.60 | 0.85 | 0 |
| Server, near-τ / census / random | 20 | 100 (full) | 0.01 – 0.58 | 0.03 – 2.17 | 0 |
| **all** | **31** | | **0.01 – 0.82** | **0.03 – 2.17** | **0** |

† The pilot ran from a scratch script whose output file was not retained (session scratch
directory); its numbers — grid 64: 4.44 %, grid 256: 3.89 %, mean |ΔA| 0.60 pp, max 0.85 pp,
signed mean +0.52 pp over 10 wavelengths, control vs archived GPU solve 0.01 pp — are
recorded here from the run's printed output only. The other 30 designs' raw solves are in
`results_pub/grid_delta/*.npz`.

Per-design detail (structure, seed, index, archived grid-64 MAE → grid-256 MAE, both
against target; `*` = 20-wavelength subsample):

```
A s42  i1   4.38 -> 4.36    A s42  i9   2.04 -> 2.11    A s42  i12 10.13 -> 10.38 (PRET)
A s123 i1   4.70 -> 4.68    A s123 i2   4.43 -> 4.62    A s123 i3*  4.44 -> 3.89 (pilot, N=17)
A s777 i18  5.32 -> 5.35 (PRET)
B s42  i0*  4.71 -> 4.67    B s42  i4*  4.82 -> 5.08->5.04 (PRET, 20-wl only)
B s42  i6*  4.13 -> 4.17    B s42  i16* 4.37 -> 4.85    B s42  i13 1.25 -> 1.35
B s123 i6*  4.12 -> 3.93    B s123 i17* 5.20 -> 5.24 (PRET)
B s777 i4*  5.27 -> 5.24 (PRET, 20-wl only)             B s777 i6*  4.06 -> 4.09
C s42  i3   0.95 -> 1.03    C s42  i9   1.27 -> 1.53    C s42  i17* 6.80 -> 6.77 (PRET)
C s123 i7   4.37 -> 4.17    C s123 i9   1.70 -> 1.58    C s123 i19* 4.09 -> 4.26
C s777 i9   1.86 -> 1.65    C s777 i19  5.24 -> 5.41 (PRET)
D s42  i0   3.24 -> 3.46    D s42  i1   0.71 -> 0.65    D s42  i19  5.29 -> 5.21 (PRET)
D s42  i25  2.92 -> 2.93    D s42  i30  0.33 -> 0.27    D s42  i37  0.93 -> 0.99
D s42  i48  1.30 -> 1.24
```

*(`B s42 i4` and `B s777 i4` show PRETENDER→PRETENDER within the 20-wavelength subsample
used for both grids — the "5.08→5.04" / "5.27→5.24" pair; their **archived, 100-wavelength**
MAE (4.82 % / 4.71 %, pass) differs from the 20-wavelength subsample value because of the
subsample, not the grid, and is not a flip.)*

**Reading it.** No sign of a bias: some designs move up, some down, by similar amounts.
The two largest single-wavelength excursions (C s123 i9: 2.17 pp at 100 wavelengths;
C s42 i17: 2.12 pp on its 20-wavelength subsample) occur on designs whose mean |ΔA| over
the solved wavelengths is 0.19 and 0.82 pp and whose MAE-vs-target moves by 0.12 and
0.03 pp — a resonance-scale local effect that averages out. The census's top-ranked design (B s42 i16, ring 0.7 px, 15 %
fill change) gave the *second*-largest mean effect (0.68 pp) but no flip. This is now an
empirical answer, not an inference from common-mode reasoning: **on the 31 designs tested,
which include every design within 1 pp of τ, the raster does not change the pretender
verdict at τ = 5 %.** It is not a population bound (see the coverage note above).

## 7b. A second solver-fidelity item the review surfaced: admissibility of oracle spectra

Codex's review of the T78 commit (2026-09-19) noticed that the grid-256 solve of C s42 i17
has A_TM = −0.024 at 909 nm (grid 64: −0.002), below the −0.005 floor Paper 1 uses to admit
a generated sample (its `reliable` mask; the grazing-order artefact at oblique incidence,
[1] S16 and `CONVERGENCE_C.md`). The audit's oracle checks finiteness only
(`oracle.py`, simulate_cached), so the same artefact can enter an archived committed-design
spectrum. Census (`scripts/oracle_admissibility_v11.py`, `results_pub/oracle_admissibility_v11.json`):
**2 of 289 archived spectra** carry one such point — C s123 i19 TE (−0.021 at 683 nm; MAE
4.55 → 4.54 clipped / 4.30 dropped) and D s42 i34 (−0.288 at 499 nm; 3.35 → 3.06 / 2.22) —
and **no verdict changes** under either treatment. Disclosed in W8 with the JSON-backed
count; the oracle is left as it is (its archived numbers must not move), and the census is
the guard.

## 8. What changed as a result

- **Paper 3** (`paper/manuscript_v11.md`): §2.1 now names B's different order rule and
  states the raster is alias-free only to N = 15; W8 now discloses the raster fidelity
  limit with the measured numbers (§7) and points readers to Section 4.7 from §3.5. Three
  instances of "true error" (implying the oracle is ground truth) → "reference-solver
  error"; "the physical oracle itself" → "the reference solver itself"; §3.5/§3.11 now say
  "the order alone" / "on the same raster" where they previously said "truncation alone",
  to scope those sentences to the one variable they actually isolate.
  After codex's review of that commit (T79): §2.1 no longer attributes an aspect-ratio rule
  to Structure D; W8 says the adaptive order "reduces the discrepancy" rather than "removes
  the error", states the wavelength coverage of the 31 re-solves (20 full, 11 subsampled),
  says the result bounds the sampled designs, not the population, and adds the
  admissibility census (§7b); "physics-violating basins" → "basins rejected by the
  reference solver"; "the only reliable detector … is the reference solver itself" →
  "within this audit, only a call to the reference solver itself separates pretenders from
  confirmed designs". `scripts/check_manuscript_v10.py` (now also checking the
  admissibility count) and the LaTeX build both pass.
- **`src_v8/oracle.py`**: `configure_solver` now computes and reports (not enforces)
  `grid_sufficient = nx ≥ 4·N_max + 2`; logs a one-line NOTE when it is false (true for
  every adaptive-profile pub run: 64 < 70). Kept out of the cache fingerprint on purpose —
  adding a key there would invalidate every archived solve. Not pushed to the server
  (T76 had already finished when this was written; no more solves are pinned against it
  yet).


## 9. Compute log

- T76 (control seeds 123/777, 12-run oracle stage): started 2026-09-16 13:22, finished
  2026-09-19 08:52 (`logs_pub/gate_seeds_DONE.txt`). Slower than the ~60 h estimate because
  every Structure-A design in these runs hit N = 17 (~1 h each); no errors.
- Grid-delta measurement: 10 designs overnight on the Mac (2026-09-17 22:22 –
  2026-09-18 01:21, CPU); 1 pilot design on the Mac (2026-09-17, CPU); 20 designs on the
  server, queued behind T76, ran 2026-09-19 08:53 – 15:06 (GPU, 2 workers). All at the
  pinned solver settings; zero failures across 31 designs.
