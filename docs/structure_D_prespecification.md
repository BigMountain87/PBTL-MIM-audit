# Structure D — pre-specified prediction, written before its inverse audit ran

**Written 2026-09-10, before any Structure-D inverse design or oracle call.**
At this moment `results_pub/` contains no D artifact of any kind; the only D quantities
that exist are the RCWA dataset (generated 2026-07-08) and the fidelity numbers below.

## What D is

The cross-patch fourth structure built as a contingency for the companion paper's
re-review (`<PBTL-MIM working copy>/structure_D/README.md`, dated 2026-07-03): a plus-shaped
Cr patch, 4-fold symmetric and therefore polarization-independent, 6 parameters
(P, L, w, t_Cr, d_SiO2, theta). It was chosen as a genuine out-of-sample test of the
transferability diagnostic — a different geometry class from the dual cavity (A),
ring-disk (B) and rectangle (C), with more re-entrant corners than a rectangle.

Its 500 samples were generated on 2026-07-08 with the published pipeline's settings
(`structD-v1-jc+adaptive+c64+grazingfix`, 400-1800 nm, Cr, per-wavelength adaptive
Fourier order, complex64, Johnson-Christy), the same configuration as the pub arm.
494 of 500 are reliable.

## The fidelity prediction that was written first, and its outcome

The 2026-07-03 README predicted, before the data existed: *"Predicted intermediate
fidelity (between A and C) … This tests the diagnostic's interpolation — does an
intermediate operating-band MAE give an intermediate transfer benefit?"*

Running `structure_D/fidelity_D.py`, which reuses `compute_fidelity_redesign.py`'s
estimator — the one that produced the companion's printed Table 5 — gives

| structure | median *r* | median operating-band MAE |
|---|---|---|
| B | +0.962 | 8.93 % |
| **D** | **+0.905** | **10.51 %** |
| A | +0.83 | 7.94 % |
| C | +0.647 | 16.94 % |

The same run reproduced the printed values for A, B and C (printed: A 0.83/7.9,
B 0.96/8.9, C 0.65/16.9), which is an independent check that this is the published
estimator. **D's operating-band MAE falls between A and C, as predicted.**

## What we predict now, before running D's inverse audit

D cannot enter the H1a–c tests: it did not exist when the protocol was frozen on
2026-06-10, so no *r* convention pre-specified an ordering that includes it. That makes
it available for something cleaner — a held-out test of whether the diagnostic predicts
*inverse* reliability at a structure nobody tuned anything on.

Our audit has so far produced these pub-arm pretender rates (`results_pub/pooled_v8.json`):
B 9/60 (15.0 %), A 2/59 (3.4 %), C 5/60 (8.3 %); pooled 16/179 (8.9 %).

**Prediction, recorded before D's inverse run:**

1. **Primary.** If the diagnostic carries information about inverse reliability, D's
   pretender rate should sit between B's and C's — its *r* (0.905) is between B and A,
   and its operating-band MAE (10.51 %) is between B and C. Concretely we predict
   **D's pretender rate falls in 5–15 %**, i.e. between 3 and 8 of 50 committed designs.
2. **Secondary.** D's surrogate-claimed success rate is 100 %, as on every structure so
   far, so the threshold self-check remains uninformative there too.
3. **What would falsify the diagnostic's usefulness here.** A D pretender rate below A's
   (< 3.4 %) or above C's (> 16.9 %) would place an intermediate-fidelity structure
   outside the range its neighbours span, which no reading of the diagnostic predicts.

These are directional predictions on a single structure, not a powered test: with 50
targets a rate anywhere in 2-14 % has overlapping intervals with the prediction band.
We record them so the D result is interpretable as confirmation or not, rather than
being read after the fact.

## What will be run

`INVERSETL_N_TARGETS=50` on the full held-out split: fine-tune from
`structure_D/results/pretrained_mphys_tmm.pt` → inverse design → adaptive-order oracle
on all 50 committed geometries → reliability. Measured generation cost was 528 s per
sample, so the oracle stage is about 7.3 h.


---

## Clarification appended 2026-09-12 (file written 08:55, committed 09:01 KST)

The full-scope codex review (slice 5) found three errors in the prediction above, and a
second pass (slice 6) found that my first version of this clarification made a fourth: it
widened the primary band. The original text is left as written; this section states what
was wrong, what the governing rule is, and what was knowable when.

**Errors in the original prediction**

1. "Between 3 and 8 of 50" is wrong arithmetic: 8/50 = 16 %, outside 15 %. The counts
   consistent with 5–15 % of 50 are **3 to 7**.
2. "Above C's (> 16.9 %)" confuses two quantities: 16.9 % is C's operating-band TMM MAE,
   not its pretender rate (8.3 %). This was carried into `scripts/structure_d_v10.py` and is
   corrected there.
3. "Between B's and C's" with the observed rates is 8.3–15.0 %, narrower than the 5–15 %
   band the same sentence went on to state.

**Governing rule: the original band, as written.** The primary prediction stays
**5–15 %, i.e. 3 to 7 of 50 committed designs**, with only its arithmetic corrected. The
falsification bounds are those the original text intended — below A's pooled rate
(**3.4 %**, fewer than 2 of 50) or above the upper neighbour B's (**15.0 %**, 8 or more of
50) — with the mislabelled 16.9 % replaced by the quantity it was meant to be. A wider
union band [3.4, 15.0] % that my first clarification proposed is reported only as a
secondary reading: it would move 2 of 50 from outside the band to inside, and a band
loosened after the fact is exactly what a pre-specification exists to prevent.

**What was knowable when this was written.** The final artifact `rcwa_D_v8.npz` did not
exist. But the oracle writes each design's result to `results_pub/oracle_cache/` as it
finishes and prints its MAE to `logs_pub/rcwa_D_pub.log`, and at the time of writing 42
of the 50 designs had been solved. I did not compute a partial pretender count from
either, and the analysis script reports nothing until the final artifact exists, but that
is a statement about conduct, not something the record can prove. The correct label for
this section is therefore **an amendment made while partial results existed on disk and
were not inspected**, not a blinded pre-specification, and D's outcome should be read with
that in mind. The fine-tune and inverse artifacts for D (2026-09-11) also predate it; those
contain no oracle information.
