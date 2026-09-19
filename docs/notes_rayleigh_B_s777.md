# Author note — the sharp RCWA feature near 710 nm in the worst-case pretender

**Not for the manuscript without an author decision** (T21 저자 결정/검토). This note records the
arithmetic behind a possible physical attribution of the narrow feature that the reference solver
produces, and the surrogate does not, in Figure 6a.

## The design

Structure B, seed 777, target 15 (dataset index 118) — the worst pretender in the whole audit:
self-reported MAE 0.51 %, reference-solver MAE 33.6 %.

| parameter | committed g(u*) | true geometry |
|---|---|---|
| P | 458.14 nm | 455.95 nm |
| R_out | 321.20 nm | 132.19 nm |
| R_in | 300.00 nm (design-box ceiling) | 122.19 nm |
| R_disk | 54.47 nm | 62.15 nm |
| t_Cr | 59.47 nm | 71.70 nm |
| d_SiO2 | 166.33 nm | 141.87 nm |
| θ, φ | 41.46°, 25.03° | 5.61°, 40.88° |

`2 R_out / P = 1.40`: the committed ring is wider than the period, so it overlaps its own periodic
images (Figure 6a inset). The generator never produces such a sample — the rule is
`R_out ≤ 0.45 P` — and `R_in` sits exactly on the box ceiling.

## Rayleigh–Wood arithmetic

A diffraction order **G** = (2π/P)(m, n) becomes propagating in a medium of index n_med when

    |k_∥ + G| = n_med k₀ ,   k_∥ = k₀ sinθ (cos φ, sin φ) ,  k₀ = 2π/λ ,

i.e. with g = λ/P,  (sinθ cos φ + m g)² + (sinθ sin φ + n g)² = n_med².

Solved for the committed geometry (θ = 41.46°, φ = 25.03°, P = 458.14 nm; n_SiO2 from the
vendored dispersion table, n_SiO2(700 nm) = 1.4553; the solver's input layer is air, eps = 1,
and the layer beneath the pattern is the SiO2 spacer):

| order (m, n) | air side (n = 1) | SiO2 side |
|---|---|---|
| (−1, 0) | **714.6 nm** | 927.3 nm |
| (0, −1) | 494.9 nm | **735.4 nm** |
| (−1, −1) | 517.1 nm | 667.6 nm |
| (1, 0) | 165.0 nm | 387.1 nm |
| (0, 1) | 238.2 nm | 483.2 nm |

The two candidates that bracket the observed feature are the air-side (−1, 0) anomaly at
**714.6 nm** and the SiO2-side (0, −1) anomaly at **735.4 nm**. The RCWA spectrum has its sharp
dip at ≈ 710 nm, immediately below the first of these.

## What this does and does not establish

* It is **consistent** with a Rayleigh–Wood anomaly opening at the first non-specular order: the
  wavelength matches to within the 14 nm sampling of the 100-point grid.
* It is **not** a proof. The alternative reading is a ring–disk Fano resonance of the overlapping
  ring, which the surrogate has no training support for either. Distinguishing the two would need
  a wavelength-refined sweep and an order-by-order power decomposition, neither of which is in the
  pre-specified protocol.
* Either way the point that the paper makes is the same and does not depend on the attribution:
  the committed geometry lies outside the generator's feasible region, the surrogate has never seen
  anything like it, and it returns a smooth spectrum that hides a sharp physical feature.

**Author decision:** whether any of this enters §3.4/§4.1 or the Figure 6 caption. The default is
to say only "a sharp spectral feature near 710 nm is present in the reference solver and absent from
the surrogate", with no mechanism named.
