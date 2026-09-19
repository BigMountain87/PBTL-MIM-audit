## S5. Pilot-stage defects corrected before pre-registration

The earlier two/three-anchor pilots are superseded. The following defects,
found in the full audit (`docs/review_iter4_full_audit.md`) and fixed here, are
why the v8 numbers differ and why several earlier headlines are **withdrawn**:

| # | v4a/v7 defect | v8 fix |
|---|---|---|
| 1 | B/C inverse run on 380–780 nm against 400–1800 nm data | per-structure native grids, pinned in artifacts |
| 2 | Adam on raw nm — optimizer frozen within ±22 nm | normalized variables + multi-start (§2.2) |
| 3 | divergent RCWA rows leaked into train/test/targets | Paper 1 filter (§2.1) |
| 4 | permuted pretrain input columns; phys stats recomputed on RCWA; C-bounds mismatch | faithful strict loading (§2.1) |
| 5 | T4 reported but never computed | implemented and measured (§2.4) |
| 6 | ρ(Δ, Surr) "confidence inversion" headline (coupling artifact) | primary stat = ρ(RCWA, Surr) (§2.5) |
| 7 | A used 6 curated targets vs B's 20 random | identical N = 20 random targets for all structures |
| 8 | single deterministic init; `manual_seed` no-op | R = 8 multi-start, archived endpoints |

Consequently, the earlier **confidence inversion**, the **7.76× amplification**,
and the named pretenders **B-11/B-16** are products of the frozen optimizer and
the grid mismatch and are **retracted**. No claim in this manuscript depends on
them.
