# Structure D — held-out test of the diagnostic

D: **4/50 pretenders = 8.0 %** [3.2, 18.8], 46 oracle-confirmed of 50 valid; median Δ +0.59 pp, max +14.57 pp.

Prediction (docs/structure_D_prespecification.md (2026-09-10), clarification of 2026-09-12 08:55 (42/50 oracle results existed, not inspected)): 5–15 %. Outcome: **inside the band**; not falsifying.

| structure | median *r* | op-band MAE | pretenders | rate [95 %] |
|---|---|---|---|---|
| B | +0.962 | 8.93 % | 9/60 | 15.0 % [8.1, 26.1] |
| D | +0.905 | 10.51 % | 4/50 | 8.0 % [3.2, 18.8] |
| A | +0.830 | 7.94 % | 2/59 | 3.4 % [0.9, 11.5] |
| C | +0.647 | 16.94 % | 5/60 | 8.3 % [3.6, 18.1] |

Feasibility (post hoc, rule L <= 0.9 P and w <= L): 20/50 committed D geometries infeasible; pretenders 4 infeasible / 0 feasible; Fisher p = 0.021.

## Four-point orderings

- **printed_r**: B > D > A > C → rates [15.0, 8.0, 3.4, 8.3]; monotone: False; Cochran–Armitage z = +0.99, p = 0.320
- **operating_band_mae**: A > B > D > C → rates [3.4, 15.0, 8.0, 8.3]; monotone: False; Cochran–Armitage z = +0.04, p = 0.968

A/B/C counts pool three training seeds (20 targets each); D is one seed over the full 50-design held-out split, and its *r* was computed by us with the printed estimator rather than taken from the companion's table.

The A/B/C intervals and the four-point trend tests are nominal: the three seeds of a structure share one target set, so the pooled counts are correlated replicates and the effective n is smaller than 60 (design effects in stats_supplement_v9.json).
