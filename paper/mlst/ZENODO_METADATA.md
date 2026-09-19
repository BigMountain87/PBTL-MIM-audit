# Zenodo / OSF deposit metadata (draft — T50)

| field | value |
|---|---|
| **Resource type** | Dataset / Software (Zenodo "Software" with a data component) |
| **Title** | Oracle-in-the-loop reliability audit of physics-transferred neural surrogates for inverse metasurface design — code, protocol and artifacts |
| **Version** | v1.0 |
| **Creators** | Choi, Sang-Bae (Independent Researcher, Busan, Republic of Korea; ORCID TODO-AUTHOR) · Choi, Je-Min (Pusan National University; ORCID TODO-AUTHOR) · Kim, Joonhyub (Pusan National University; ORCID TODO-AUTHOR) · Kang, Chang-Mo (Pusan National University; ORCID TODO-AUTHOR) |
| **Licence** | code MIT (`LICENSE`); data, checkpoints and results CC-BY-4.0 (`LICENSE-DATA`) — author decision 2026-09-19 |
| **Related identifiers** | `isIdenticalTo` — github.com/BigMountain87/PBTL-MIM-audit (release tag) · `isSupplementTo` — this manuscript (DOI assigned on acceptance, TODO-AUTHOR) · `isDerivedFrom` — 10.1016/j.photonics.2026.101617 · `isSupplementTo` — the pre-registration record, if deposited separately |
| **Keywords** | neural surrogate; inverse design; transfer learning; metasurface absorber; offline model-based optimization; reliability audit; pre-specified protocol |
| **Languages** | English |
| **Funding** | TODO-AUTHOR |

## Description (≤ 120 words)

Code, pre-specified protocol and complete artifacts for an oracle-in-the-loop reliability audit of
physics-transferred neural surrogates used as the forward block of a gradient-based inverse-design
loop. Three metal–insulator–metal absorber structures are run through one parameterized codebase at
three training seeds each; every committed geometry is re-simulated with a rigorous coupled-wave
reference solver. The deposit contains the audited (as-submitted) and printed-pipeline arms, both
vendored upstream solver trees, the TMM-pretrained checkpoints the audit fine-tunes from, per-run
artifacts, the aggregated evidence files behind every number in the paper, the figure scripts, and
SHA-256 manifests for code and inputs. An analysis-only reproduction path runs on CPU without
PyTorch.

## Upload checklist

- [ ] tarball re-staged after the printed-pipeline results land (T54/T55)
- [x] `LICENSE` and `LICENSE-DATA` committed (2026-09-19)
- [x] co-author consent recorded (preprint and release, 2026-09-19)
- [ ] ORCIDs filled in **[AUTHOR]**
- [ ] DOI returned and substituted throughout the manuscript **[AUTHOR]**
