# Compute environment (results_v8 legacy arm and results_pub printed-pipeline arm)

| item | value |
|---|---|
| host | `compute-host` (Ubuntu, 20 CPU cores, 31 GB RAM) |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB; driver 560.35.05 |
| python | `~/anaconda3/bin/python` 3.12 |
| torch | 2.5.1+cu121 (CUDA 12.1) |
| torcwa | 0.1.4.2 |
| numpy / scipy / matplotlib | 1.26.4 / 1.13.1 / 3.9.2 |
| source of pins | upstream `provenance/requirements_lock_homeserver.txt`; `requirements_v8.txt` in this repo |
| upstream code | legacy arm: commit 920b1bd (vendored `src_v8/upstream_920b1bd`); printed-pipeline arm: commit cb486b5 (vendored `src_v8/upstream_cb486b5`, `data/ref` J&C tables) |
| RCWA fidelity | legacy: 64×64 grid, fixed Fourier order [5,5], complex128 (16–39 s per geometry); pub: per-wavelength adaptive order N ∈ {9,13,17}, complex64, J&C materials (4–52 min per geometry) |
| analysis-only path (Mac) | system python3 (numpy 1.26.4, scipy 1.11.4, matplotlib 3.10) — no torch needed for `scripts/reproduce_analysis_only.sh` |

Every stage writes `env_info()` (torch/cuda/cudnn/gpu/numpy/torcwa/profile/upstream commit) into its JSON/npz output.
CUDA training is not bitwise deterministic (`torch.use_deterministic_algorithms` deliberately not enabled); GPU
stages reproduce statistically, the RCWA stage reproduces bitwise (see `docs/pinning.txt`).
