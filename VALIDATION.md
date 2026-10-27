# Validation

Validated on 2026-10-27 with Python 3.12.2, PyTorch 2.5.1, NumPy 2.1.3, SciPy 1.14.1, and tqdm 4.66.5. Reference verification used CUDA; single-posture generation was also exercised on CPU.

All released reference designs were evaluated with the released surrogate weights and target postures:

| Task | Targets | Designs | Saved-sample success rate | Maximum absolute saved/recomputed MSE difference |
|---|---:|---:|---:|---:|
| Single posture | 30 | 3,000 | 96.8667% | 0 |
| Three postures | 15 | 1,500 | 93.6000% | 6.52e-9 |
| Trajectory | 7 | 700 | 88.5714% | 2.98e-8 |

All 5,200 geometries passed the released feasibility check at tolerance 1e-5. These are surrogate-based results. No FEA validation was performed or included.

The reference distributions are historical saved samples. No claim is made that new random runs produce exactly these values or statistically equivalent distributions.

The checkpoint loaders, reference schemas, target counts, and comparison command were checked. The inference tensors match the source checkpoints.

Generation smoke checks use one design on the first target with one DE generation and the complete 500 diffusion steps. They validate execution only, not benchmark performance. Full-budget optimization has not been rerun as part of packaging.

All three generation smoke checks (single posture on CPU; three postures and trajectory on CUDA) completed successfully. The resulting samples also passed independent MSE recomputation and geometry checks. The comparison command was checked from outside the repository directory.
