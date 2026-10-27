# DeCo-DIFFUSE

Inference and surrogate-based optimization evaluation for generative inverse design of an LCE-driven elephant-trunk soft robot.

This release contains the final Bi-LSTM surrogate and conditional diffusion architectures, inference-only model weights, constrained design generation, differential-evolution (DE) temperature optimization, benchmark targets, and saved reference samples. Training code, training datasets, plotting code, and finite-element validation are outside this release.

## Setup

Use Python 3.10 or newer. Install PyTorch for your CPU/CUDA platform, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The exact locally tested versions are recorded in `VALIDATION.md` and pinned in `requirements-tested.txt`. CUDA is recommended for full evaluation; CPU execution is supported. Run commands from this directory. Input assets are resolved relative to the code, independent of the shell's working directory. Output paths are relative to the shell's working directory.

## Verify the saved samples

Recompute surrogate predictions at the released geometries and temperatures, without rerunning generation or optimization:

```bash
python scripts/evaluate.py verify --task single
python scripts/evaluate.py verify --task three
python scripts/evaluate.py verify --task trajectory
```

Use `--limit 1` for a quick check. The report includes success rate, median and p90 maximum posture MSE, maximum difference between saved and recomputed MSE, and geometric feasibility. A maximum MSE difference below `1e-5` is a numerical sanity check, not a guarantee of bitwise agreement or identical classification for samples near the success threshold.

## Generate designs and optimize temperatures

```bash
python scripts/evaluate.py generate --task single --output results/single.json
python scripts/evaluate.py generate --task three --output results/three.json
python scripts/evaluate.py generate --task trajectory --output results/trajectory.json
```

Defaults: 100 designs per target, guidance 1.0, full 500-step diffusion schedule, constraint projection, DE population size 90 and maximum 200 generations. Multi-posture and trajectory generation uses ULA with 10 correction steps and gamma 0.5. Three-posture temperatures are optimized independently. Trajectory optimization warm-starts each frame from the preceding temperatures and adds `5e-4 * ||t - t_previous||^2` to its objective; reported MSE excludes this penalty. Temperature ordering is `t3 >= t2`.

For an inexpensive execution check:

```bash
python scripts/evaluate.py generate --task single --limit 1 --samples 1 --maxiter 1 --device cpu --output results/smoke.json
```

This reduced optimization budget is not the benchmark protocol. Full evaluation can take substantial time. Completed targets are saved atomically after each target; automatic resume is not implemented. Existing output files are protected from overwriting.

Each run uses fresh random states unless the user supplies `--seed` for their own experiment. Independent runs can produce different designs, temperatures, and metrics. Optional user seeds support repeatability within a compatible environment; different hardware or library versions can still alter numerical results.

## Compare a new run with reference samples

```bash
python scripts/evaluate.py compare --task single --input results/single.json
```

The comparison reports descriptive statistics on matching target IDs. For a meaningful benchmark comparison, use the full target set and the default optimization budget and guidance. Repeated independent runs and target-level uncertainty estimates are preferable to treating individual generated designs as independent observations. Similar distributions must be assessed empirically; this script does not establish statistical equivalence. Historical reference samples are fixed observations, not outputs expected to match a new run design by design.

## Data schema and units

- `targets/{single,three,trajectory}.json`: 30 single-posture targets, 15 three-posture sets, and 7 trajectories. Each entry has `target_id` and `postures` of shape `(K, 100, 3)`. Coordinates are normalized by 100 mm and are already in the model's coordinate convention. Preserve node order and do not renormalize them.
- `reference/*.json`: `task`, `guidance`, and `results`. Each target contains `designs`, each with `geometry` (6 values), `temperatures` (`K x 3`), and `mse` (`K` values). Only final samples and their pure surrogate MSE are included.
- Geometry order: `[h3, theta1, theta3, beta31, beta32, delta3]`. Multiply elementwise by `[1, 100, 100, 360, 360, 180]` for the physical convention (h3 is a height fraction; remaining parameters are angles in degrees).
- Temperature values are normalized: `T_C = 20 + 120 * t`.
- Diffusion operates internally in `[-1, 1]^6`; released sample geometry uses `DEFAULT_GEOM_BOUNDS`, not diffusion coordinates. Conversion is performed by `remap`.
- `par2sig` converts the six geometry and three temperature parameters to the surrogate's `(100, 6)` signal representation.
- MSE is averaged over all 100 nodes and three coordinates. A design succeeds if its **maximum** MSE over the target postures/frames is strictly less than `1e-3`. Aggregate success is the mean of per-target design success rates. Reported median/p90 describe per-design maximum MSE.

The reference samples use guidance 1.0. Model files contain tensor state dictionaries only.

## Scope of the evidence

All success and error values in this package are **surrogate-based**. They measure agreement with the released surrogate and do not establish agreement with an independent finite-element solver or physical experiment. Abaqus models, simulation results, and FEA verification are not included. The package supports recomputation of saved-sample surrogate errors and independent generation plus temperature optimization. It does not support retraining or exact replay of historical stochastic runs.

## Layout

```text
models/          Final network architectures
validation/      Geometry sampling, projection, and optimization utilities
checkpoints/     Inference-only tensor state dictionaries
runtime.py       Independent model and diffusion-schedule loading
scripts/         Readable evaluation entry point
targets/        Benchmark target postures
reference/       Released final samples
```

License: pending.
