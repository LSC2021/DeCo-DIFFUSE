# Evaluation protocol

The default evaluation uses 100 designs per target and guidance 1.0. The diffusion schedule has 500 steps. Geometric constraints are enforced by intermediate projection and final repair. Multi-posture and trajectory sampling uses ULA with 10 correction steps and gamma 0.5.

Temperature optimization uses differential evolution with a population of 90, at most 200 generations, and the constraint `t3 >= t2`. Three-posture targets have independently optimized temperature triples. For trajectories, each frame is warm-started from the preceding frame and the objective adds `5e-4 * ||t - t_previous||^2`. Saved MSE excludes this continuity penalty.

Each run uses fresh random states by default. `--seed` sets a seed for a new experiment. Independent runs may produce different designs and metrics; library versions and hardware can also affect numerical results. Reference samples support direct error verification and distribution comparisons rather than exact stochastic replay.

The comparison command reports success rates, error quantiles, and per-target success rates for matching target IDs. Use the full benchmark and the default optimization budget for a benchmark comparison. Repeated runs can help assess variability. The output is descriptive and does not establish statistical equivalence.

Outputs are saved after each completed target. Existing output files cannot be overwritten; automatic resume is not implemented. Assets are resolved relative to the repository, while output paths are relative to the current working directory.

## Data schema and units

- `targets/{single,three,trajectory}.json`: 30 single-posture targets, 15 three-posture sets, and 7 trajectories. Each entry has `target_id` and `postures` of shape `(K, 100, 3)`. Coordinates are normalized by 100 mm and are already in the model's coordinate convention. Preserve node order and do not renormalize them.
- `reference/*.json`: `task`, `guidance`, and `results`. Each target contains `designs`, each with `geometry` (6 values), `temperatures` (`K x 3`), and `mse` (`K` values). Only final samples and their pure surrogate MSE are included.
- Geometry order: `[h3, theta1, theta3, beta31, beta32, delta3]`. Multiply elementwise by `[1, 100, 100, 360, 360, 180]` for the physical convention (h3 is a height fraction; remaining parameters are angles in degrees).
- Temperature values are normalized: `T_C = 20 + 120 * t`.
- Diffusion operates internally in `[-1, 1]^6`; released sample geometry uses `DEFAULT_GEOM_BOUNDS`, not diffusion coordinates. Conversion is performed by `remap`.
- `par2sig` converts the six geometry and three temperature parameters to the surrogate's `(100, 6)` signal representation.
- MSE is averaged over all 100 nodes and three coordinates. A design succeeds if its **maximum** MSE over the target postures/frames is strictly less than `1e-3`. Aggregate success is the mean of per-target design success rates. Reported median/p90 describe per-design maximum MSE.

The reference samples use guidance 1.0. Model files contain tensor state dictionaries only.

## Verification tolerance

The verification command recomputes predictions from the saved geometries and temperatures. It checks geometric feasibility and reports the largest absolute difference from saved MSE. A difference below `1e-5` passes the numerical check. This tolerance does not imply bitwise agreement or identical success classifications for designs close to the `1e-3` threshold.
