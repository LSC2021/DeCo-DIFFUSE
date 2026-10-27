# DeCo-DIFFUSE

Generative inverse design of LCE-driven soft robots with compositional diffusion and actuation optimization.

## Overview

Designing a soft robot for a prescribed deformation requires choosing both its geometry and its actuation. Multiple geometries may produce similar responses, while a single geometry may need to satisfy several target postures under different actuation conditions.

**DeCo-DIFFUSE** combines a posture-conditioned diffusion model with a Bi-LSTM forward surrogate and differential-evolution temperature optimization. The diffusion model generates candidate fiber geometries from target postures. Compositional sampling accommodates multiple postures or trajectory frames, and geometric projection enforces design constraints. For each candidate geometry, temperature optimization finds actuation settings that minimize the surrogate-predicted posture error.

The repository provides three benchmark tasks:

1. **Single-posture design:** generate geometries for one target posture.
2. **Multi-posture design:** find a shared geometry that achieves three target postures with different temperatures.
3. **Trajectory design:** find a shared geometry and a sequence of temperatures for an ordered set of target frames.

Pretrained models, benchmark targets, evaluation code, and reference designs are included. Evaluation uses the forward surrogate; finite-element validation is outside the scope of this repository.

## Repository Structure

```text
.
├── models/
│   ├── lstm.py                     # Bi-LSTM forward surrogate
│   └── cfg_diffusion.py            # Conditional geometry diffusion model
├── validation/
│   ├── design_generation_utils.py # Sampling and geometry projection
│   └── opt_utils.py               # Parameter conversion and DE optimization
├── checkpoints/
│   ├── surrogate.pt               # Pretrained forward surrogate
│   └── diffusion.pt               # Pretrained diffusion model
├── targets/
│   ├── single.json                # 30 single-posture targets
│   ├── three.json                 # 15 three-posture sets
│   └── trajectory.json            # 7 trajectories
├── reference/                     # Geometry, temperatures, and MSE per design
├── scripts/
│   └── evaluate.py                # Generation, verification, and comparison
├── runtime.py                     # Model loading and diffusion schedule
├── docs/evaluation.md             # Evaluation protocol and data conventions
├── requirements.txt
├── requirements-tested.txt
└── VALIDATION.md                  # Verified reference metrics and environment
```

## Installation

Clone the repository and install the dependencies in a Python 3.10+ environment:

```bash
git clone https://github.com/LSC2021/DeCo-DIFFUSE.git
cd DeCo-DIFFUSE
python -m pip install -r requirements.txt
```

CUDA is recommended for full evaluation; CPU execution is also supported. Install the appropriate PyTorch build for your hardware. The versions used for validation are listed in [requirements-tested.txt](requirements-tested.txt).

## Quick Start

### 1. Verify the reference designs

Recompute surrogate errors for the provided geometries and optimized temperatures:

```bash
python scripts/evaluate.py verify --task single
python scripts/evaluate.py verify --task three
python scripts/evaluate.py verify --task trajectory
```

For a quick check, add `--limit 1` to evaluate only the first target. No design generation or optimization is needed for this step.

### 2. Generate and optimize new designs

Run the task of interest:

```bash
# Single-posture design
python scripts/evaluate.py generate --task single --output results/single.json

# Three-posture design
python scripts/evaluate.py generate --task three --output results/three.json

# Trajectory design
python scripts/evaluate.py generate --task trajectory --output results/trajectory.json
```

Each command generates 100 designs per target and optimizes their actuation temperatures. Results contain the geometry, temperatures, and surrogate MSE for each design.

To check execution with a smaller optimization budget:

```bash
python scripts/evaluate.py generate --task single --limit 1 --samples 1 --maxiter 1 --device cpu --output results/example.json
```

This short run checks the workflow; it does not reproduce benchmark performance.

### 3. Compare with the reference results

```bash
python scripts/evaluate.py compare --task single --input results/single.json
```

The report compares success rates, error quantiles, and per-target success rates. New runs use fresh random states by default, so generated designs and metrics may differ from the reference samples. Use `--seed` to specify a seed for your own run.

## Pretrained Models

Both pretrained models are included as inference-only PyTorch state dictionaries:

| Model | Checkpoint | Role |
|---|---|---|
| Bi-LSTM surrogate | `checkpoints/surrogate.pt` | Predict posture from geometry and temperatures |
| Conditional diffusion | `checkpoints/diffusion.pt` | Generate geometry conditioned on target postures |

Training code and training datasets are not included. All examples run directly with the provided weights.

## Reproducing the Reference Results

The reference set contains 100 designs per target, for a total of 5,200 designs. A design succeeds when its maximum surrogate MSE across the required postures or frames is below `1e-3`.

| Task | Targets | Reference designs | Success rate | Verification command |
|---|---:|---:|---:|---|
| Single posture | 30 | 3,000 | 96.87% | `python scripts/evaluate.py verify --task single` |
| Three postures | 15 | 1,500 | 93.60% | `python scripts/evaluate.py verify --task three` |
| Trajectory | 7 | 700 | 88.57% | `python scripts/evaluate.py verify --task trajectory` |

These values describe the saved reference samples. Independent generation and optimization runs support statistical comparison rather than exact replay. All reported errors and success rates are **surrogate-based** and do not constitute FEA or experimental validation.

See [Evaluation Protocol](docs/evaluation.md) for sampling settings, optimization parameters, data formats, units, and numerical tolerances. [Validation](VALIDATION.md) records the checks performed on the provided samples.

## License

License selection is pending.
