from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import NonlinearConstraint, differential_evolution
# Geometric bounds
NATURAL_GEOM_BOUNDS: list[tuple[float, float]] = [
    (0.0, 1.0),
    (10.0, 100.0),
    (10.0, 100.0),
    (-360.0, 360.0),
    (-360.0, 360.0),
    (0.0, 180.0),
]

DEFAULT_GEOM_BOUNDS: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.1, 1.0),
    (0.1, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 1.0),
]

DEFAULT_VALIDATION_GEOM_BOUNDS: list[tuple[float, float]] = [
    (0.05, 0.95),
    (0.1, 1.0),
    (0.1, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 1.0),
]
# Temperature Bounds
DEFAULT_TEMP_BOUNDS: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
]

NATURAL_TEMP_BOUNDS : list[tuple[float, float]] = [
    (20.0, 140.0),
    (20.0, 140.0),
    (20.0, 140.0),
]
# Geometric Constraints
def overlap_23(params):
    return np.minimum((params[5]*180+params[3]*360+params[4]*360)%360-180+params[2]/2*100,
                      180+params[2]/2*100-(params[5]*180+params[3]*360+params[4]*360)%360)
GEOM_CONSTRAINTS = [
    NonlinearConstraint(lambda params: np.atleast_1d(params[5]*180-((params[1]+params[2])/2*100)), 0, np.inf),
    NonlinearConstraint(lambda params: np.atleast_1d(params[5]*180+params[2]/2*100-180), -np.inf, 0),
    NonlinearConstraint(lambda params: np.atleast_1d(((params[1]+params[2])/2*100)-(params[5]*180+params[3]*360+params[4]*360)%360), -np.inf, 0),
    NonlinearConstraint(lambda params: np.atleast_1d((params[5]*180+params[3]*360+params[4]*360)%360-360+((params[1]+params[2])/2*100)), -np.inf, 0),
    NonlinearConstraint(lambda params: np.atleast_1d(overlap_23(params)), -np.inf, 0),
]
# Helper Functions
def _init_population(
    seed: np.ndarray,
    total: int,
    bounds: list[tuple[float, float]],
    rng_seed=None,
) -> np.ndarray:
    """Build a DE initial population with seed as the first member, rest uniform random."""
    n = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    pop = rng.uniform(lo, hi, size=(total, n))
    pop[0] = np.clip(np.asarray(seed, dtype=np.float64), lo, hi)
    return pop


def par2sig(param_matrix):
    """Convert (N, 9) parameter rows into (N, 100, 6) surrogate input signals."""
    if isinstance(param_matrix, torch.Tensor):
        param = param_matrix.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        param = np.asarray(param_matrix, dtype=np.float32)

    h3 = param[:, 0]
    theta1 = param[:, 1]
    theta3 = param[:, 2]
    beta31 = param[:, 3]
    beta32 = param[:, 4]
    delta3 = param[:, 5]
    t1 = param[:, 6]
    t2 = param[:, 7]
    t3 = param[:, 8]

    ns = param.shape[0]
    z = np.linspace(0.0, 1.0, 100, dtype=np.float32)

    eps = np.finfo(np.float32).eps
    h3_clipped = np.clip(h3, eps, 1.0 - eps)

    Z = z[None, :]
    H = h3_clipped[:, None]
    B31 = beta31[:, None] * 360.0
    B32 = beta32[:, None] * 360.0
    D3 = delta3[:, None] * 180.0

    left = (-D3 - B31 / H * Z) / 180.0
    right =(-D3 - B31 - B32 / (1.0 - H) * (Z - H)) / 180.0
    s3 = np.where(Z < H, left, right)
    s3 = np.mod(s3 + 1.0, 2.0) - 1.0

    out = np.empty((ns, 100, 6), dtype=np.float32)
    out[:, :, 0] = s3
    out[:, :, 1] = theta1[:, None]
    out[:, :, 2] = theta3[:, None]
    out[:, :, 3] = t1[:, None]
    out[:, :, 4] = t2[:, None]
    out[:, :, 5] = t3[:, None]
    return out

def _prepare_target_tensor(target: np.ndarray | torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Target postures are prepared to be of shape [n, 100, 3], n being the number of postures"""
    target_t = target if isinstance(target, torch.Tensor) else torch.from_numpy(np.asarray(target, dtype=np.float32))
    target_t = target_t.to(device=device, dtype=torch.float32)
    if target_t.dim() == 2:
        target_t = target_t.unsqueeze(0)
    if target_t.dim() != 3 or target_t.shape[1:] != (100, 3):
        raise ValueError(f"Expected target posture shape (100,3) or (1,100,3), got {tuple(target_t.shape)}")
    return target_t
# Loss Functions
def loss_fn_temp(
    temp_params: np.ndarray,
    target: np.ndarray | torch.Tensor,
    selected_design: np.ndarray,
    forward_model: nn.Module,
    device: torch.device | str,
) -> float:
    """MSE objective for optimizing 3 temperature parameters with fixed geometry.

    Parameters
    ----------
    temp_params : array of shape (3,)
        Normalized temperature parameters [t1, t2, t3] ∈ [0, 1].
    selected_design : array of shape (6,)
        Fixed normalized geometry parameters [h3, θ1, θ3, β31, β32, δ3].
    """
    device = torch.device(device)
    temp = np.asarray(temp_params, dtype=np.float32)
    design = np.asarray(selected_design, dtype=np.float32)

    all_params = np.hstack([design, temp])[None, :]
    input_signals = torch.from_numpy(par2sig(all_params)).to(device)
    target_t = _prepare_target_tensor(target, device)

    forward_model.eval()
    with torch.no_grad():
        pred = forward_model(input_signals)
        mse = torch.mean((pred - target_t) ** 2).item()

    return float(mse)


def loss_fn_joint(
    all_params: np.ndarray,
    target: np.ndarray | torch.Tensor,
    forward_model: nn.Module,
    device: torch.device | str,
) -> float:
    """MSE objective for jointly optimizing 6 geometry + 3 temperature parameters.

    Parameters
    ----------
    all_params : array of shape (9,)
        Normalized joint parameter vector
        [h3, θ1, θ3, β31, β32, δ3, t1, t2, t3], all in [0, 1]
        except h3 ∈ (0.05, 0.95) per DEFAULT_VALIDATION_GEOM_BOUNDS.
    """
    device = torch.device(device)
    params = np.asarray(all_params, dtype=np.float32)[None, :]  # (1, 9)
    input_signals = torch.from_numpy(par2sig(params)).to(device)
    target_t = _prepare_target_tensor(target, device)

    forward_model.eval()
    with torch.no_grad():
        pred = forward_model(input_signals)
        mse = torch.mean((pred - target_t) ** 2).item()

    return float(mse)
# Optimization
def run_de_temp(
    target: np.ndarray | torch.Tensor,
    design: np.ndarray,
    forward_model: nn.Module,
    device: torch.device | str,
    num_points: int = 5,
    maxiter: int = 200,
    seed: Optional[int] = None,
    init_point: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Run differential evolution over 3 temperatures with fixed geometry.

    Parameters
    ----------
    init_point : array of shape (3,), optional
        Seed for the first member of the initial population.
        Defaults to [0, 0, 0] (no actuation / cold start).

    Returns
    -------
    best_temp : np.ndarray, shape (3,)
    best_loss : float
    loss_history : list of per-iteration best losses
    """
    temp_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    # Temperature ordering constraint: t3 >= t2 (indices 2 and 1 in the 3-vector)
    temp_symmetry = NonlinearConstraint(lambda t: t[2] - t[1], 0.0, np.inf)

    seed_temp = np.zeros(3, dtype=np.float64) if init_point is None else np.asarray(init_point, dtype=np.float64)
    total_pop = max(num_points * len(temp_bounds), 5)
    init_pop  = _init_population(seed_temp, total_pop, temp_bounds, rng_seed=seed)

    opt_loss_history: list[float] = []

    def callback_fn(xk: np.ndarray, convergence: float) -> None:
        del convergence
        opt_loss_history.append(loss_fn_temp(xk, target, design, forward_model, device))

    result = differential_evolution(
        loss_fn_temp,
        bounds=temp_bounds,
        args=(target, design, forward_model, device),
        strategy="best1bin",
        init=init_pop,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=False,
        callback=callback_fn,
        maxiter=maxiter,
        seed=seed,
        polish=True,
        updating="deferred",
        constraints=(temp_symmetry,),
    )

    return result.x.astype(np.float32), float(result.fun), opt_loss_history


def loss_fn_temp_with_prev(
    temp_params: np.ndarray,
    target: np.ndarray | torch.Tensor,
    selected_design: np.ndarray,
    forward_model: nn.Module,
    device: torch.device | str,
    prev_temp: np.ndarray,
    lam: float = 5e-4,
) -> float:
    """MSE reconstruction loss plus a continuity penalty against the previous temperature.

    Parameters
    ----------
    prev_temp : array of shape (3,)
        Optimized temperature from the preceding trajectory step.
    lam : float
        Coefficient weighting the L2 penalty  λ·||t - t_prev||².
    """
    mse = loss_fn_temp(temp_params, target, selected_design, forward_model, device)
    reg = lam * float(np.sum((np.asarray(temp_params, dtype=np.float32) - prev_temp) ** 2))
    return mse + reg


def run_de_temp_trajectory(
    targets: np.ndarray | torch.Tensor,
    design: np.ndarray,
    forward_model: nn.Module,
    device: torch.device | str,
    num_points: int = 5,
    maxiter: int = 200,
    lam: float = 5e-4,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, list[float], list[list[float]]]:
    """Sequentially optimize temperature for every posture in a trajectory.

    Posture 0 is handled by plain ``run_de_temp`` (no prior exists).
    Postures 1 … n-1 minimize  MSE + λ·||t - t_prev||²  so that the optimized
    temperatures evolve smoothly along the trajectory.

    Parameters
    ----------
    targets : array of shape (n_postures, 100, 3)
        Ordered posture sequence in [0, 1] (model conditioning scale).
    design : array of shape (6,)
        Fixed normalized geometry in DEFAULT_GEOM_BOUNDS.
    lam : float
        Continuity coefficient (default 5e-4).
    seed : int or None
        Base RNG seed; posture i uses seed+i when provided.

    Returns
    -------
    best_temps : np.ndarray, shape (n_postures, 3)
        Optimized temperature triple for each posture.
    best_losses : list[float]
        Pure MSE (regularization excluded) for each posture.
    loss_histories : list[list[float]]
        Per-iteration best objective recorded by the DE callback
        (includes regularization for postures > 0).
    """
    if isinstance(targets, torch.Tensor):
        targets_np = targets.cpu().numpy().astype(np.float32)
    else:
        targets_np = np.asarray(targets, dtype=np.float32)

    best_temps: list[np.ndarray] = []
    best_losses: list[float] = []
    loss_histories: list[list[float]] = []

    prev_temp: Optional[np.ndarray] = None

    temp_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    temp_symmetry = NonlinearConstraint(lambda t: t[2] - t[1], 0.0, np.inf)

    for i, target_i in enumerate(targets_np):
        i_seed = None if seed is None else seed + i

        if prev_temp is None:
            # First posture: no predecessor, fall back to standard DE
            opt_temp, opt_loss, history = run_de_temp(
                target=target_i,
                design=design,
                forward_model=forward_model,
                device=device,
                num_points=num_points,
                maxiter=maxiter,
                seed=i_seed,
            )
        else:
            _prev = prev_temp.copy()  # snapshot for closures
            history: list[float] = []

            total_pop = max(num_points * len(temp_bounds), 5)
            init_pop  = _init_population(_prev, total_pop, temp_bounds, rng_seed=i_seed)

            def _callback(xk: np.ndarray, convergence: float) -> None:
                del convergence
                history.append(
                    loss_fn_temp_with_prev(
                        xk, target_i, design, forward_model, device, _prev, lam
                    )
                )

            result = differential_evolution(
                loss_fn_temp_with_prev,
                bounds=temp_bounds,
                args=(target_i, design, forward_model, device, _prev, lam),
                strategy="best1bin",
                init=init_pop,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False,
                callback=_callback,
                maxiter=maxiter,
                seed=i_seed,
                polish=True,
                updating="deferred",
                constraints=(temp_symmetry,),
            )

            opt_temp = result.x.astype(np.float32)
            # Recompute pure MSE for the reported loss (no regularization)
            opt_loss = loss_fn_temp(opt_temp, target_i, design, forward_model, device)

        best_temps.append(opt_temp)
        best_losses.append(float(opt_loss))
        loss_histories.append(history)
        prev_temp = opt_temp

    return np.array(best_temps, dtype=np.float32), best_losses, loss_histories


def run_de_joint(
    target: np.ndarray | torch.Tensor,
    forward_model: nn.Module,
    device: torch.device | str,
    num_points: int = 5,
    maxiter: int = 200,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Run differential evolution jointly over 6 geometry + 3 temperature parameters.

    Geometry bounds follow DEFAULT_VALIDATION_GEOM_BOUNDS (normalized) and are
    subject to GEOM_CONSTRAINTS.  Temperatures are constrained to [0, 1] with
    the ordering requirement t3 >= t2.

    Returns
    -------
    best_params : np.ndarray, shape (9,)
        Optimized [h3, θ1, θ3, β31, β32, δ3, t1, t2, t3].
    best_loss : float
    loss_history : list of per-iteration best losses
    """
    # 6 geometry bounds (normalized) followed by 3 temperature bounds
    joint_bounds = DEFAULT_VALIDATION_GEOM_BOUNDS + [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    # Temperature ordering constraint: t3 >= t2 (indices 7 and 8 in the 9-vector)
    temp_symmetry = NonlinearConstraint(lambda p: np.atleast_1d(p[8] - p[7]), 0.0, np.inf)

    # Geometry constraints operate on indices 0-5, which still align in the 9-vector
    all_constraints = list(GEOM_CONSTRAINTS) + [temp_symmetry]

    opt_loss_history: list[float] = []

    def callback_fn(xk: np.ndarray, convergence: float) -> None:
        del convergence
        opt_loss_history.append(loss_fn_joint(xk, target, forward_model, device))

    result = differential_evolution(
        loss_fn_joint,
        bounds=[tuple(map(float, b)) for b in joint_bounds],
        args=(target, forward_model, device),
        strategy="best1bin",
        popsize=num_points,
        init="latinhypercube",
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=False,
        callback=callback_fn,
        maxiter=maxiter,
        seed=seed,
        polish=True,
        updating="deferred",
        constraints=all_constraints,
    )

    return result.x.astype(np.float32), float(result.fun), opt_loss_history