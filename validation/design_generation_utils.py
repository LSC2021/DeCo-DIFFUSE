from typing import Optional

import secrets

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from tqdm import trange
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

DIFFUSION_GEOM_BOUNDS = [
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
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

_DDPM_EPS = 1e-8   # numerical stability in DDPM reverse update
_PROJ_MAXITER = 200  # cap per-sample work inside the nonlinear projection solve
_PROJ_TOL = 1e-8  # feasibility/objective tolerance for the SciPy projection solve
_PROJ_INTERVAL = 100  # only project every N reverse-diffusion steps to limit overhead
_FINAL_REPAIR_EPS = 1e-3  # keep the final repaired sample slightly inside the feasible set
# Parameter remapping
def remap(
    x: "np.ndarray | torch.Tensor",
    src_bounds: list[tuple[float, float]],
    dst_bounds: list[tuple[float, float]],
) -> "np.ndarray | torch.Tensor":
    """Linearly remap each column of x from src_bounds to dst_bounds.

    Accepts numpy arrays or torch tensors; returns the same type.
    """
    if isinstance(x, torch.Tensor):
        lo_s = torch.tensor([b[0] for b in src_bounds], dtype=x.dtype, device=x.device)
        hi_s = torch.tensor([b[1] for b in src_bounds], dtype=x.dtype, device=x.device)
        lo_d = torch.tensor([b[0] for b in dst_bounds], dtype=x.dtype, device=x.device)
        hi_d = torch.tensor([b[1] for b in dst_bounds], dtype=x.dtype, device=x.device)
    else:
        lo_s = np.array([b[0] for b in src_bounds], dtype=np.float32)
        hi_s = np.array([b[1] for b in src_bounds], dtype=np.float32)
        lo_d = np.array([b[0] for b in dst_bounds], dtype=np.float32)
        hi_d = np.array([b[1] for b in dst_bounds], dtype=np.float32)
    return lo_d + (x - lo_s) / (hi_s - lo_s) * (hi_d - lo_d)
# DDPM helpers
def extract(coefficients: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Extract timestep-indexed coefficients and reshape to match `x` rank."""
    return coefficients.index_select(0, t).reshape((t.shape[0],) + (1,) * (x.dim() - 1))
# Constraint projection — box clip + joint nonlinear solve
def _geom_bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return geometry box bounds as NumPy arrays for SciPy projection."""
    lo = np.array([b[0] for b in DEFAULT_VALIDATION_GEOM_BOUNDS], dtype=np.float64)
    hi = np.array([b[1] for b in DEFAULT_VALIDATION_GEOM_BOUNDS], dtype=np.float64)
    return lo, hi


_GEOM_BOUNDS_LO, _GEOM_BOUNDS_HI = _geom_bounds_arrays()
_GEOM_SCIPY_BOUNDS = Bounds(_GEOM_BOUNDS_LO, _GEOM_BOUNDS_HI)  # reused by every projection call


def is_geom_feasible(params: np.ndarray, tol: float = _PROJ_TOL) -> "bool | np.ndarray":
    """Check GEOM_CONSTRAINTS for samples in DEFAULT_GEOM_BOUNDS space.

    Accepts a single sample (1-D) and returns bool, or a batch (2-D) and returns a bool array.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.ndim == 1:
        for constraint in GEOM_CONSTRAINTS:
            values = np.atleast_1d(constraint.fun(params))
            lower = np.broadcast_to(np.asarray(constraint.lb, dtype=np.float64), values.shape)
            upper = np.broadcast_to(np.asarray(constraint.ub, dtype=np.float64), values.shape)
            if np.any(values < lower - tol) or np.any(values > upper + tol):
                return False
        return True
    return np.array([is_geom_feasible(row, tol=tol) for row in params])


def is_geom_within_bounds(x: np.ndarray) -> "bool | np.ndarray":
    """Check if each sample (DEFAULT_GEOM_BOUNDS space) lies within DEFAULT_VALIDATION_GEOM_BOUNDS.

    Accepts a single sample (1-D) and returns bool, or a batch (2-D) and returns a bool array.
    """
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    x = np.atleast_2d(x)
    lo = np.array([b[0] for b in DEFAULT_VALIDATION_GEOM_BOUNDS])
    hi = np.array([b[1] for b in DEFAULT_VALIDATION_GEOM_BOUNDS])
    mask = np.all((x >= lo) & (x <= hi), axis=1)
    return bool(mask[0]) if single else mask


def _project_single_geom(
    params: np.ndarray,
    max_iter: int = _PROJ_MAXITER,
    tol: float = _PROJ_TOL,
) -> np.ndarray:
    """Project one geometry row with a joint nonlinear minimum-distance solve.

    The projection pipeline is intentionally generic:
      1. Clip onto the box bounds exactly.
      2. Solve one constrained optimization problem over all nonlinear constraints.

    This keeps every non-box constraint in the same projection stage rather than
    baking in hand-derived sequential repairs for one specific geometry.
    """
    # Clip to parameter bounds before nonlinear projection.
    clipped = np.clip(np.asarray(params, dtype=np.float64), _GEOM_BOUNDS_LO, _GEOM_BOUNDS_HI)
    if is_geom_feasible(clipped, tol=tol):
        return clipped.astype(np.float32)

    def objective(candidate: np.ndarray) -> float:
        # Euclidean projection onto the nonlinear feasible set inside the box.
        diff = candidate - clipped
        return 0.5 * float(np.dot(diff, diff))

    result = minimize(
        objective,
        x0=clipped,
        method="SLSQP",
        bounds=_GEOM_SCIPY_BOUNDS,
        constraints=GEOM_CONSTRAINTS,
        options={"maxiter": max_iter, "ftol": tol, "disp": False},
    )

    projected = (
        np.clip(result.x, _GEOM_BOUNDS_LO, _GEOM_BOUNDS_HI)
        if result.x is not None else clipped
    )
    if not is_geom_feasible(projected, tol=10.0 * tol):
        raise RuntimeError(
            "Joint geometry projection failed to find a feasible point. "
            f"Solver message: {result.message}"
        )
    return projected.astype(np.float32)


def _project_geom_constraints_batch(
    x: torch.Tensor,
    max_iter: int = _PROJ_MAXITER,
    tol: float = _PROJ_TOL,
) -> torch.Tensor:
    """Project geometry samples with a generic two-stage feasibility pipeline.

    Stage 1 clips each sample onto DEFAULT_GEOM_BOUNDS exactly.
    Stage 2 solves one joint nonlinear projection problem over GEOM_CONSTRAINTS.

    SciPy operates on one row at a time, so this wrapper keeps the public sampler
    interface batched while applying the same projection rule to every sample.
    """
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)

    # Project in DEFAULT_GEOM_BOUNDS space because that is where GEOM_CONSTRAINTS live.
    g = remap(x.detach().cpu().numpy(), DIFFUSION_GEOM_BOUNDS, DEFAULT_GEOM_BOUNDS)
    projected = np.stack(
        [_project_single_geom(row, max_iter=max_iter, tol=tol) for row in g],
        axis=0,
    )

    # Map the feasible geometries back into diffusion space for the next DDPM step.
    x_proj = remap(projected, DEFAULT_GEOM_BOUNDS, DIFFUSION_GEOM_BOUNDS)
    x_proj = torch.as_tensor(x_proj, dtype=x.dtype, device=x.device)
    return x_proj.squeeze(0) if squeeze else x_proj
# Final repair for constraints projection: Resolve tol issues

def _project_interval_np(val: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Exact interval projection used by the final analytical repair."""
    return np.minimum(np.maximum(val, lo), hi)


def _project_union_interval_np(
    val: np.ndarray,
    lo1: np.ndarray, hi1: np.ndarray,
    lo2: np.ndarray, hi2: np.ndarray,
) -> np.ndarray:
    """Project onto the union of two intervals by snapping to the nearest endpoint."""
    feasible = ((val >= lo1) & (val <= hi1)) | ((val >= lo2) & (val <= hi2))
    endpoints = np.stack([lo1, hi1, lo2, hi2], axis=-1)
    nearest = endpoints[np.arange(endpoints.shape[0]), np.abs(endpoints - val[:, None]).argmin(axis=-1)]
    return np.where(feasible, val, nearest)


def _project_beta_pair_to_sum_np(
    beta31: np.ndarray,
    beta32: np.ndarray,
    target_sum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest projection of (beta31, beta32) onto beta31 + beta32 = target_sum."""
    y1 = _project_interval_np(
        0.5 * (target_sum + beta31 - beta32),
        np.maximum(-1.0, target_sum - 1.0),
        np.minimum(1.0, target_sum + 1.0),
    )
    return y1, np.clip(target_sum - y1, -1.0, 1.0)


def _final_analytic_repair_batch(x: torch.Tensor) -> torch.Tensor:
    """Apply one closed-form repair at the end so the returned samples are strictly feasible."""
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)

    # Repair in DEFAULT_GEOM_BOUNDS space because the constraints are defined there.
    g = remap(x.detach().cpu().numpy().astype(np.float64), DIFFUSION_GEOM_BOUNDS, DEFAULT_GEOM_BOUNDS)
    g = np.clip(g, _GEOM_BOUNDS_LO, _GEOM_BOUNDS_HI)

    theta1 = g[:, 1] * 100.0
    theta3 = g[:, 2] * 100.0
    beta31 = g[:, 3]
    beta32 = g[:, 4]
    delta3 = g[:, 5] * 180.0

    avg_theta = 0.5 * (theta1 + theta3)
    half_theta3 = 0.5 * theta3
    eps = np.full_like(avg_theta, _FINAL_REPAIR_EPS)

    # C1 and C2 reduce to a single interval for delta3.
    delta3 = _project_interval_np(delta3, avg_theta + eps, 180.0 - half_theta3 - eps)

    # C3, C4, and C5 constrain the wrapped bottom angle phi.
    phi = np.mod(delta3 + 360.0 * beta31 + 360.0 * beta32, 360.0)
    phi = _project_union_interval_np(
        phi,
        avg_theta + eps, 180.0 - half_theta3 - eps,
        180.0 + half_theta3 + eps, 360.0 - avg_theta - eps,
    )

    # Choose the valid winding number closest to the current beta sum, then
    # project the beta pair onto that repaired sum exactly.
    winding = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    candidates = (phi[:, None] - delta3[:, None] + 360.0 * winding[None, :]) / 360.0
    valid = (candidates >= -2.0) & (candidates <= 2.0)
    cost = np.where(valid, np.abs(candidates - (beta31 + beta32)[:, None]), np.inf)
    target_sum = candidates[np.arange(candidates.shape[0]), cost.argmin(axis=-1)]
    beta31, beta32 = _project_beta_pair_to_sum_np(beta31, beta32, target_sum)

    g[:, 3] = beta31
    g[:, 4] = beta32
    g[:, 5] = delta3 / 180.0
    g = np.clip(g, _GEOM_BOUNDS_LO, _GEOM_BOUNDS_HI)

    x_repaired = remap(g, DEFAULT_GEOM_BOUNDS, DIFFUSION_GEOM_BOUNDS)
    x_repaired = torch.as_tensor(x_repaired, dtype=x.dtype, device=x.device)
    return x_repaired.squeeze(0) if squeeze else x_repaired
# Sampler
def run_single_posture_sampler(
    model: nn.Module,
    schedule: dict[str, torch.Tensor],
    posture: Optional[torch.Tensor],
    num_samples: int,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
    device: Optional[torch.device | str] = None,
    record_flow: bool = True,
    use_constraint_projection: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reverse-sample geometry designs conditioned on a single posture."""

    if seed is None:
        seed = secrets.randbits(32)
    torch.manual_seed(seed)

    # Number of diffusion steps is encoded in the schedule tensor length.
    n_steps = schedule["betas"].shape[0]

    # Expand the single posture to match the sample batch dimension.
    posture_batch = (
        posture.reshape(1, -1).expand(num_samples, -1).to(device)
        if posture is not None else None
    )

    model.eval()
    with torch.no_grad():
        # Sampling starts from pure Gaussian noise in the normalised 6D design space.
        x = torch.randn(num_samples, model.diffusion_net.geom_dim, device=device)
        flow_states = [x.detach().cpu()] if record_flow else []

        for step in trange(n_steps - 1, -1, -1, desc="Diffusion sampling"):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)

            # Classifier-free guidance for a single conditioning signal.
            eps_cond   = model(x, t.float().unsqueeze(-1), posture=posture_batch)
            eps_uncond = model(x, t.float().unsqueeze(-1), posture=None, force_unconditional=True)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # Standard DDPM reverse-update written in epsilon-prediction form.
            beta_t           = extract(schedule["betas"], t, x)
            sqrt_one_minus_ab = extract(schedule["sqrt_one_minus_alpha_cumprod"], t, x)
            sqrt_recip_alpha  = extract(schedule["sqrt_recip_alphas"], t, x)
            mean = sqrt_recip_alpha * (x - beta_t / (sqrt_one_minus_ab + _DDPM_EPS) * eps)

            if step > 0:
                var = extract(schedule["posterior_variance"], t, x)
                x = mean + torch.sqrt(var + _DDPM_EPS) * torch.randn_like(x)
            else:
                x = mean

            if use_constraint_projection and step % _PROJ_INTERVAL == 0:
                x = _project_geom_constraints_batch(x)

            if record_flow:
                flow_states.append(x.detach().cpu())

    if use_constraint_projection:
        # Apply one exact repair at the end without changing the diffusion trajectory.
        x = _final_analytic_repair_batch(x)
        if record_flow:
            flow_states[-1] = x.detach().cpu()

    if record_flow:
        flow_tensor = torch.stack(flow_states, dim=0)
        flow_steps  = torch.arange(n_steps, -1, -1, device="cpu")
    else:
        flow_tensor = torch.empty(0)
        flow_steps  = torch.empty(0, dtype=torch.long)

    return x, {"steps": flow_steps, "states": flow_tensor}


def run_multiposture_sampler(
    model: nn.Module,
    schedule: dict[str, torch.Tensor],
    postures: torch.Tensor,
    num_samples: int,
    guidance_scale: float = 1.0,
    n_mcmc_steps: int = 5,
    gamma: float = 0.05,
    seed: Optional[int] = None,
    device: Optional[torch.device | str] = None,
    use_constraint_projection: bool = False,
    proj_interval: int = _PROJ_INTERVAL,
    record_flow: bool = False,
    condition_batch_size: int = 0,
) -> "torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]":
    """Generate geometry from composed posture conditions using DDPM and ULA.

    Guidance is divided by the number of conditions. Constraint projection
    runs at the requested interval and final repair enforces feasibility.
    """
    n_steps = schedule["betas"].shape[0]
    if seed is None:
        seed = secrets.randbits(32)
    torch.manual_seed(seed)

    if postures.dim() == 2:
        postures = postures.unsqueeze(0)

    posture_batches = [p.unsqueeze(0).to(device).expand(num_samples, -1, -1) for p in postures]
    n_cond = len(posture_batches)
    guidance_eff = guidance_scale / n_cond

    def conditional_predictions(xx, tt):
        if condition_batch_size <= 0:
            return [model(xx, tt.float().unsqueeze(-1), posture=pb) for pb in posture_batches]
        predictions = []
        for first in range(0, n_cond, condition_batch_size):
            group = posture_batches[first:first + condition_batch_size]
            result = model(xx.repeat(len(group), 1),
                           tt.float().unsqueeze(-1).repeat(len(group), 1),
                           posture=torch.cat(group, dim=0))
            predictions.extend(result.split(num_samples, dim=0))
        return predictions

    model.eval()

    with torch.no_grad():
        x = torch.randn(num_samples, model.diffusion_net.geom_dim, device=device)
        flow_states = [x.detach().cpu().clone()] if record_flow else []

        for step in trange(n_steps - 1, -1, -1, desc="Diffusion sampling"):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)

            # Predictor step: compose unconditional and conditional scores first.
            eps_uncond = model(x, t.float().unsqueeze(-1), posture=None, force_unconditional=True)
            eps_cond = conditional_predictions(x, t)
            sum_cond = torch.stack(eps_cond, dim=0).sum(dim=0)
            eps_comp = guidance_eff * sum_cond + (1.0 - guidance_eff * n_cond) * eps_uncond

            beta_t = extract(schedule["betas"], t, x)
            sqrt_one_minus_ab = extract(schedule["sqrt_one_minus_alpha_cumprod"], t, x)
            sqrt_recip_alpha = extract(schedule["sqrt_recip_alphas"], t, x)

            mean = sqrt_recip_alpha * (x - beta_t / (sqrt_one_minus_ab + _DDPM_EPS) * eps_comp)

            if step > 0:
                var = extract(schedule["posterior_variance"], t, x)
                x = mean + torch.sqrt(var + _DDPM_EPS) * torch.randn_like(x)
            else:
                x = mean

            if step > 0:
                # Corrector: refine the sample at the current timestep with MCMC.
                def _comp_score(xx):
                    eps_u = model(xx, t.float().unsqueeze(-1), posture=None, force_unconditional=True)
                    eps_c = conditional_predictions(xx, t)
                    sum_c = torch.stack(eps_c, dim=0).sum(dim=0)
                    eps_now = guidance_eff * sum_c + (1.0 - guidance_eff * n_cond) * eps_u
                    return -eps_now / (sqrt_one_minus_ab + _DDPM_EPS)

                for _ in range(n_mcmc_steps):
                    score = _comp_score(x)
                    step_size = gamma * torch.sqrt(beta_t)
                    x = x + (step_size ** 2 / 2.0) * score + step_size * torch.randn_like(x)

            if use_constraint_projection and proj_interval > 0 and step % proj_interval == 0:
                x = _project_geom_constraints_batch(x)

            if record_flow:
                flow_states.append(x.detach().cpu().clone())

    if use_constraint_projection:
    # Apply one exact repair at the end without changing the diffusion trajectory.
        x = _final_analytic_repair_batch(x)

    if record_flow:
        flow_states[-1] = x.detach().cpu().clone()
        return x, {"steps": torch.arange(n_steps, -1, -1), "states": torch.stack(flow_states)}
    return x