"""Discrete-time iLQR joint-space trajectory optimization (JAX Riccati + PyBullet collision)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .collision import batch_collision_penalty_and_gradient, is_state_valid
from .lqr_jax import IlqrJaxSolver
from .scene import PlanningScene
from .trajectory_utils import pin_trajectory_endpoints, resample_trajectory

_JAX_SOLVER: Optional[IlqrJaxSolver] = None


def _get_solver() -> IlqrJaxSolver:
    global _JAX_SOLVER
    if _JAX_SOLVER is None:
        _JAX_SOLVER = IlqrJaxSolver()
    return _JAX_SOLVER


def _clip_states(traj: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    out = traj.copy()
    np.clip(out, low, high, out=out)
    return out


def _initial_guess(start_q: np.ndarray, goal_q: np.ndarray, horizon: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, horizon)
    return start_q + t[:, None] * (goal_q - start_q)[None, :]


def plan_lqr_trajectory(
    scene: PlanningScene,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    horizon: int = 80,
    n_ilqr_iterations: int = 35,
    q_weight: float = 8.0,
    qf_weight: float = 120.0,
    r_weight: float = 2.5,
    clearance_margin: float = 0.02,
    penetration_weight: float = 5.0e4,
    clearance_weight: float = 80.0,
    reg: float = 1e-2,
    max_control_step: float = 0.12,
    line_search_alphas: Optional[Sequence[float]] = None,
    resample_to: int = 150,
    enable_collision_penalty: bool = False,
) -> Optional[np.ndarray]:
    """
    Optimize joint trajectory with JAX-compiled discrete iLQR.

    When ``enable_collision_penalty`` is False (default), only joint-to-goal tracking
    and control regularization are optimized (no PyBullet queries in the iLQR loop).
    """
    start = np.asarray(start_q, dtype=np.float64)
    goal = np.asarray(goal_q, dtype=np.float64)
    horizon = max(4, int(horizon))
    low = np.asarray(scene.lower_limits, dtype=np.float64)
    high = np.asarray(scene.upper_limits, dtype=np.float64)

    if enable_collision_penalty:
        if not is_state_valid(scene, start) or not is_state_valid(scene, goal):
            return None

    penalty_kw = dict(
        clearance_margin=clearance_margin,
        penetration_weight=penetration_weight,
        clearance_weight=clearance_weight,
    )
    alphas = list(line_search_alphas or (1.0, 0.5, 0.25, 0.1, 0.05))
    solver = _get_solver()

    traj = _initial_guess(start, goal, horizon)
    traj[0] = start
    traj[-1] = goal
    traj = _clip_states(traj, low, high)
    u_nom = traj[1:] - traj[:-1]

    zero_pen = np.zeros(horizon, dtype=np.float64)
    zero_grad = np.zeros((horizon, start.shape[0]), dtype=np.float64)

    for _ in range(n_ilqr_iterations):
        if enable_collision_penalty:
            col_pen, col_grad = batch_collision_penalty_and_gradient(scene, traj, **penalty_kw)
        else:
            col_pen, col_grad = zero_pen, zero_grad
        total = solver.total_cost(traj, u_nom, goal, col_pen, q_weight, r_weight, qf_weight)

        k_ff, k_fb = solver.backward(
            traj, u_nom, goal, col_grad, q_weight, r_weight, qf_weight, reg
        )
        candidates = solver.forward_candidates(
            traj, u_nom, k_ff, k_fb, start, goal, alphas, low, high, max_control_step
        )

        accepted = False
        for alpha_idx, candidate in enumerate(candidates):
            new_traj = np.array(candidate, dtype=np.float64, copy=True)
            new_traj[0] = start
            new_traj[-1] = goal
            new_u = new_traj[1:] - new_traj[:-1]
            if enable_collision_penalty:
                col_pen_new, _ = batch_collision_penalty_and_gradient(scene, new_traj, **penalty_kw)
            else:
                col_pen_new = zero_pen
            trial = solver.total_cost(new_traj, new_u, goal, col_pen_new, q_weight, r_weight, qf_weight)
            if trial <= total * 1.02:
                traj = np.array(new_traj, dtype=np.float64, copy=True)
                u_nom = traj[1:] - traj[:-1]
                accepted = True
                if alpha_idx > 0:
                    pass  # used smaller alpha after line search
                break

        if not accepted:
            break

    dense = resample_trajectory(traj, resample_to)
    dense = pin_trajectory_endpoints(dense, start, goal)
    return dense
