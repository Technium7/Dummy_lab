"""CHOMP-style trajectory smoothing (covariant gradient optimization, Python implementation)."""

from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np

from .scene import PlanningScene


def _finite_diff_gradient(traj: np.ndarray, idx: int, joint: int, eps: float, cost_fn) -> float:
    saved = float(traj[idx, joint])
    traj[idx, joint] = saved + eps
    c_plus = cost_fn(traj)
    traj[idx, joint] = saved - eps
    c_minus = cost_fn(traj)
    traj[idx, joint] = saved
    return (c_plus - c_minus) / (2.0 * eps)


def chomp_smooth(
    scene: PlanningScene,
    waypoints: np.ndarray,
    collision_cost_fn: Callable[[Sequence[float]], float],
    is_valid: Callable[[Sequence[float]], bool],
    start_q: Sequence[float] | None = None,
    goal_q: Sequence[float] | None = None,
    n_iterations: int = 80,
    learning_rate: float = 0.04,
    smooth_weight: float = 18.0,
    obstacle_weight: float = 25.0,
    waypoint_weight: float = 6.0,
    grad_eps: float = 1e-3,
) -> np.ndarray:
    """
    Optimize trajectory for smoothness + obstacle cost while staying near RRTConnect waypoints.
    Matches CHOMP objective structure (smooth + obstacle + path prior).
    """
    traj = np.asarray(waypoints, dtype=np.float64).copy()
    n, dof = traj.shape
    if n < 3:
        return traj

    if start_q is not None:
        traj[0] = np.asarray(start_q, dtype=np.float64)
    if goal_q is not None:
        traj[-1] = np.asarray(goal_q, dtype=np.float64)

    prior = traj.copy()
    low = np.asarray(scene.lower_limits, dtype=np.float64)
    high = np.asarray(scene.upper_limits, dtype=np.float64)

    def total_cost(t: np.ndarray) -> float:
        smooth = 0.0
        for i in range(1, n - 1):
            d2 = t[i + 1] - 2.0 * t[i] + t[i - 1]
            smooth += float(np.dot(d2, d2))
        obs = sum(collision_cost_fn(t[i]) for i in range(n))
        prior_err = float(np.sum((t - prior) ** 2))
        return smooth_weight * smooth + obstacle_weight * obs + waypoint_weight * prior_err

    for _ in range(n_iterations):
        grad = np.zeros_like(traj)
        for i in range(1, n - 1):
            for j in range(dof):
                grad[i, j] = _finite_diff_gradient(traj, i, j, grad_eps, total_cost)
        traj[1:-1] -= learning_rate * grad[1:-1]
        np.clip(traj, low, high, out=traj)
        if start_q is not None:
            traj[0] = np.asarray(start_q, dtype=np.float64)
        else:
            traj[0] = prior[0]
        if goal_q is not None:
            traj[-1] = np.asarray(goal_q, dtype=np.float64)
        else:
            traj[-1] = prior[-1]

        if not all(is_valid(traj[i]) for i in range(n)):
            traj = 0.7 * traj + 0.3 * prior
            np.clip(traj, low, high, out=traj)

    return traj


def resample_trajectory(traj: np.ndarray, n_samples: int) -> np.ndarray:
    if len(traj) < 2:
        return np.repeat(traj, n_samples, axis=0) if len(traj) else traj
    t_old = np.linspace(0.0, 1.0, len(traj))
    t_new = np.linspace(0.0, 1.0, n_samples)
    out = np.zeros((n_samples, traj.shape[1]), dtype=np.float64)
    for j in range(traj.shape[1]):
        out[:, j] = np.interp(t_new, t_old, traj[:, j])
    return out


def pin_trajectory_endpoints(
    traj: np.ndarray,
    start_q: Sequence[float],
    goal_q: Sequence[float],
) -> np.ndarray:
    out = np.asarray(traj, dtype=np.float64).copy()
    out[0] = np.asarray(start_q, dtype=np.float64)
    out[-1] = np.asarray(goal_q, dtype=np.float64)
    return out
