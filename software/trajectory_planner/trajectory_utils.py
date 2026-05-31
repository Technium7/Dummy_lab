"""Shared trajectory resampling helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Joint angles closer than this (rad) are treated as the same configuration.
DEFAULT_WAYPOINT_DEDUP_TOL_RAD = 1e-5


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
    goal_q: Sequence[float] | None = None,
    *,
    pin_goal: bool = True,
) -> np.ndarray:
    out = np.asarray(traj, dtype=np.float64).copy()
    out[0] = np.asarray(start_q, dtype=np.float64)
    if pin_goal and goal_q is not None:
        out[-1] = np.asarray(goal_q, dtype=np.float64)
    return out


def spread_terminal_approach(
    traj: np.ndarray,
    goal_q: Sequence[float],
    max_step_rad: float = 0.05,
) -> np.ndarray:
    """
    Replace a single jump into ``goal_q`` with a bounded per-step ramp.

    Walks backward from the last sample so each consecutive pair differs by at most
    ``max_step_rad`` per joint (avoids end-of-path snap after hard goal pinning).
    """
    out = np.asarray(traj, dtype=np.float64).copy()
    goal = np.asarray(goal_q, dtype=np.float64)
    step = float(max_step_rad)
    out[-1] = goal
    for i in range(len(out) - 2, -1, -1):
        diff = out[i + 1] - out[i]
        out[i] = out[i + 1] - np.clip(diff, -step, step)
    return out


def compress_duplicate_waypoints(
    trajectory: np.ndarray,
    tol_rad: float = DEFAULT_WAYPOINT_DEDUP_TOL_RAD,
) -> np.ndarray:
    """Drop consecutive duplicate poses (keeps first of each run)."""
    traj = np.asarray(trajectory, dtype=np.float64)
    if len(traj) <= 1:
        return traj
    keep = [0]
    for i in range(1, len(traj)):
        if float(np.max(np.abs(traj[i] - traj[keep[-1]]))) > tol_rad:
            keep.append(i)
    if keep[-1] != len(traj) - 1:
        keep.append(len(traj) - 1)
    return traj[keep]


def count_trailing_duplicate_waypoints(
    trajectory: np.ndarray,
    tol_rad: float = DEFAULT_WAYPOINT_DEDUP_TOL_RAD,
) -> int:
    """How many tail rows share the same joints as the final waypoint (includes last row)."""
    traj = np.asarray(trajectory, dtype=np.float64)
    if len(traj) == 0:
        return 0
    goal = traj[-1]
    run = 0
    for i in range(len(traj) - 1, -1, -1):
        if float(np.max(np.abs(traj[i] - goal))) <= tol_rad:
            run += 1
        else:
            break
    return run


def representative_waypoint_map(
    trajectory: np.ndarray,
    tol_rad: float = DEFAULT_WAYPOINT_DEDUP_TOL_RAD,
) -> Tuple[List[int], List[int]]:
    """
    Map duplicate joint poses to representatives.

    Returns ``(representative_indices, rep_for_waypoint)`` where
    ``rep_for_waypoint[i]`` is the trajectory index used for waypoint ``i``.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    n = len(traj)
    if n == 0:
        return [], []
    representatives = [0]
    rep_for: List[int] = [0]
    for i in range(1, n):
        ref = representatives[-1]
        if float(np.max(np.abs(traj[i] - traj[ref]))) > tol_rad:
            representatives.append(i)
            rep_for.append(i)
        else:
            rep_for.append(ref)
    return representatives, rep_for


def assert_trajectory_endpoints(
    trajectory: np.ndarray,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    tol_rad: float = 1e-5,
) -> None:
    """Raise if cached/planned endpoints do not match ``start_q`` / ``goal_q``."""
    traj = np.asarray(trajectory, dtype=np.float64)
    start = np.asarray(start_q, dtype=np.float64)
    goal = np.asarray(goal_q, dtype=np.float64)
    if len(traj) == 0:
        raise ValueError("Trajectory is empty.")
    d0 = float(np.max(np.abs(traj[0] - start)))
    d1 = float(np.max(np.abs(traj[-1] - goal)))
    if d0 > tol_rad or d1 > tol_rad:
        raise ValueError(
            f"Trajectory endpoints do not match plan (max |Δ| rad: start={d0:.2e}, goal={d1:.2e}). "
            "Replan without --replay to refresh last_plan.npz."
        )


def prepare_playback_trajectory(
    trajectory: np.ndarray,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    hold_steps: int = 15,
    *,
    pin_endpoints: bool = True,
) -> np.ndarray:
    """Optionally pin endpoints, then append goal hold samples.

    Set ``pin_endpoints=False`` on replay so cached joint samples are not rewritten.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if pin_endpoints:
        traj = pin_trajectory_endpoints(traj, start_q, goal_q)
    if hold_steps <= 0:
        return traj
    goal_row = np.asarray(goal_q, dtype=np.float64)
    trailing = count_trailing_duplicate_waypoints(traj)
    extra_hold = max(0, int(hold_steps) - max(0, trailing - 1))
    if extra_hold == 0:
        return traj
    hold = np.tile(goal_row, (extra_hold, 1))
    return np.vstack([traj, hold])
