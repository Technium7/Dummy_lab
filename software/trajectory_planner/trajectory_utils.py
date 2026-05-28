"""Shared trajectory resampling helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


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
