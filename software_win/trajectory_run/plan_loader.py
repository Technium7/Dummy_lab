"""Load ``last_plan.npz`` without importing the full trajectory_planner stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

_PLANNER_DIR = Path(__file__).resolve().parent.parent / "trajectory_planner"


def default_plan_path() -> Path:
    return _PLANNER_DIR / "last_plan.npz"


def load_plan_npz(path: Path | None = None) -> Dict[str, Any]:
    p = path or default_plan_path()
    if not p.is_file():
        raise FileNotFoundError(p)
    data = np.load(p, allow_pickle=False)
    out: Dict[str, Any] = {
        "trajectory": np.asarray(data["trajectory"], dtype=np.float64),
        "goal_q": np.asarray(data["goal_q"], dtype=np.float64),
        "start_q": np.asarray(data["start_q"], dtype=np.float64),
    }
    if "target_pos" in data.files:
        out["target_pos"] = np.asarray(data["target_pos"], dtype=np.float64)
    if "target_rpy" in data.files:
        out["target_rpy"] = np.asarray(data["target_rpy"], dtype=np.float64)
    return out


def assert_trajectory_endpoints(
    trajectory: np.ndarray,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    tol_rad: float = 1e-5,
) -> None:
    traj = np.asarray(trajectory, dtype=np.float64)
    start = np.asarray(start_q, dtype=np.float64)
    goal = np.asarray(goal_q, dtype=np.float64)
    if len(traj) == 0:
        raise ValueError("Trajectory is empty.")
    d0 = float(np.max(np.abs(traj[0] - start)))
    d1 = float(np.max(np.abs(traj[-1] - goal)))
    if d0 > tol_rad or d1 > tol_rad:
        raise ValueError(
            f"Trajectory endpoints mismatch (max |Δ| rad: start={d0:.2e}, goal={d1:.2e}). "
            "Replan to refresh last_plan.npz."
        )


def load_planned_trajectory_rad(
    plan_path: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = load_plan_npz(plan_path)
    trajectory = cache["trajectory"]
    start_q = cache["start_q"]
    goal_q = cache["goal_q"]
    assert_trajectory_endpoints(trajectory, start_q, goal_q)
    return trajectory, start_q, goal_q
