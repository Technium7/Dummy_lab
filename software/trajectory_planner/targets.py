"""Load ``plan_targets.json`` and cache plans to ``last_plan.npz``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from manual_control.mapping import HOME_JOINTS_DEG
from manual_control.pose_conventions import (
    measure_home_abc_ref_for_scene,
    resolve_target_end_link,
)

_PACKAGE_DIR = Path(__file__).resolve().parent


def targets_path() -> Path:
    return _PACKAGE_DIR / "plan_targets.json"


def plan_cache_path() -> Path:
    return _PACKAGE_DIR / "last_plan.npz"


def load_plan_targets_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_plan_targets(
    data: dict[str, Any],
    *,
    home_abc_ref: Sequence[float],
) -> Tuple[List[float], Tuple[float, float, float], Tuple[float, float, float], Optional[List[float]]]:
    """Resolve JSON targets to planner world-frame pose (matches manual_control GUI convention)."""
    start_deg = [float(v) for v in data.get("start_joints_deg", HOME_JOINTS_DEG)]
    tgt = data["target_end_link"]
    pos, rpy = resolve_target_end_link(tgt, home_abc_ref=home_abc_ref)
    goal_deg_raw = data.get("goal_joints_deg")
    goal_deg = [float(v) for v in goal_deg_raw] if goal_deg_raw is not None else None
    return start_deg, pos, rpy, goal_deg


def load_plan_targets(
    path: Path,
    *,
    home_abc_ref: Sequence[float] | None = None,
) -> Tuple[List[float], Tuple[float, float, float], Tuple[float, float, float], Optional[List[float]]]:
    data = load_plan_targets_raw(path)
    tgt = data["target_end_link"]

    if home_abc_ref is not None and (
        "orientation_abc_deg" in tgt or "position_robot_m" in tgt
    ):
        return resolve_plan_targets(data, home_abc_ref=home_abc_ref)

    # Legacy: world-frame position + RPY stored directly.
    start_deg = [float(v) for v in data.get("start_joints_deg", HOME_JOINTS_DEG)]
    pos = tuple(float(v) for v in tgt["position_m"])
    rpy = tuple(float(v) for v in tgt["orientation_rpy_rad"])
    goal_deg_raw = data.get("goal_joints_deg")
    goal_deg = [float(v) for v in goal_deg_raw] if goal_deg_raw is not None else None
    return start_deg, pos, rpy, goal_deg


def resolve_plan_targets_from_scene(scene, path: Path) -> Tuple[List[float], Tuple[float, float, float], Tuple[float, float, float], Optional[List[float]]]:
    """Load targets using HOME A/B/C reference measured on the live planning scene."""
    data = load_plan_targets_raw(path)
    home_abc_ref = measure_home_abc_ref_for_scene(
        scene.robot_id,
        scene.end_link_idx,
        scene.joint_indices,
        data.get("start_joints_deg", HOME_JOINTS_DEG),
    )
    return resolve_plan_targets(data, home_abc_ref=home_abc_ref)


def save_plan_cache(
    trajectory: np.ndarray,
    goal_q: Sequence[float],
    start_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    start_deg: Sequence[float],
    seed: int,
    *,
    goal_pose: bool = False,
    collision_penalty: bool = False,
) -> None:
    np.savez(
        plan_cache_path(),
        trajectory=trajectory,
        goal_q=np.asarray(goal_q, dtype=np.float64),
        start_q=np.asarray(start_q, dtype=np.float64),
        target_pos=np.asarray(target_pos, dtype=np.float64),
        target_rpy=np.asarray(target_rpy, dtype=np.float64),
        start_deg=np.asarray(start_deg, dtype=np.float64),
        seed=np.int64(seed),
        goal_pose=np.bool_(goal_pose),
        collision_penalty=np.bool_(collision_penalty),
    )


def load_plan_cache() -> Optional[Dict[str, Any]]:
    path = plan_cache_path()
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    out: Dict[str, Any] = {
        "trajectory": data["trajectory"],
        "goal_q": data["goal_q"].tolist(),
        "start_q": data["start_q"].tolist(),
        "target_pos": tuple(float(v) for v in data["target_pos"]),
        "target_rpy": tuple(float(v) for v in data["target_rpy"]),
        "start_deg": [float(v) for v in data["start_deg"]],
        "seed": int(data["seed"]),
    }
    if "goal_pose" in data.files:
        out["goal_pose"] = bool(data["goal_pose"])
    if "collision_penalty" in data.files:
        out["collision_penalty"] = bool(data["collision_penalty"])
    return out
