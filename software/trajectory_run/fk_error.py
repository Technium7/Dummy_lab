"""Forward kinematics and Cartesian error for robot joint logs."""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from manual_control.mapping import transformed_deg_to_sim_rad  # noqa: E402
from movement_sim.joint_controller import (  # noqa: E402
    build_package_map,
    choose_description,
    discover_descriptions,
    materialize_urdf,
)
from trajectory_planner.scene import (  # noqa: E402
    PlanningScene,
    build_planning_scene,
    disconnect_planning_scene,
    get_end_link_pose,
    pose_error_rad,
    set_joint_positions,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def robot_deg_to_sim_rad(joints_deg: Sequence[float]) -> List[float]:
    return [transformed_deg_to_sim_rad(j, float(joints_deg[j])) for j in range(6)]


@dataclass
class PoseError:
    pos_mm: float
    ori_deg: float
    pos_delta_mm: Tuple[float, float, float]
    ee_pos_m: Tuple[float, float, float]


@contextmanager
def fk_scene(
    *,
    root: Path | None = None,
    robot_hint: str = "Dummy_ros2",
    env_rel: str = "physical_env/packaging_station_edit.obj",
) -> Iterator[PlanningScene]:
    root = root or _repo_root()
    env = root / env_rel
    descs = discover_descriptions(root)
    selected = choose_description(descs, robot_hint, root)
    pkg = build_package_map(root)
    urdf, tmp = materialize_urdf(selected, pkg)
    extra = {root, selected.parent, urdf.parent, *pkg.values()}
    scene = build_planning_scene(
        urdf, env, 1.0, True, extra, False, "", False
    )
    try:
        yield scene
    finally:
        disconnect_planning_scene(scene)
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def ee_pose_from_sim_q(scene: PlanningScene, q_sim_rad: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    set_joint_positions(scene, q_sim_rad)
    pos, rpy = get_end_link_pose(scene)
    return np.asarray(pos, dtype=np.float64), np.asarray(rpy, dtype=np.float64)


def ee_pose_from_robot_deg(scene: PlanningScene, joints_deg: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    return ee_pose_from_sim_q(scene, robot_deg_to_sim_rad(joints_deg))


def position_error_vs_reference_q(
    scene: PlanningScene,
    measured_deg: Sequence[float],
    reference_q_sim: Sequence[float],
    *,
    target_pos: Sequence[float] | None = None,
    target_rpy: Sequence[float] | None = None,
) -> PoseError:
    ref_pos, ref_rpy = ee_pose_from_sim_q(scene, reference_q_sim)
    act_pos, act_rpy = ee_pose_from_robot_deg(scene, measured_deg)
    delta = act_pos - ref_pos
    pos_mm = float(np.linalg.norm(delta) * 1000.0)

    if target_pos is not None and target_rpy is not None:
        cart_pos_mm, cart_ori_deg = pose_error_rad(act_pos, act_rpy, target_pos, target_rpy)
        ori_deg = float(math.degrees(cart_ori_deg))
        # Report vs Cartesian target for goal checks; pos_mm stays vs FK reference.
        _ = cart_pos_mm
    else:
        _, ori_rad = pose_error_rad(act_pos, act_rpy, ref_pos, ref_rpy)
        ori_deg = float(math.degrees(ori_rad))

    return PoseError(
        pos_mm=pos_mm,
        ori_deg=ori_deg,
        pos_delta_mm=(float(delta[0] * 1000), float(delta[1] * 1000), float(delta[2] * 1000)),
        ee_pos_m=(float(act_pos[0]), float(act_pos[1]), float(act_pos[2])),
    )


def cartesian_error_vs_target(
    scene: PlanningScene,
    measured_deg: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
) -> Tuple[float, float]:
    act_pos, act_rpy = ee_pose_from_robot_deg(scene, measured_deg)
    pos_m, ori_rad = pose_error_rad(act_pos, act_rpy, target_pos, target_rpy)
    return float(pos_m * 1000.0), float(math.degrees(ori_rad))
