from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from movement_sim.joint_controller import (  # noqa: E402
    build_package_map,
    choose_description,
    discover_descriptions,
    materialize_urdf,
)
from trajectory_planner.goals import solve_goal_from_pose  # noqa: E402
from trajectory_planner.planning import finalize_trajectory_for_cache, plan_joint_trajectory  # noqa: E402
from trajectory_planner.scene import (  # noqa: E402
    build_planning_scene,
    disconnect_planning_scene,
    joint_array_from_transformed_deg,
)
from trajectory_planner.trajectory_utils import compress_duplicate_waypoints  # noqa: E402

from .types import GrabSensorConfig, GraspTargets


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_urdf(cfg: GrabSensorConfig) -> Tuple[Path, Optional[Path], set[Path]]:
    root = _repo_root()
    candidates = discover_descriptions(root)
    selected = choose_description(candidates, cfg.robot_name, root)
    package_map = build_package_map(root)
    urdf_path, temp_file = materialize_urdf(selected, package_map)
    search_paths = {root, selected.parent, urdf_path.parent, *package_map.values()}
    return urdf_path, temp_file, search_paths


def build_scene(cfg: GrabSensorConfig, *, use_gui: bool = False):
    urdf_path, temp_file, search_paths = _resolve_urdf(cfg)
    scene = build_planning_scene(
        urdf_path=urdf_path,
        env_path=cfg.env_path,
        mesh_scale=cfg.mesh_scale,
        fixed_base=True,
        extra_search_paths=search_paths,
        load_plane=False,
        end_link_hint=cfg.end_link,
        use_gui=use_gui,
    )
    return scene, temp_file


def start_q_from_robot_joints_deg(scene, joints_deg: Sequence[float]) -> List[float]:
    return joint_array_from_transformed_deg(scene, joints_deg)


def plan_segment(
    scene,
    start_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    *,
    enable_collision: bool,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    goal_q = solve_goal_from_pose(scene, target_pos, target_rpy, start_q)
    if goal_q is None:
        return None, None

    traj = plan_joint_trajectory(
        scene,
        start_q,
        goal_q,
        target_pos,
        target_rpy,
        enable_collision_penalty=enable_collision,
    )
    return traj, goal_q


def plan_approach_and_grasp(
    cfg: GrabSensorConfig,
    start_joints_deg: Sequence[float],
    targets: GraspTargets,
    *,
    use_gui: bool = False,
) -> Tuple[np.ndarray, dict]:
    scene, temp_file = build_scene(cfg, use_gui=use_gui)
    meta: dict = {}
    try:
        start_q = start_q_from_robot_joints_deg(scene, start_joints_deg)

        pre_traj, pre_goal_q = plan_segment(
            scene,
            start_q,
            targets.pre_grasp_pos_m,
            targets.pre_grasp_rpy_rad,
            enable_collision=cfg.enable_collision,
        )
        if pre_traj is None or pre_goal_q is None:
            raise RuntimeError(
                "Failed to plan pre-grasp segment (IK or iLQR). "
                "Typical causes: uncalibrated camera_to_robot in grasp config, "
                "grasp target out of reach, or wrong depth segmentation band."
            )

        grasp_traj, grasp_goal_q = plan_segment(
            scene,
            pre_goal_q,
            targets.grasp_pos_m,
            targets.grasp_rpy_rad,
            enable_collision=cfg.enable_collision,
        )
        if grasp_traj is None or grasp_goal_q is None:
            raise RuntimeError("Failed to plan final grasp segment (IK or iLQR).")

        combined = np.vstack([pre_traj, grasp_traj[1:]])
        combined = compress_duplicate_waypoints(combined)
        combined = finalize_trajectory_for_cache(combined, start_q, grasp_goal_q)

        meta.update(
            {
                "n_pre": len(pre_traj),
                "n_grasp": len(grasp_traj),
                "n_total": len(combined),
                "start_q": start_q,
                "pre_goal_q": pre_goal_q,
                "grasp_goal_q": grasp_goal_q,
            }
        )
        return combined, meta
    finally:
        disconnect_planning_scene(scene)
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
