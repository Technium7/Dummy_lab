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
from trajectory_planner.collision import is_state_valid  # noqa: E402
from trajectory_planner.scene import (  # noqa: E402
    build_planning_scene,
    disconnect_planning_scene,
    get_end_link_pose,
    joint_array_from_transformed_deg,
    refine_joint_config_to_pose,
    set_joint_positions,
)
from trajectory_planner.trajectory_utils import compress_duplicate_waypoints, resample_trajectory  # noqa: E402

from .models import GrabSensorConfig, GraspTargets


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


def _refine_to_position(
    scene,
    target_pos: Sequence[float],
    q_init: Sequence[float],
    *,
    pos_tol_m: float = 0.005,
) -> List[float]:
    """Joint refinement that matches Cartesian position only (orientation free)."""
    low = np.asarray(scene.lower_limits, dtype=np.float64)
    high = np.asarray(scene.upper_limits, dtype=np.float64)
    target_pos_arr = np.asarray(target_pos, dtype=np.float64)
    q = np.clip(np.asarray(q_init, dtype=np.float64), low, high)

    def metric(q_try: np.ndarray) -> float:
        set_joint_positions(scene, q_try)
        pos, _ = get_end_link_pose(scene)
        return float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - target_pos_arr))

    steps = [0.02, 0.005, 0.001, 0.0002, 0.00005, 0.00001]
    for step in steps:
        improved = True
        while improved:
            improved = False
            base = metric(q)
            for j in range(len(q)):
                for delta in (-step, step):
                    q_try = q.copy()
                    q_try[j] = float(np.clip(q_try[j] + delta, low[j], high[j]))
                    err = metric(q_try)
                    if err + 1e-15 < base:
                        q = q_try
                        base = err
                        improved = True
            if base <= pos_tol_m:
                break

    set_joint_positions(scene, q)
    return q.tolist()


def plan_cartesian_approach_segment(
    scene,
    start_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    *,
    enable_collision: bool,
    n_steps: int = 24,
    pos_tol_m: float = 0.015,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """
    Short straight-line approach in Cartesian space, holding orientation from ``start_q``.

    Used when full-pose IK fails on the final pre-grasp -> grasp move: pre-grasp IK
    often hits position with large orientation error, so re-solving grasp with the
    ideal ``target_rpy`` can miss the taught grasp point by tens of mm.
    """
    set_joint_positions(scene, start_q)
    start_pos, start_rpy = get_end_link_pose(scene)
    start_pos_arr = np.asarray(start_pos, dtype=np.float64)
    target_pos_arr = np.asarray(target_pos, dtype=np.float64)

    q = list(start_q)
    waypoints: List[List[float]] = [list(start_q)]
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp_pos = (1.0 - alpha) * start_pos_arr + alpha * target_pos_arr
        q = refine_joint_config_to_pose(
            scene,
            interp_pos,
            start_rpy,
            q,
            pos_tol_m=0.002,
            ori_tol_rad=0.6,
        )
        if not is_state_valid(scene, q):
            return None, None
        waypoints.append(list(q))

    grasp_goal_q = waypoints[-1]
    set_joint_positions(scene, grasp_goal_q)
    end_pos, _ = get_end_link_pose(scene)
    pos_err = float(np.linalg.norm(np.asarray(end_pos, dtype=np.float64) - target_pos_arr))
    print(
        f"Final approach (Cartesian hold-orientation): pos_err={pos_err * 1000:.3f} mm"
    )
    if pos_err > pos_tol_m:
        grasp_goal_q = _refine_to_position(
            scene,
            target_pos_arr,
            grasp_goal_q,
            pos_tol_m=pos_tol_m,
        )
        if not is_state_valid(scene, grasp_goal_q):
            return None, None
        set_joint_positions(scene, grasp_goal_q)
        end_pos, _ = get_end_link_pose(scene)
        pos_err = float(np.linalg.norm(np.asarray(end_pos, dtype=np.float64) - target_pos_arr))
        print(
            f"Final approach (position-only refine): pos_err={pos_err * 1000:.3f} mm"
        )
        if pos_err > pos_tol_m:
            return None, None

    traj = plan_joint_trajectory(
        scene,
        start_q,
        grasp_goal_q,
        target_pos,
        target_rpy,
        enable_collision_penalty=enable_collision,
    )
    if traj is None:
        raw = np.asarray(waypoints, dtype=np.float64)
        traj = resample_trajectory(raw, max(12, len(raw) * 3))
    return traj, grasp_goal_q


def plan_grasp_segment(
    scene,
    start_q: Sequence[float],
    targets: GraspTargets,
    *,
    enable_collision: bool,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    grasp_traj, grasp_goal_q = plan_segment(
        scene,
        start_q,
        targets.grasp_pos_m,
        targets.grasp_rpy_rad,
        enable_collision=enable_collision,
    )
    if grasp_traj is not None and grasp_goal_q is not None:
        return grasp_traj, grasp_goal_q

    print(
        "[grab_sensor] Full-pose IK failed for final grasp; "
        "retrying with Cartesian approach from pre-grasp."
    )
    return plan_cartesian_approach_segment(
        scene,
        start_q,
        targets.grasp_pos_m,
        targets.grasp_rpy_rad,
        enable_collision=enable_collision,
    )


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

        grasp_traj, grasp_goal_q = plan_grasp_segment(
            scene,
            pre_goal_q,
            targets,
            enable_collision=cfg.enable_collision,
        )
        if grasp_traj is None or grasp_goal_q is None:
            raise RuntimeError(
                "Failed to plan final grasp segment (IK or iLQR). "
                "Typical causes: grasp target out of reach, or fixed_grasp_rpy_robot_deg "
                "incompatible with the arm at the taught grasp location."
            )

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
