from __future__ import annotations

import math
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
from trajectory_planner.planning import finalize_trajectory_for_cache, plan_joint_trajectory  # noqa: E402
from trajectory_planner.collision import evaluate_collision_metrics, is_state_valid  # noqa: E402
from trajectory_planner.scene import (  # noqa: E402
    build_planning_scene,
    disconnect_planning_scene,
    get_end_link_pose,
    joint_array_from_transformed_deg,
    pose_error_rad,
    set_joint_positions,
)
from trajectory_planner.trajectory_utils import compress_duplicate_waypoints, resample_trajectory  # noqa: E402

from manual_control.mapping import HOME_JOINTS_DEG  # noqa: E402

from .models import GrabSensorConfig, GraspTargets
from .grasp import prepare_fixed_robot_grasp_targets
from .table_position import robot_point_table_display
from .tcp_ik import (
    TcpOffset,
    enforce_goal_orientation,
    fk_grasp_points,
    link_pose_from_q,
    measure_tcp_target_pose_error,
    measure_target_pose_error,
    position_error_mm,
    refine_joint_config_to_target_pose,
    refine_joint_config_to_target_position,
    grasp_ik_goal_from_home,
    solve_tcp_ik,
    solve_goal_from_target_pose,
    tcp_world_rpy_from_q,
    target_pose_from_q,
    tcp_error_mm,
    wrist_tcp_distance_mm,
)

# Grasp targets are in PyBullet planner world (= robot GUI). The scene loads the robot
# with 180° Z; do not apply an extra XY flip on positions.


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
        env_path=cfg.env_path if cfg.load_environment else None,
        mesh_scale=cfg.mesh_scale,
        fixed_base=True,
        extra_search_paths=search_paths,
        load_plane=use_gui,
        end_link_hint=cfg.end_link,
        tcp_link_hint=cfg.tcp_link,
        use_gui=use_gui,
    )
    return scene, temp_file


def start_q_from_robot_joints_deg(scene, joints_deg: Sequence[float]) -> List[float]:
    return joint_array_from_transformed_deg(scene, joints_deg)


def _target_pos(pos_m: Sequence[float]) -> np.ndarray:
    return np.asarray(pos_m, dtype=np.float64)[:3]


def _tcp_position_error_m_fn(tcp: TcpOffset):
    """Callback for ``plan_joint_trajectory`` terminal check (gizmo TCP FK)."""

    def _fn(scene, q: Sequence[float], target_pos: Sequence[float]) -> float:
        if not tcp.has_tcp(scene):
            return 0.0
        return tcp_error_mm(scene, target_pos, q, tcp) * 1e-3

    return _fn


def _report_pose_error(
    fk_pos: Sequence[float],
    fk_rpy: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
) -> tuple[float, float]:
    from trajectory_planner.scene import pose_error_rad

    pos_err_m, ori_err_rad = pose_error_rad(fk_pos, fk_rpy, target_pos, target_rpy)
    return pos_err_m * 1000.0, math.degrees(ori_err_rad)


def _ik_seed_q(start_q: Sequence[float], ik_warm_start_q: Sequence[float] | None) -> List[float]:
    if ik_warm_start_q is not None:
        return [float(v) for v in ik_warm_start_q]
    return [float(v) for v in start_q]


def plan_segment(
    scene,
    tcp: TcpOffset,
    start_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    *,
    enable_collision: bool,
    ik_ori_tol_rad: float = 0.35,
    refine_ori_tol_rad: float = 0.02,
    ik_pos_tol_m: float = 0.08,
    ik_warm_start_q: Sequence[float] | None = None,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    target_pos_arr = _target_pos(target_pos)
    ik_seed = _ik_seed_q(start_q, ik_warm_start_q)
    set_joint_positions(scene, ik_seed)
    goal_q = solve_goal_from_target_pose(
        scene,
        target_pos_arr,
        target_rpy,
        ik_seed,
        tcp,
        ik_ori_tol_rad=ik_ori_tol_rad,
        refine_ori_tol_rad=refine_ori_tol_rad,
        ik_pos_tol_m=ik_pos_tol_m,
    )
    if goal_q is None:
        return None, None

    tip_err_mm = tcp_error_mm(scene, target_pos_arr, goal_q, tcp)
    if tip_err_mm > ik_pos_tol_m * 1000.0:
        print(
            f"[grab_sensor] gizmo TCP IK rejected: tip_err={tip_err_mm:.2f} mm "
            f"(limit {ik_pos_tol_m * 1000:.1f} mm)"
        )
        return None, None

    pos_err, ori_err = measure_target_pose_error(scene, target_pos_arr, target_rpy, goal_q, tcp)
    if ori_err > ik_ori_tol_rad:
        print(
            f"[grab_sensor] gizmo TCP IK rejected: ori_err={math.degrees(ori_err):.1f} deg "
            f"(limit {math.degrees(ik_ori_tol_rad):.1f} deg)"
        )
        return None, None

    traj = plan_joint_trajectory(
        scene,
        start_q,
        goal_q,
        target_pos_arr,
        target_rpy,
        enable_collision_penalty=enable_collision,
        tcp_position_error_m=_tcp_position_error_m_fn(tcp),
    )
    return traj, goal_q


def solve_goal_position_first(
    scene,
    tcp: TcpOffset,
    target_pos: Sequence[float],
    start_q: Sequence[float],
    *,
    pos_tol_m: float = 0.05,
    max_pos_err_m: float = 0.08,
) -> Optional[List[float]]:
    """Reach ``target_pos`` in Cartesian space; orientation comes from FK at the solution."""
    q = refine_joint_config_to_target_position(scene, target_pos, start_q, tcp, pos_tol_m=pos_tol_m)
    if not is_state_valid(scene, q):
        return None
    pos_err_m = position_error_mm(scene, target_pos, q, tcp) * 1e-3
    if pos_err_m <= pos_tol_m:
        print(f"Position-first IK: pos_err={pos_err_m * 1000:.3f} mm")
        return q
    print(
        f"Position-first IK (best effort): pos_err={pos_err_m * 1000:.3f} mm "
        f"(limit {max_pos_err_m * 1000:.0f} mm)"
    )
    if pos_err_m <= max_pos_err_m:
        return q
    return None


def _solve_goal_from_position_seed(
    scene,
    tcp: TcpOffset,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    start_q: Sequence[float],
    *,
    pos_tol_m: float,
    max_pos_err_m: float,
    ik_ori_tol_rad: float,
    refine_ori_tol_rad: float,
    ik_warm_start_q: Sequence[float] | None = None,
) -> Optional[List[float]]:
    ik_seed = _ik_seed_q(start_q, ik_warm_start_q)
    set_joint_positions(scene, ik_seed)
    q_pos = solve_goal_position_first(
        scene, tcp, target_pos, ik_seed, pos_tol_m=pos_tol_m, max_pos_err_m=max_pos_err_m
    )
    if q_pos is None:
        return None

    q_pose = solve_goal_from_target_pose(
        scene,
        _target_pos(target_pos),
        target_rpy,
        q_pos,
        tcp,
        ik_ori_tol_rad=ik_ori_tol_rad,
        refine_ori_tol_rad=refine_ori_tol_rad,
        ik_pos_tol_m=max_pos_err_m,
    )
    if q_pose is not None:
        return q_pose

    return enforce_goal_orientation(
        scene,
        target_pos,
        target_rpy,
        q_pos,
        tcp,
        pos_tol_m=pos_tol_m,
        max_pos_err_m=max_pos_err_m,
        ori_tol_rad=ik_ori_tol_rad,
        refine_ori_tol_rad=refine_ori_tol_rad,
    )


def _goal_self_penetration_mm(scene, q: Sequence[float]) -> float:
    return float(evaluate_collision_metrics(scene, q).self_penetration_m) * 1000.0


def _goal_is_acceptable(
    scene,
    tcp: TcpOffset,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    q: Sequence[float],
    *,
    max_pos_err_m: float,
    ori_tol_rad: float,
    max_self_pen_mm: float = 1.0,
) -> bool:
    if not is_state_valid(scene, q):
        return False
    if _goal_self_penetration_mm(scene, q) > max_self_pen_mm:
        return False
    pos_err, ori_err = measure_target_pose_error(scene, target_pos, target_rpy, q, tcp)
    return pos_err <= max_pos_err_m and ori_err <= ori_tol_rad


def _plan_trajectory_to_goal(
    scene,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    *,
    tcp: TcpOffset,
    enable_collision: bool,
) -> Optional[np.ndarray]:
    traj = plan_joint_trajectory(
        scene,
        start_q,
        goal_q,
        _target_pos(target_pos),
        target_rpy,
        enable_collision_penalty=enable_collision,
        tcp_position_error_m=_tcp_position_error_m_fn(tcp),
    )
    return traj


def _plan_pose_segment_with_fallbacks(
    scene,
    tcp: TcpOffset,
    start_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    *,
    enable_collision: bool,
    segment_name: str,
    ik_ori_tol_rad: float = 0.35,
    refine_ori_tol_rad: float = 0.02,
    pos_tol_m: float = 0.05,
    max_pos_err_m: float = 0.08,
    cartesian_hold_rpy: Sequence[float] | None = None,
    position_first_fallback: bool = False,
    ik_warm_start_q: Sequence[float] | None = None,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    traj, goal_q = plan_segment(
        scene,
        tcp,
        start_q,
        target_pos,
        target_rpy,
        enable_collision=enable_collision,
        ik_ori_tol_rad=ik_ori_tol_rad,
        refine_ori_tol_rad=refine_ori_tol_rad,
        ik_pos_tol_m=max_pos_err_m,
        ik_warm_start_q=ik_warm_start_q,
    )
    if traj is not None and goal_q is not None:
        return traj, goal_q

    print(
        f"[grab_sensor] Full-pose IK failed for {segment_name}; "
        "retrying with Cartesian approach."
    )
    traj, goal_q = plan_cartesian_approach_segment(
        scene,
        tcp,
        start_q,
        target_pos,
        target_rpy,
        enable_collision=enable_collision,
        pos_tol_m=max_pos_err_m,
        ik_ori_tol_rad=ik_ori_tol_rad,
        hold_rpy=cartesian_hold_rpy,
    )
    if traj is not None and goal_q is not None:
        cartesian_goal_q = goal_q
        ik_seed = _ik_seed_q(cartesian_goal_q, ik_warm_start_q)
        set_joint_positions(scene, ik_seed)
        goal_q = solve_goal_from_target_pose(
            scene,
            _target_pos(target_pos),
            target_rpy,
            ik_seed,
            tcp,
            ik_ori_tol_rad=ik_ori_tol_rad,
            refine_ori_tol_rad=refine_ori_tol_rad,
            ik_pos_tol_m=max_pos_err_m,
        )
        if goal_q is None:
            goal_q = enforce_goal_orientation(
                scene,
                target_pos,
                target_rpy,
                cartesian_goal_q,
                tcp,
                pos_tol_m=pos_tol_m,
                max_pos_err_m=max_pos_err_m,
                ori_tol_rad=ik_ori_tol_rad,
                refine_ori_tol_rad=refine_ori_tol_rad,
            )
        if goal_q is not None:
            traj = _plan_trajectory_to_goal(
                scene,
                start_q,
                goal_q,
                target_pos,
                target_rpy,
                enable_collision=enable_collision,
            )
            if traj is not None:
                return traj, goal_q

    print(
        f"[grab_sensor] Cartesian {segment_name} approach failed; "
        f"retrying with position seed + orientation IK "
        f"(limit {math.degrees(ik_ori_tol_rad):.1f} deg)."
    )
    goal_q = _solve_goal_from_position_seed(
        scene,
        tcp,
        target_pos,
        target_rpy,
        start_q,
        pos_tol_m=pos_tol_m,
        max_pos_err_m=max_pos_err_m,
        ik_ori_tol_rad=ik_ori_tol_rad,
        refine_ori_tol_rad=refine_ori_tol_rad,
        ik_warm_start_q=ik_warm_start_q,
    )
    if goal_q is None:
        if not position_first_fallback:
            return None, None
        print(
            f"[grab_sensor] Position/orientation IK failed for {segment_name}; "
            "retrying position-first (orientation from FK)."
        )
        ik_seed = _ik_seed_q(start_q, ik_warm_start_q)
        set_joint_positions(scene, ik_seed)
        goal_q = solve_goal_position_first(
            scene,
            tcp,
            target_pos,
            ik_seed,
            pos_tol_m=pos_tol_m,
            max_pos_err_m=max_pos_err_m,
        )
        if goal_q is None:
            return None, None
        set_joint_positions(scene, goal_q)
        _, fk_rpy = get_end_link_pose(scene)
        traj = plan_joint_trajectory(
            scene,
            start_q,
            goal_q,
            _target_pos(target_pos),
            fk_rpy,
            enable_collision_penalty=enable_collision,
            tcp_position_error_m=_tcp_position_error_m_fn(tcp),
        )
        return (traj, goal_q) if traj is not None else (None, None)

    if not _goal_is_acceptable(
        scene,
        tcp,
        target_pos,
        target_rpy,
        goal_q,
        max_pos_err_m=max_pos_err_m,
        ori_tol_rad=ik_ori_tol_rad,
    ):
        pos_err, ori_err = measure_target_pose_error(scene, target_pos, target_rpy, goal_q, tcp)
        self_pen = _goal_self_penetration_mm(scene, goal_q)
        print(
            f"[grab_sensor] {segment_name} goal rejected: "
            f"pos_err={pos_err * 1000:.1f} mm, ori_err={math.degrees(ori_err):.1f} deg "
            f"(limit {math.degrees(ik_ori_tol_rad):.1f} deg), self_pen={self_pen:.2f} mm"
        )
        return None, None

    traj = _plan_trajectory_to_goal(
        scene,
        start_q,
        goal_q,
        target_pos,
        target_rpy,
        tcp=tcp,
        enable_collision=enable_collision,
    )
    if traj is None:
        return None, None
    return traj, goal_q


def plan_pre_grasp_segment(
    scene,
    cfg: GrabSensorConfig,
    start_q: Sequence[float],
    targets: GraspTargets,
    *,
    enable_collision: bool,
    pos_tol_m: float = 0.05,
    max_pos_err_m: float = 0.08,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """Pre-grasp: reach retreat pose; orientation tolerance is relaxed vs final grasp."""
    pre_ori_tol_rad = math.radians(float(cfg.approach.pre_grasp_orientation_tol_deg))
    refine_ori_tol_rad = min(pre_ori_tol_rad, math.radians(15.0))
    home_q = start_q_from_robot_joints_deg(scene, HOME_JOINTS_DEG)
    return _plan_pose_segment_with_fallbacks(
        scene,
        TcpOffset(),
        start_q,
        targets.pre_grasp_pos_m,
        targets.pre_grasp_rpy_rad,
        enable_collision=enable_collision,
        segment_name="pre-grasp",
        ik_ori_tol_rad=pre_ori_tol_rad,
        refine_ori_tol_rad=refine_ori_tol_rad,
        pos_tol_m=pos_tol_m,
        max_pos_err_m=max_pos_err_m,
        cartesian_hold_rpy=targets.pre_grasp_rpy_rad,
        position_first_fallback=True,
        ik_warm_start_q=home_q,
    )


def plan_cartesian_approach_segment(
    scene,
    tcp: TcpOffset,
    start_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    *,
    enable_collision: bool,
    n_steps: int = 24,
    pos_tol_m: float = 0.08,
    ik_ori_tol_rad: float = 0.78,
    hold_rpy: Sequence[float] | None = None,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """Straight-line approach in planner target frame (TCP when offset active)."""
    set_joint_positions(scene, start_q)
    approach_rpy = list(hold_rpy) if hold_rpy is not None else list(get_end_link_pose(scene)[1])
    target_pos_arr = _target_pos(target_pos)
    start_pos_arr, _ = target_pose_from_q(scene, start_q, tcp)

    q = list(start_q)
    waypoints: List[List[float]] = [list(start_q)]
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp_pos = (1.0 - alpha) * start_pos_arr + alpha * target_pos_arr
        q = refine_joint_config_to_target_pose(
            scene,
            interp_pos,
            approach_rpy,
            q,
            tcp,
            pos_tol_m=0.002,
            ori_tol_rad=ik_ori_tol_rad,
        )
        if not is_state_valid(scene, q):
            return None, None
        waypoints.append(list(q))

    grasp_goal_q = waypoints[-1]
    end_pos, _ = target_pose_from_q(scene, grasp_goal_q, tcp)
    pos_err = float(np.linalg.norm(np.asarray(end_pos, dtype=np.float64) - target_pos_arr))
    print(f"Final approach (Cartesian hold-orientation): pos_err={pos_err * 1000:.3f} mm")
    if pos_err > pos_tol_m:
        grasp_goal_q = refine_joint_config_to_target_position(
            scene, target_pos_arr, grasp_goal_q, tcp, pos_tol_m=pos_tol_m
        )
        if not is_state_valid(scene, grasp_goal_q):
            return None, None
        end_pos, _ = target_pose_from_q(scene, grasp_goal_q, tcp)
        pos_err = float(np.linalg.norm(np.asarray(end_pos, dtype=np.float64) - target_pos_arr))
        print(f"Final approach (position-only refine): pos_err={pos_err * 1000:.3f} mm")
        if pos_err > pos_tol_m:
            return None, None

    traj = plan_joint_trajectory(
        scene,
        start_q,
        grasp_goal_q,
        target_pos_arr,
        target_rpy,
        enable_collision_penalty=enable_collision,
        tcp_position_error_m=_tcp_position_error_m_fn(tcp),
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
    ik_ori_tol_rad: float = 0.35,
    max_pos_err_m: float = 0.08,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """Final grasp: TCP IK warm-started from ``start_q`` (pre-grasp), then iLQR."""
    tcp = TcpOffset()
    target_pos = _target_pos(targets.grasp_pos_m)
    target_rpy = targets.grasp_rpy_rad
    refine_ori_tol_rad = min(ik_ori_tol_rad, math.radians(5.0))

    set_joint_positions(scene, start_q)
    goal_q = solve_goal_from_target_pose(
        scene,
        target_pos,
        target_rpy,
        start_q,
        tcp,
        ik_ori_tol_rad=ik_ori_tol_rad,
        refine_ori_tol_rad=refine_ori_tol_rad,
        ik_pos_tol_m=max_pos_err_m,
    )
    if goal_q is None:
        home_q = start_q_from_robot_joints_deg(scene, HOME_JOINTS_DEG)
        print("[grab_sensor] Grasp TCP IK from pre-grasp failed; retrying warm-start from HOME.")
        goal_q = grasp_ik_goal_from_home(scene, targets, home_q, tcp)
    if goal_q is None:
        print("[grab_sensor] Final grasp TCP IK failed.")
        return None, None

    tip_err_mm = tcp_error_mm(scene, target_pos, goal_q, tcp)
    if tip_err_mm > max_pos_err_m * 1000.0:
        print(
            f"[grab_sensor] Final grasp TCP IK rejected: tip_err={tip_err_mm:.2f} mm "
            f"(limit {max_pos_err_m * 1000:.1f} mm)"
        )
        return None, None

    _, ori_err = measure_tcp_target_pose_error(scene, target_pos, target_rpy, goal_q, tcp)
    if ori_err > ik_ori_tol_rad:
        print(
            f"[grab_sensor] Final grasp TCP IK rejected: tcp_ori_err={math.degrees(ori_err):.1f} deg "
            f"(limit {math.degrees(ik_ori_tol_rad):.1f} deg)"
        )
        return None, None

    traj = plan_joint_trajectory(
        scene,
        start_q,
        goal_q,
        target_pos,
        target_rpy,
        enable_collision_penalty=enable_collision,
        tcp_position_error_m=_tcp_position_error_m_fn(tcp),
    )
    if traj is None:
        return None, None
    return traj, goal_q


def _post_grasp_lift_pos_m(targets: GraspTargets, lift_mm: float) -> np.ndarray:
    pos = np.asarray(targets.grasp_pos_m, dtype=np.float64)[:3].copy()
    pos[2] += float(lift_mm) * 1e-3
    return pos


def plan_vertical_lift_hold_orientation(
    scene,
    tcp: TcpOffset,
    start_q: Sequence[float],
    lift_delta_z_m: float,
    *,
    enable_collision: bool,
    n_steps: int = 16,
    pos_tol_m: float = 0.005,
    ori_tol_rad: float = 0.05,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """Pure +Z TCP move; TCP orientation locked to FK at ``start_q`` (grasp pose)."""
    import pybullet as p

    if lift_delta_z_m <= 0.0:
        return None, None

    set_joint_positions(scene, start_q)
    start_tip = tcp.tip_world(scene)
    hold_rpy = tcp_world_rpy_from_q(scene, start_q, tcp)
    hold_orn = p.getQuaternionFromEuler([float(v) for v in hold_rpy])

    target_tip = start_tip.copy()
    target_tip[2] += float(lift_delta_z_m)

    q = [float(v) for v in start_q]
    waypoints: List[List[float]] = [list(q)]
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp_tip = (1.0 - alpha) * start_tip + alpha * target_tip
        set_joint_positions(scene, q)
        try:
            q = solve_tcp_ik(scene, interp_tip, hold_orn, q)
        except (ValueError, TypeError):
            return None, None
        if not is_state_valid(scene, q):
            return None, None
        waypoints.append(list(q))

    goal_q = q
    pos_err_m, ori_err = measure_tcp_target_pose_error(scene, target_tip, hold_rpy, goal_q, tcp)
    print(
        f"[grab_sensor] vertical lift FK: tip_err={pos_err_m * 1000:.2f} mm, "
        f"ori_err={math.degrees(ori_err):.2f} deg (TCP orientation held from grasp)"
    )
    if pos_err_m > pos_tol_m * 4.0 or ori_err > ori_tol_rad:
        return None, None

    traj = plan_joint_trajectory(
        scene,
        start_q,
        goal_q,
        target_tip,
        hold_rpy,
        enable_collision_penalty=enable_collision,
        tcp_position_error_m=_tcp_position_error_m_fn(tcp),
    )
    if traj is None:
        raw = np.asarray(waypoints, dtype=np.float64)
        traj = resample_trajectory(raw, max(12, len(raw) * 3))
    return traj, goal_q


def plan_post_grasp_lift_segment(
    scene,
    cfg: GrabSensorConfig,
    start_q: Sequence[float],
    targets: GraspTargets,
    *,
    enable_collision: bool,
    max_pos_err_m: float = 0.08,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """Straight +Z lift after grasp; orientation locked to FK at grasp."""
    lift_mm = float(cfg.approach.post_grasp_lift_mm)
    if lift_mm <= 0.0:
        return None, None

    lift_pos = _post_grasp_lift_pos_m(targets, lift_mm)
    tcp = TcpOffset()

    print(
        f"[grab_sensor] post-grasp lift: +{lift_mm:.0f} mm along world +Z "
        f"to {lift_pos.round(4).tolist()} (orientation held from grasp FK)"
    )

    return plan_vertical_lift_hold_orientation(
        scene,
        tcp,
        start_q,
        lift_mm * 1e-3,
        enable_collision=enable_collision,
        pos_tol_m=max_pos_err_m,
    )


def plan_return_home_segment(
    scene,
    start_q: Sequence[float],
    *,
    enable_collision: bool,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """Joint-space iLQR from ``start_q`` back to HOME."""
    home_q = start_q_from_robot_joints_deg(scene, HOME_JOINTS_DEG)
    set_joint_positions(scene, home_q)
    home_pos, home_rpy = get_end_link_pose(scene)
    print("[grab_sensor] return HOME...")
    traj = plan_joint_trajectory(
        scene,
        start_q,
        home_q,
        home_pos,
        home_rpy,
        enable_collision_penalty=enable_collision,
    )
    if traj is None:
        return None, None
    return traj, home_q


def _finalize_motion_segment(
    trajectory: np.ndarray | None,
    start_q: Sequence[float],
    goal_q: Sequence[float],
) -> np.ndarray | None:
    if trajectory is None or len(trajectory) < 2:
        return None
    compressed = compress_duplicate_waypoints(np.asarray(trajectory, dtype=np.float64))
    return finalize_trajectory_for_cache(compressed, start_q, goal_q)


def _stack_motion_segments(segments: Sequence[np.ndarray | None]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for seg in segments:
        if seg is None or len(seg) < 1:
            continue
        if not parts:
            parts.append(np.asarray(seg, dtype=np.float64))
        else:
            parts.append(np.asarray(seg[1:], dtype=np.float64))
    if not parts:
        raise RuntimeError("No motion segments to combine.")
    return np.vstack(parts)


def plan_approach_and_grasp(
    cfg: GrabSensorConfig,
    start_joints_deg: Sequence[float],
    targets: GraspTargets,
    *,
    use_gui: bool = False,
) -> Tuple[np.ndarray, dict]:
    del use_gui  # planning always runs DIRECT; use playback_grasp_plan_in_gui for GUI
    scene, temp_file = build_scene(cfg, use_gui=False)
    meta: dict = {}
    try:
        start_q = start_q_from_robot_joints_deg(scene, start_joints_deg)

        prepare_fixed_robot_grasp_targets(cfg, targets)
        targets.debug["pre_grasp_rpy_deg"] = [
            float(math.degrees(v)) for v in targets.pre_grasp_rpy_rad
        ]
        if (rpy := targets.debug.get("grasp_rpy_world_deg")) is not None and (
            targets.debug.get("grasp_position_source") != "table_base_xy_height"
        ):
            abc = targets.debug.get("grasp_abc_input_deg")
            abc_s = f"  abc={[round(float(v), 2) for v in abc]}" if abc else ""
            print(f"[grab_sensor] grasp world RPY target (deg): {[round(float(v), 2) for v in rpy]}{abc_s}")
            if (heading := targets.debug.get("grasp_heading_deg")) is not None and abc is not None:
                print(
                    f"[grab_sensor] yaw = {cfg.grasp_rpy_yaw_base_deg:.0f} + c - atan2(x,y) "
                    f"({float(abc[2]):.0f}+{cfg.grasp_rpy_yaw_base_deg:.0f}-{heading:.2f}) = {rpy[2]:.2f} deg"
                )

        tcp = TcpOffset()
        ori_tol_rad = math.radians(float(cfg.grasp_orientation_tol_deg))
        refine_ori_tol_rad = min(ori_tol_rad, math.radians(5.0))
        pos_tol_m = float(cfg.grasp_position_tol_mm) * 1e-3 * 0.625
        max_pos_err_m = float(cfg.grasp_position_tol_mm) * 1e-3

        pre_traj = None
        pre_goal_q = start_q
        if cfg.approach.plan_pre_grasp:
            pre_traj, pre_goal_q = plan_pre_grasp_segment(
                scene,
                cfg,
                start_q,
                targets,
                enable_collision=cfg.enable_collision,
                pos_tol_m=pos_tol_m,
                max_pos_err_m=max_pos_err_m,
            )
            if pre_traj is None or pre_goal_q is None:
                raise RuntimeError(
                    "Failed to plan pre-grasp segment (IK or iLQR). "
                    "Typical causes: uncalibrated camera_to_robot in grasp config, "
                    "grasp target out of reach, or wrong depth segmentation band."
                )

            set_joint_positions(scene, pre_goal_q)
            pre_fk_pos, pre_fk_rpy = target_pose_from_q(scene, pre_goal_q, tcp)
            pre_pos_err_mm, pre_ori_err_deg = _report_pose_error(
                pre_fk_pos,
                pre_fk_rpy,
                targets.pre_grasp_pos_m,
                targets.pre_grasp_rpy_rad,
            )
            print(
                f"[grab_sensor] planned pre-grasp (table): "
                f"{robot_point_table_display(pre_fk_pos, cfg.robot_in_table_frame)}  "
                f"(FK err {pre_pos_err_mm:.1f} mm, ori {pre_ori_err_deg:.1f} deg)"
            )
            meta["pre_grasp_fk_pos_err_mm"] = pre_pos_err_mm
            meta["pre_grasp_fk_ori_err_deg"] = pre_ori_err_deg
        else:
            print("[grab_sensor] pre-grasp skipped (plan_pre_grasp: false); planning HOME -> grasp only.")

        grasp_traj, grasp_goal_q = plan_grasp_segment(
            scene,
            pre_goal_q,
            targets,
            enable_collision=cfg.enable_collision,
            ik_ori_tol_rad=ori_tol_rad,
            max_pos_err_m=max_pos_err_m,
        )
        if grasp_traj is None or grasp_goal_q is None:
            raise RuntimeError(
                "Failed to plan final grasp segment (IK or iLQR). "
                "Typical causes: grasp target out of reach, or fixed_grasp_abc_input_deg "
                "incompatible with the arm at the taught grasp location."
            )

        pre_seg = (
            _finalize_motion_segment(pre_traj, start_q, pre_goal_q)
            if pre_traj is not None
            else None
        )
        grasp_seg = _finalize_motion_segment(grasp_traj, pre_goal_q, grasp_goal_q)
        if grasp_seg is None:
            raise RuntimeError("Failed to finalize grasp motion segment.")

        lift_traj = None
        lift_goal_q = grasp_goal_q
        lift_seg = None
        if float(cfg.approach.post_grasp_lift_mm) > 0.0:
            lift_traj, lift_goal_q = plan_post_grasp_lift_segment(
                scene,
                cfg,
                grasp_goal_q,
                targets,
                enable_collision=cfg.enable_collision,
                max_pos_err_m=max_pos_err_m,
            )
            if lift_traj is None or lift_goal_q is None:
                raise RuntimeError(
                    "Failed to plan post-grasp lift segment (+Z retreat after grasp)."
                )
            lift_seg = _finalize_motion_segment(lift_traj, grasp_goal_q, lift_goal_q)
            set_joint_positions(scene, lift_goal_q)
            lift_fk_pos, _ = target_pose_from_q(scene, lift_goal_q, tcp)
            lift_err_mm = float(
                np.linalg.norm(lift_fk_pos - _post_grasp_lift_pos_m(targets, cfg.approach.post_grasp_lift_mm))
            ) * 1000.0
            print(f"[grab_sensor] post-grasp lift FK: pos_err={lift_err_mm:.1f} mm")
            meta["post_lift_fk_pos_err_mm"] = lift_err_mm
            meta["lift_goal_q"] = lift_goal_q

        home_traj = None
        final_goal_q = lift_goal_q
        home_seg = None
        if cfg.approach.plan_return_home:
            home_traj, home_goal_q = plan_return_home_segment(
                scene,
                lift_goal_q,
                enable_collision=cfg.enable_collision,
            )
            if home_traj is None or home_goal_q is None:
                raise RuntimeError("Failed to plan return HOME segment after grasp lift.")
            home_seg = _finalize_motion_segment(home_traj, lift_goal_q, home_goal_q)
            final_goal_q = home_goal_q
            meta["home_goal_q"] = home_goal_q

        segments = {
            "pre_grasp": pre_seg,
            "grasp": grasp_seg,
            "lift": lift_seg,
            "home": home_seg,
        }
        meta["segments"] = segments
        combined = _stack_motion_segments([pre_seg, grasp_seg, lift_seg, home_seg])
        meta["n_through_grasp"] = len(_stack_motion_segments([pre_seg, grasp_seg]))

        set_joint_positions(scene, grasp_goal_q)
        fk_points = fk_grasp_points(scene, grasp_goal_q, tcp)
        fk_tip_arr = fk_points[0] if fk_points is not None else None
        fk_pos_arr = fk_points[1] if fk_points is not None else None
        tip_target = targets.debug.get("grasp_tip_pos_m")
        tip_err_mm = (
            float(np.linalg.norm(fk_tip_arr - np.asarray(tip_target, dtype=np.float64))) * 1000.0
            if fk_tip_arr is not None and tip_target is not None
            else None
        )
        pos_err_mm = (
            tip_err_mm
            if tip_err_mm is not None
            else float(np.linalg.norm(fk_pos_arr - targets.grasp_pos_m)) * 1000.0
            if fk_pos_arr is not None
            else 0.0
        )
        wrist_tcp_mm = wrist_tcp_distance_mm(scene, grasp_goal_q, tcp)

        meta.update(
            {
                "n_pre": len(pre_seg) if pre_seg is not None else 0,
                "plan_pre_grasp": cfg.approach.plan_pre_grasp,
                "n_grasp": len(grasp_seg),
                "n_lift": len(lift_seg) if lift_seg is not None else 0,
                "n_home": len(home_seg) if home_seg is not None else 0,
                "post_grasp_lift_mm": float(cfg.approach.post_grasp_lift_mm),
                "plan_return_home": cfg.approach.plan_return_home,
                "n_total": len(combined),
                "start_q": start_q,
                "pre_goal_q": pre_goal_q,
                "grasp_goal_q": grasp_goal_q,
                "grasp_fk_m": fk_pos_arr.round(4).tolist() if fk_pos_arr is not None else None,
                "grasp_fk_pos_err_mm": pos_err_mm,
                "wrist_tcp_mm": wrist_tcp_mm,
            }
        )
        print(
            f"[grab_sensor] motion segments: pre={meta['n_pre']} grasp={meta['n_grasp']} "
            f"lift={meta['n_lift']} home={meta['n_home']} total={meta['n_total']}"
        )
        report_err_mm = tip_err_mm if tip_err_mm is not None else pos_err_mm
        if fk_tip_arr is not None and tip_err_mm is not None:
            meta["grasp_fk_tip_m"] = fk_tip_arr.round(4).tolist()
            meta["grasp_fk_tip_err_mm"] = tip_err_mm

        planned_tip = fk_tip_arr if fk_tip_arr is not None else fk_pos_arr
        if planned_tip is not None:
            print(
                f"[grab_sensor] planned grasp (table): "
                f"{robot_point_table_display(planned_tip, cfg.robot_in_table_frame)}  "
                f"(FK err {report_err_mm:.1f} mm)"
            )

        fk_tcp_rpy = tcp_world_rpy_from_q(scene, grasp_goal_q, tcp)
        target_rpy_deg = [round(math.degrees(v), 2) for v in targets.grasp_rpy_rad]
        planned_tcp_rpy_deg = [round(math.degrees(v), 2) for v in fk_tcp_rpy]
        _, grasp_ori_err = measure_target_pose_error(
            scene, targets.grasp_pos_m, targets.grasp_rpy_rad, grasp_goal_q, tcp
        )
        meta["grasp_fk_tcp_rpy_world_deg"] = [float(v) for v in planned_tcp_rpy_deg]
        meta["grasp_target_rpy_world_deg"] = [float(v) for v in target_rpy_deg]
        meta["grasp_fk_ori_err_deg"] = float(math.degrees(grasp_ori_err))
        if report_err_mm > 20.0:
            print(
                f"[grab_sensor] WARNING: arm FK misses grasp target by {report_err_mm:.1f} mm "
                f"(tolerance {cfg.grasp_position_tol_mm:.0f} mm). "
                "Check arm reach, HOME start pose, or orientation."
            )
        return combined, meta
    finally:
        disconnect_planning_scene(scene)
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
