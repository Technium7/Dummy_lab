"""PyBullet debug markers for grasp TCP alignment (--ik-gui / --plan-gui)."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from trajectory_planner.scene import (
    disconnect_planning_scene,
    frame_camera_on_scene,
    set_joint_positions,
)

from .models import GrabSensorConfig, GraspTargets
from .tcp_ik import (
    TcpOffset,
    fk_grasp_points,
    link_pose_from_q,
    measure_target_pose_error,
    tcp_error_mm,
    wrist_tcp_distance_mm,
)

_CROSS_HALF_M = 0.012


def _planner_tip_target(cfg: GrabSensorConfig, targets: GraspTargets) -> np.ndarray | None:
    tip_raw = targets.debug.get("grasp_tip_pos_m")
    if tip_raw is not None:
        return np.asarray(tip_raw, dtype=np.float64)
    return np.asarray(targets.grasp_pos_m, dtype=np.float64)


def _sync_scene_kinematics(scene, q: Sequence[float]) -> None:
    import pybullet as p

    set_joint_positions(scene, q)
    p.performCollisionDetection(physicsClientId=scene.client_id)
    p.stepSimulation(physicsClientId=scene.client_id)


def _draw_attached_cross(
    scene,
    tcp: TcpOffset,
    local_m: Sequence[float],
    color: Sequence[float],
    *,
    attach: str = "tcp",
    half_m: float = _CROSS_HALF_M,
) -> None:
    import pybullet as p

    kw = tcp.tcp_attach_kwargs(scene) if attach == "tcp" else tcp.wrist_attach_kwargs(scene)
    center = [float(local_m[0]), float(local_m[1]), float(local_m[2])]
    c = [float(v) for v in color[:3]]
    for axis_i in (0, 1):
        start = center.copy()
        end = center.copy()
        start[axis_i] -= half_m
        end[axis_i] += half_m
        p.addUserDebugLine(start, end, c, 3.0, 0, **kw)


def _draw_tcp_on_arm(scene, tcp: TcpOffset) -> None:
    """Yellow cross on ``grasp_tcp_link``."""
    if not tcp.has_tcp(scene):
        return
    _draw_attached_cross(scene, tcp, (0.0, 0.0, 0.0), [1.0, 1.0, 0.0])


def _draw_world_cross(
    pos: Sequence[float],
    *,
    label: str,
    color: Sequence[float],
    half_m: float = 0.015,
) -> None:
    import pybullet as p

    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    c = [float(v) for v in color[:3]]
    p.addUserDebugLine([x - half_m, y, z], [x + half_m, y, z], c, 3.0, 0)
    p.addUserDebugLine([x, y - half_m, z], [x, y + half_m, z], c, 3.0, 0)
    p.addUserDebugText(label, [x, y, z + 0.012], textColorRGB=c, textSize=1.1, lifeTime=0)


def _draw_world_rpy_axes(
    pos: Sequence[float],
    rpy_rad: Sequence[float],
    color: Sequence[float],
    *,
    half_m: float = 0.03,
) -> None:
    """Short local X/Y axes in world frame (orientation hint)."""
    import pybullet as p

    quat = p.getQuaternionFromEuler(list(rpy_rad))
    origin = [float(pos[0]), float(pos[1]), float(pos[2])]
    c = [float(v) for v in color[:3]]
    for local_axis, scale in (([half_m, 0.0, 0.0], 1.0), ([0.0, half_m, 0.0], 0.7)):
        end, _ = p.multiplyTransforms(origin, quat, local_axis, [0.0, 0.0, 0.0, 1.0])
        p.addUserDebugLine(origin, end, [c[0] * scale, c[1] * scale, c[2] * scale], 2.5, 0)


def _print_ik_goal_summary(
    cfg: GrabSensorConfig,
    targets: GraspTargets,
    goal_q: Sequence[float],
    scene,
) -> float:
    tcp = TcpOffset()
    _sync_scene_kinematics(scene, goal_q)
    planned = _planner_tip_target(cfg, targets)
    tip_world = tcp.tip_world(scene)
    wrist_world = tcp.wrist_world(scene)
    gap_mm = (
        float(np.linalg.norm(tip_world - planned)) * 1000.0 if planned is not None else float("nan")
    )
    tcp_name = scene.tcp_link_name or "offset"
    _, ori_err = measure_target_pose_error(
        scene, targets.grasp_pos_m, targets.grasp_rpy_rad, goal_q, tcp
    )

    print("[grab_sensor] IK goal (robot world m, sim rad joints):")
    print(f"  planned:   {planned.round(6).tolist() if planned is not None else None}")
    print(f"  TCP FK:    {tip_world.round(6).tolist()}  ({tcp_name})")
    print(f"  wrist:     {wrist_world.round(6).tolist()}  (link6_1)")
    print(f"  TCP err:   {gap_mm:.3f} mm")
    print(f"  ori err:   {np.degrees(ori_err):.2f} deg")
    return gap_mm


def show_ik_goal_debug_gui(
    cfg: GrabSensorConfig,
    targets: GraspTargets,
    goal_q: Sequence[float],
) -> None:
    """Arm at IK goal: planned target (world) vs TCP on ``grasp_tcp_link``."""
    import pybullet as p

    from .plan_motion import build_scene

    scene, temp_file = build_scene(cfg, use_gui=True)
    tcp = TcpOffset()
    goal_q = [float(v) for v in goal_q]
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.removeAllUserDebugItems()
        frame_camera_on_scene(scene)
        _sync_scene_kinematics(scene, goal_q)

        planned = _planner_tip_target(cfg, targets)
        if planned is not None:
            _draw_world_cross(planned, label="planned", color=[1.0, 1.0, 0.0])
        _draw_tcp_on_arm(scene, tcp)
        _draw_attached_cross(scene, tcp, (0.0, 0.0, 0.0), [1.0, 0.2, 0.2], attach="wrist")
        wrist_world = tcp.wrist_world(scene)
        _, actual_rpy, _ = link_pose_from_q(scene, goal_q)
        _draw_world_rpy_axes(wrist_world, targets.grasp_rpy_rad, [1.0, 1.0, 0.0])
        _draw_world_rpy_axes(wrist_world, actual_rpy, [1.0, 0.2, 0.2])

        gap_mm = _print_ik_goal_summary(cfg, targets, goal_q, scene)
        print(
            f"[grab_sensor] IK GUI: yellow=planned tip, yellow axes=planned wrist ori, "
            f"red axes=actual wrist ori. TCP err {gap_mm:.2f} mm. Close window to exit."
        )

        dt = 1.0 / 30.0
        while p.isConnected():
            _sync_scene_kinematics(scene, goal_q)
            time.sleep(dt)
    finally:
        disconnect_planning_scene(scene)
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


def draw_goal_alignment(
    cfg: GrabSensorConfig,
    targets: GraspTargets,
    *,
    goal_q: Sequence[float],
    scene,
) -> None:
    """At goal joints: planner target vs live TCP FK."""
    import pybullet as p

    tip_tgt = _planner_tip_target(cfg, targets)
    tcp = TcpOffset()
    set_joint_positions(scene, goal_q)
    _draw_tcp_on_arm(scene, tcp)
    tip_fk, _ = fk_grasp_points(scene, goal_q, tcp) or (None, None)

    if tip_tgt is not None:
        _draw_world_cross(tip_tgt, label="planned", color=[1.0, 1.0, 0.0])

    if tip_fk is not None and tip_tgt is not None:
        err_mm = tcp_error_mm(scene, tip_tgt, goal_q, tcp)
        mid = 0.5 * (tip_tgt + tip_fk)
        p.addUserDebugLine(list(tip_fk), list(tip_tgt), [0.2, 0.95, 0.35], 3.0, 0)
        p.addUserDebugText(
            f"TCP err {err_mm:.1f} mm",
            [float(mid[0]), float(mid[1]), float(mid[2]) + 0.015],
            textColorRGB=[0.2, 0.95, 0.35],
            textSize=1.1,
            lifeTime=0,
        )
        print(
            f"[grab_sensor] goal: TCP FK={tip_fk.round(4).tolist()}  "
            f"planned={tip_tgt.round(4).tolist()}  err={err_mm:.2f} mm"
        )


def playback_grasp_plan_in_gui(
    cfg: GrabSensorConfig,
    trajectory: np.ndarray,
    targets: GraspTargets,
) -> None:
    """PyBullet GUI: play path, then show TCP vs planner target at goal."""
    import pybullet as p

    from .plan_motion import build_scene

    scene, temp_file = build_scene(cfg, use_gui=True)
    tcp = TcpOffset()
    goal_q = trajectory[-1]
    dt = 1.0 / 40.0
    traj = np.asarray(trajectory, dtype=np.float64)

    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        frame_camera_on_scene(scene)
        p.setGravity(0.0, 0.0, 0.0)
        p.setRealTimeSimulation(0)
        set_joint_positions(scene, traj[0])
        _draw_tcp_on_arm(scene, tcp)
        time.sleep(0.5)

        for i in range(len(traj) - 1):
            if not p.isConnected():
                break
            q0, q1 = traj[i], traj[i + 1]
            n_sub = max(1, int(np.ceil(float(np.max(np.abs(q1 - q0))) / 0.012)))
            for k in range(1, n_sub + 1):
                if not p.isConnected():
                    break
                t = k / n_sub
                set_joint_positions(scene, (1.0 - t) * q0 + t * q1)
                time.sleep(dt)

        set_joint_positions(scene, goal_q)
        draw_goal_alignment(cfg, targets, goal_q=goal_q, scene=scene)
        wrist_tcp_mm = wrist_tcp_distance_mm(scene, goal_q, tcp)
        print(f"[grab_sensor] wrist→TCP distance at goal={wrist_tcp_mm:.1f} mm (URDF FK)")

        while p.isConnected():
            set_joint_positions(scene, goal_q)
            time.sleep(dt)
    finally:
        disconnect_planning_scene(scene)
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
