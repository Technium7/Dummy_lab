from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from manual_control.mapping import HOME_JOINTS_DEG, clamp_firmware_joints_deg  # noqa: E402
from robot_ascii_interface import RobotAsciiInterface  # noqa: E402
from trajectory_run.run_planned_trajectory import (  # noqa: E402
    firmware_home,
    prepare_robot_trajectory,
    read_physical_joints,
    sim_trajectory_to_robot_deg,
    stream_waypoints_deg,
    wait_until_joints,
)

from .models import GrabSensorConfig, GraspTargets


def ensure_robot_ready(
    robot: RobotAsciiInterface,
    *,
    cmd_mode: int = 2,
    home_first: bool = True,
    open_hand: bool = False,
    label: str = "",
) -> bool:
    print(f"{label}HAND_EN -> {robot.hand_enable()}")
    time.sleep(0.1)
    if open_hand:
        robot.hand_open()
        time.sleep(0.3)

    if not home_first:
        print(f"{label}!START -> {robot.start()}")
        time.sleep(0.2)
        print(f"{label}#CMDMODE {cmd_mode} -> {robot.set_command_mode(cmd_mode)}")
        time.sleep(0.2)
        return True

    ok, at_home = firmware_home(
        robot,
        HOME_JOINTS_DEG,
        cmd_mode=cmd_mode,
        label=label,
    )
    if not ok:
        return False

    # Hold HOME in stream mode before the first approach segment (avoids HOME→stream jerk).
    hold_q = at_home if at_home is not None else read_physical_joints(robot)
    if hold_q is not None:
        from trajectory_run.run_planned_trajectory import _hold_stream_speed

        hold_q = clamp_firmware_joints_deg(hold_q)
        robot.hold_stream_pose(hold_q, _hold_stream_speed())
        time.sleep(0.3)
    return True


def execute_grasp_trajectory(
    robot: RobotAsciiInterface,
    trajectory_rad: np.ndarray,
    *,
    smooth: bool = True,
    send_interval_ms: float = 120.0,
    speed_scale: float = 0.85,
    label: str = "",
    segment: str = "",
    arm: bool = True,
    seed_from_robot: bool = True,
    sync_start_to_robot: bool = False,
    ramp_speed_scale: float | None = None,
) -> bool:
    traj = np.asarray(trajectory_rad, dtype=np.float64)
    if traj.size == 0:
        return True
    waypoints_deg, velocities_deg, info = prepare_robot_trajectory(
        traj,
        smooth=smooth,
        sample_dt_s=send_interval_ms / 1000.0,
        max_vel_deg_s=25.0,
        max_acc_deg_s2=80.0,
    )
    if sync_start_to_robot:
        current = read_physical_joints(robot)
        if current is not None:
            waypoints_deg = np.asarray(waypoints_deg, dtype=np.float64).copy()
            waypoints_deg[0] = clamp_firmware_joints_deg(current)
            if velocities_deg is not None and len(velocities_deg) > 0:
                velocities_deg = np.asarray(velocities_deg, dtype=np.float64).copy()
                velocities_deg[0] = 0.0
    seg = f" ({segment})" if segment else ""
    print(f"{label}Prepared {len(waypoints_deg)} robot waypoints{seg} ({info}).")
    planned_duration_s = info.get("duration_s") if info.get("smoothed") else None
    stream_scale = float(speed_scale)
    if ramp_speed_scale is not None and ramp_speed_scale < stream_scale:
        stream_scale = float(ramp_speed_scale)
    return stream_waypoints_deg(
        robot,
        waypoints_deg,
        velocities_deg_s=velocities_deg,
        send_interval_ms=send_interval_ms,
        planned_duration_s=planned_duration_s,
        speed_scale=stream_scale,
        label=label,
        arm=arm,
        seed_from_robot=seed_from_robot,
    )


def open_gripper(
    robot: RobotAsciiInterface,
    *,
    label: str = "",
    settle_s: float = 0.35,
) -> None:
    print(f"{label}Opening gripper at pre-grasp...")
    robot.hand_enable()
    time.sleep(0.1)
    robot.hand_open()
    time.sleep(settle_s)


def close_gripper(
    robot: RobotAsciiInterface,
    *,
    label: str = "",
    settle_s: float = 0.5,
) -> None:
    print(f"{label}Closing gripper at grasp...")
    robot.hand_close()
    time.sleep(settle_s)


def _populate_pre_grasp_joints_deg(exec_params: dict[str, Any], meta: Mapping[str, Any]) -> None:
    if exec_params.get("pre_grasp_joints_deg") is not None:
        return
    pre_goal_q = meta.get("pre_goal_q")
    if pre_goal_q is None:
        return
    pre_deg = sim_trajectory_to_robot_deg(np.asarray([pre_goal_q], dtype=np.float64))[0]
    exec_params["pre_grasp_joints_deg"] = pre_deg.tolist()


def retreat_after_close_if_needed(
    robot: RobotAsciiInterface,
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> None:
    if cfg.approach.plan_return_home or float(cfg.approach.post_grasp_lift_mm) > 0.0:
        return
    if not bool(params.get("retreat_after_close", True)):
        return

    pre_deg = params.get("pre_grasp_joints_deg")
    if pre_deg is None:
        print(f"{label}No pre-grasp joints for retreat; staying at grasp pose.")
        return

    q = [float(v) for v in pre_deg[:6]]
    print(f"{label}Retreat to pre-grasp joints...")
    stream_waypoints_deg(
        robot,
        np.asarray([q, q], dtype=np.float64),
        send_interval_ms=float(params.get("send_interval_ms", 120.0)),
        speed_scale=float(params.get("speed_scale", 0.85)),
        label=label,
        arm=False,
        seed_from_robot=False,
    )


def close_gripper_and_retreat(
    robot: RobotAsciiInterface,
    targets: GraspTargets,
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> None:
    del targets  # kept for future Cartesian retreat
    close_gripper(robot, label=label)
    retreat_after_close_if_needed(robot, cfg, params, label=label)


def _home_joint_error_deg(robot: RobotAsciiInterface) -> float | None:
    current = read_physical_joints(robot)
    if current is None:
        return None
    target = clamp_firmware_joints_deg(HOME_JOINTS_DEG)
    return float(np.max(np.abs(np.asarray(current[:6]) - np.asarray(target))))


def _settle_return_home_stream(
    robot: RobotAsciiInterface,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> None:
    """Stream settle only checks joint stability, not distance to HOME — wait for the goal."""
    err_before = _home_joint_error_deg(robot)
    if err_before is not None:
        print(f"{label}After return stream: max |Δ| from HOME = {err_before:.2f} deg")
    if err_before is not None and err_before <= float(params.get("home_stream_goal_tol_deg", 2.0)):
        return

    timeout = float(params.get("home_stream_settle_timeout_sec", 90.0))
    tol = float(params.get("home_stream_goal_tol_deg", 2.0))
    print(f"{label}Waiting for streamed return to reach HOME (up to {timeout:.0f} s)...")
    ok, _ = wait_until_joints(
        robot,
        HOME_JOINTS_DEG,
        timeout_sec=timeout,
        tol_deg=tol,
        label=label,
        name="streamed return HOME",
    )
    if not ok:
        err = _home_joint_error_deg(robot)
        if err is not None:
            print(
                f"{label}Warning: streamed return still {err:.2f} deg from HOME; "
                "will run !HOME."
            )


def _confirm_return_home(
    robot: RobotAsciiInterface,
    *,
    cmd_mode: int,
    params: Mapping[str, Any],
    label: str = "",
) -> None:
    """Streamed return HOME is approximate; finish with firmware HOME like startup."""
    settle = float(params.get("home_settle_timeout_sec", 120.0))
    print(f"{label}Confirming firmware HOME...")
    ok, final = firmware_home(
        robot,
        HOME_JOINTS_DEG,
        cmd_mode=cmd_mode,
        settle_timeout_sec=settle,
        label=label,
    )
    if not ok:
        print(
            f"{label}Warning: return trajectory finished but firmware HOME confirm failed. "
            "Joints may not match HOME."
        )
        return
    if final is not None:
        err = float(np.max(np.abs(np.asarray(final[:6]) - np.asarray(HOME_JOINTS_DEG))))
        print(f"{label}At HOME (max |Δ|={err:.3f} deg).")


def _stream_segment_or_raise(
    robot: RobotAsciiInterface,
    trajectory_rad: np.ndarray,
    *,
    segment: str,
    stream_kw: Mapping[str, Any],
    arm: bool = True,
    seed_from_robot: bool = True,
    sync_start_to_robot: bool = False,
    ramp_speed_scale: float | None = None,
) -> None:
    if not execute_grasp_trajectory(
        robot,
        trajectory_rad,
        segment=segment,
        arm=arm,
        seed_from_robot=seed_from_robot,
        sync_start_to_robot=sync_start_to_robot,
        ramp_speed_scale=ramp_speed_scale,
        **stream_kw,
    ):
        raise RuntimeError(f"{segment} trajectory streaming failed.")


def _motion_segments(meta: Mapping[str, Any]) -> dict[str, np.ndarray | None]:
    raw = meta.get("segments")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, np.ndarray | None] = {}
    for key in ("pre_grasp", "grasp", "lift", "home"):
        seg = raw.get(key)
        if seg is None:
            out[key] = None
        else:
            arr = np.asarray(seg, dtype=np.float64)
            out[key] = arr if arr.ndim == 2 and len(arr) >= 2 else None
    return out


def _legacy_segment_slices(
    trajectory: np.ndarray,
    meta: Mapping[str, Any],
) -> dict[str, np.ndarray | None]:
    n_total = int(trajectory.shape[0])
    n_pre = int(meta.get("n_pre") or 0)
    n_grasp_end = meta.get("n_through_grasp")
    n_grasp_end = int(n_grasp_end) if n_grasp_end is not None else n_total
    n_grasp_end = max(1, min(n_grasp_end, n_total))

    pre_seg = trajectory[:n_pre] if n_pre > 0 else None
    grasp_start = max(0, n_pre - 1) if n_pre > 0 else 0
    grasp_seg = trajectory[grasp_start:n_grasp_end] if n_grasp_end > grasp_start else None
    post_seg = trajectory[n_grasp_end - 1 :] if n_grasp_end < n_total else None
    return {
        "pre_grasp": pre_seg,
        "grasp": grasp_seg,
        "lift": post_seg,
        "home": None,
    }


def execute_on_robot(
    robot: RobotAsciiInterface,
    trajectory_rad: np.ndarray,
    targets: GraspTargets,
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> None:
    trajectory = np.asarray(trajectory_rad, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < 1:
        raise ValueError("execute_on_robot requires a non-empty joint trajectory.")

    exec_params = dict(params)
    meta = exec_params.get("plan_meta") or {}
    _populate_pre_grasp_joints_deg(exec_params, meta)

    segments = _motion_segments(meta)
    if not segments or segments.get("grasp") is None:
        segments = _legacy_segment_slices(trajectory, meta)

    pre_seg = segments.get("pre_grasp")
    grasp_seg = segments.get("grasp")
    lift_seg = segments.get("lift")
    home_seg = segments.get("home")

    open_gripper_at_pre = bool(params.get("open_gripper", True))
    close_at_grasp = bool(params.get("close_gripper", True))

    cmd_mode = int(params.get("cmd_mode", 2))
    home_first = bool(params.get("home_first", True))
    if not ensure_robot_ready(
        robot,
        cmd_mode=cmd_mode,
        home_first=home_first,
        open_hand=False,
        label=label,
    ):
        raise RuntimeError("Failed to reach HOME before grasp execution.")

    stream_kw = {
        "smooth": bool(params.get("smooth", True)),
        "send_interval_ms": float(params.get("send_interval_ms", 120.0)),
        "speed_scale": float(params.get("speed_scale", 0.85)),
        "label": label,
    }

    armed = home_first

    # Phase 1: HOME -> pre-grasp (already homed + held; no re-arm or seed jerk)
    if pre_seg is not None:
        print(f"{label}Phase 1: HOME -> pre-grasp ({len(pre_seg)} planner samples).")
        ramp = float(params.get("approach_speed_scale", 0.65))
        _stream_segment_or_raise(
            robot,
            pre_seg,
            segment="HOME -> pre-grasp",
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
            sync_start_to_robot=True,
            ramp_speed_scale=min(ramp, float(stream_kw["speed_scale"])),
        )
        armed = True
        if open_gripper_at_pre:
            open_gripper(robot, label=label)

    # Phase 2: pre-grasp -> grasp (gripper open)
    if grasp_seg is not None:
        phase = "pre-grasp -> grasp" if pre_seg is not None else "HOME -> grasp"
        if open_gripper_at_pre and pre_seg is None:
            open_gripper(robot, label=label)
        print(f"{label}Phase 2: {phase} ({len(grasp_seg)} planner samples).")
        _stream_segment_or_raise(
            robot,
            grasp_seg,
            segment=phase,
            stream_kw=stream_kw,
            arm=not armed,
            seed_from_robot=not armed,
        )
        armed = True
    elif open_gripper_at_pre and pre_seg is None:
        open_gripper(robot, label=label)

    if close_at_grasp:
        close_gripper(robot, label=label)

    # Phase 3: lift (gripper closed, object in hand)
    if lift_seg is not None:
        print(f"{label}Phase 3: post-grasp lift ({len(lift_seg)} planner samples).")
        _stream_segment_or_raise(
            robot,
            lift_seg,
            segment="post-grasp lift",
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
        )

    # Phase 4: return HOME (stream approximates path; firmware HOME confirms at end)
    return_home = bool(meta.get("plan_return_home", cfg.approach.plan_return_home))
    if home_seg is not None:
        print(f"{label}Phase 4: return HOME ({len(home_seg)} planner samples).")
        home_ramp = params.get("return_home_speed_scale")
        _stream_segment_or_raise(
            robot,
            home_seg,
            segment="return HOME",
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
            sync_start_to_robot=True,
            ramp_speed_scale=float(home_ramp) if home_ramp is not None else None,
        )
        _settle_return_home_stream(robot, params, label=label)
    elif close_at_grasp and lift_seg is None:
        retreat_after_close_if_needed(robot, cfg, exec_params, label=label)

    if return_home:
        _confirm_return_home(robot, cmd_mode=cmd_mode, params=params, label=label)
