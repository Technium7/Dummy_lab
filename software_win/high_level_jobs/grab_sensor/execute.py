from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parents[2]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from manual_control.mapping import (  # noqa: E402
    HOME_JOINTS_DEG,
    clamp_firmware_joints_deg,
    continuous_firmware_joint,
    shortest_deg_delta,
    unwrap_firmware_trajectory_deg,
)
from manual_control.robot_session import apply_arm_current_limits  # noqa: E402
from robot_ascii_interface import RobotAsciiInterface  # noqa: E402
from trajectory_run.run_planned_trajectory import (  # noqa: E402
    _hold_stream_speed,
    debug_robot_state,
    ensure_robot_armed,
    ensure_robot_enabled,
    pin_to_joint_target,
    prepare_robot_trajectory,
    read_physical_joints,
    sim_trajectory_to_robot_deg,
    stream_waypoints_deg,
    wait_until_joints,
)

from .models import GrabSensorConfig, GraspTargets


def _init_robot_drives(
    robot: RobotAsciiInterface,
    *,
    cmd_mode: int,
    joint_current_limit_a: float,
    label: str = "",
) -> bool:
    """Same init order as stable manual_control teleop: I_LIMIT before START."""
    if joint_current_limit_a > 0.0:
        print(f"{label}[Robot] I_LIMIT_J 1-6 -> {joint_current_limit_a:g}A (before START)")
        apply_arm_current_limits(robot, joint_current_limit_a, label=label)
    print(f"{label}!START...")
    start_resp = robot.start()
    print(f"{label}!START -> {start_resp}")
    time.sleep(0.15)
    if cmd_mode == 2:
        robot.ensure_cmdmode_stream()
        print(f"{label}#CMDMODE 2 -> stream armed")
    else:
        print(f"{label}#CMDMODE {cmd_mode} -> {robot.set_command_mode(cmd_mode)}")
        time.sleep(0.2)
    print(f"{label}HAND_EN -> {robot.hand_enable()}")
    time.sleep(0.1)
    if read_physical_joints(robot, retries=8) is None:
        debug_robot_state(
            robot,
            step="init — no joint reads after !START",
            label=label,
            target_deg=HOME_JOINTS_DEG,
        )
        return False
    return True


def _stream_to_joint_target(
    robot: RobotAsciiInterface,
    target_deg: Sequence[float],
    *,
    cmd_mode: int = 2,
    tol_deg: float = 0.5,
    timeout_sec: float = 90.0,
    speed_scale: float = 0.65,
    label: str = "",
    name: str = "HOME",
) -> tuple[bool, list[float] | None]:
    """Stream ``&`` waypoints to a joint target (teleop-style; avoids blocking ``!HOME``)."""
    target = clamp_firmware_joints_deg([float(target_deg[j]) for j in range(6)])
    current = read_physical_joints(robot)
    if current is not None:
        err = float(np.max(np.abs(np.asarray(current[:6]) - np.asarray(target))))
        print(f"{label}{name}: max |Δ|={err:.2f} deg before move")
        if err <= tol_deg:
            print(f"{label}{name}: already within {tol_deg:.2f} deg.")
            ensure_robot_armed(robot, cmd_mode=cmd_mode, label=label)
            robot.hold_stream_pose(target, _hold_stream_speed())
            time.sleep(0.2)
            return True, current

    start_q = np.asarray(
        clamp_firmware_joints_deg(current if current is not None else target),
        dtype=np.float64,
    )
    deltas = np.zeros(6, dtype=np.float64)
    for j in range(6):
        if continuous_firmware_joint(j):
            deltas[j] = shortest_deg_delta(float(start_q[j]), float(target[j]))
        else:
            deltas[j] = float(target[j]) - float(start_q[j])
    err = float(np.max(np.abs(deltas)))
    n_pts = max(2, min(32, int(np.ceil(err / 2.0)) + 1))
    waypoints = np.asarray(
        [start_q + deltas * (t / (n_pts - 1)) for t in range(n_pts)],
        dtype=np.float64,
    )
    # Keep limited joints in-range; wrap continuous into ±180 for the wire.
    waypoints = np.asarray(
        [clamp_firmware_joints_deg(row) for row in waypoints],
        dtype=np.float64,
    )

    print(f"{label}{name}: streaming {len(waypoints)} waypoints (max |Δ|={err:.1f} deg)...")
    ensure_robot_armed(robot, cmd_mode=cmd_mode, label=label)
    if not stream_waypoints_deg(
        robot,
        waypoints,
        send_interval_ms=120.0,
        speed_scale=speed_scale,
        label=label,
        arm=False,
        seed_from_robot=True,
        hold_end=True,
    ):
        return False, read_physical_joints(robot)

    ok, final = wait_until_joints(
        robot,
        target,
        timeout_sec=timeout_sec,
        tol_deg=tol_deg,
        label=label,
        name=name,
    )
    if ok and final is not None:
        robot.hold_stream_pose(clamp_firmware_joints_deg(final), _hold_stream_speed())
        time.sleep(0.2)
    return ok, final


def ensure_robot_ready(
    robot: RobotAsciiInterface,
    *,
    cmd_mode: int = 2,
    home_first: bool = True,
    open_hand: bool = False,
    joint_current_limit_a: float = 0.0,
    label: str = "",
    start_joints_deg: Sequence[float] | None = None,
) -> bool:
    if not _init_robot_drives(
        robot,
        cmd_mode=cmd_mode,
        joint_current_limit_a=joint_current_limit_a,
        label=label,
    ):
        return False

    if open_hand:
        robot.hand_open()
        time.sleep(0.3)

    if not home_first:
        return True

    target = (
        [float(v) for v in start_joints_deg[:6]]
        if start_joints_deg is not None and len(start_joints_deg) >= 6
        else list(HOME_JOINTS_DEG)
    )
    ok, at_home = _stream_to_joint_target(
        robot,
        target,
        cmd_mode=cmd_mode,
        label=label,
        name="WORK",
    )
    if not ok:
        debug_robot_state(
            robot,
            step="ensure_robot_ready — stream WORK failed",
            label=label,
            target_deg=target,
        )
        return False

    hold_q = at_home if at_home is not None else read_physical_joints(robot)
    if hold_q is not None:
        robot.hold_stream_pose(clamp_firmware_joints_deg(hold_q), _hold_stream_speed())
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
    settle_timeout_sec: float = 120.0,
    hold_end: bool = True,
    min_speed_scale: float = 0.25,
    on_waypoint=None,
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
            # Wrap GETJPOS into firmware range (e.g. 241° → -119°), then re-branch
            # the rest of the path so J1/J4/J6 do not take the long way around.
            seed = clamp_firmware_joints_deg(current)
            waypoints_deg[0] = seed
            if len(waypoints_deg) >= 2:
                waypoints_deg = np.asarray(
                    unwrap_firmware_trajectory_deg(waypoints_deg, reference_deg=seed),
                    dtype=np.float64,
                )
            if velocities_deg is not None and len(velocities_deg) > 0:
                velocities_deg = np.asarray(velocities_deg, dtype=np.float64).copy()
                velocities_deg[0] = 0.0
    seg = f" ({segment})" if segment else ""
    if info.get("smooth_fallback"):
        reject = info.get("smooth_reject") or {}
        print(
            f"{label}Smooth rejected{seg}: peak_v={reject.get('peak_vel_deg_s', '?')} "
            f"peak_a={reject.get('peak_acc_deg_s2', '?')} — using raw plan samples."
        )
    print(f"{label}Prepared {len(waypoints_deg)} robot waypoints{seg} ({info}).")
    planned_duration_s = info.get("duration_s") if info.get("smoothed") else None
    stream_scale = float(speed_scale)
    ramp_from = None
    # Ease from prior cruise to target whenever speed changes.
    if ramp_speed_scale is not None and abs(float(ramp_speed_scale) - stream_scale) > 1e-6:
        ramp_from = float(ramp_speed_scale)
    return stream_waypoints_deg(
        robot,
        waypoints_deg,
        velocities_deg_s=velocities_deg,
        send_interval_ms=send_interval_ms,
        planned_duration_s=planned_duration_s,
        speed_scale=stream_scale,
        ramp_from_speed_scale=ramp_from,
        label=label,
        arm=arm,
        seed_from_robot=seed_from_robot,
        settle_timeout_sec=settle_timeout_sec,
        hold_end=hold_end,
        min_speed_scale=min_speed_scale,
        on_waypoint=on_waypoint,
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


def _home_joint_error_deg(
    robot: RobotAsciiInterface,
    target_joints_deg: Sequence[float] | None = None,
) -> float | None:
    current = read_physical_joints(robot)
    if current is None:
        return None
    target = clamp_firmware_joints_deg(
        target_joints_deg if target_joints_deg is not None else HOME_JOINTS_DEG
    )
    current_n = clamp_firmware_joints_deg(current)
    errs = []
    for j in range(6):
        if continuous_firmware_joint(j):
            errs.append(abs(shortest_deg_delta(target[j], current_n[j])))
        else:
            errs.append(abs(current_n[j] - target[j]))
    return float(max(errs))


def _log_home_error(
    robot: RobotAsciiInterface,
    *,
    label: str = "",
    step: str,
    target_joints_deg: Sequence[float] | None = None,
) -> None:
    target = list(target_joints_deg) if target_joints_deg is not None else list(HOME_JOINTS_DEG)
    err = _home_joint_error_deg(robot, target)
    if err is not None:
        print(f"{label}{step}: max |Δ| from WORK = {err:.2f} deg")
    else:
        debug_robot_state(robot, step=step, label=label, target_deg=target)


def _confirm_return_home(
    robot: RobotAsciiInterface,
    *,
    cmd_mode: int,
    params: Mapping[str, Any],
    label: str = "",
    target_joints_deg: Sequence[float] | None = None,
) -> None:
    """Verify WORK after the streamed return segment; stream/pin correction only."""
    target = (
        [float(v) for v in target_joints_deg[:6]]
        if target_joints_deg is not None and len(target_joints_deg) >= 6
        else list(HOME_JOINTS_DEG)
    )
    tol_deg = float(params.get("home_arrive_tol_deg", 0.5))
    attempts = max(1, int(params.get("home_confirm_attempts", 3)))
    last_err: str | None = None

    for attempt in range(attempts):
        err = _home_joint_error_deg(robot, target)
        if err is not None and err <= tol_deg:
            print(f"{label}At WORK (max |Δ|={err:.3f} deg).")
            return

        if attempt > 0:
            print(f"{label}WORK confirm retry {attempt + 1}/{attempts}...")
            if not ensure_robot_enabled(robot, label=label):
                last_err = "no joint reads after !START"
                continue
            time.sleep(0.3)

        print(f"{label}Confirming WORK (stream)...")
        ok, final = _stream_to_joint_target(
            robot,
            target,
            cmd_mode=cmd_mode,
            tol_deg=tol_deg,
            timeout_sec=float(params.get("home_settle_timeout_sec", 90.0)),
            label=label,
            name="WORK confirm",
        )
        if ok and final is not None:
            err = float(np.max(np.abs(np.asarray(final[:6]) - np.asarray(target))))
            print(f"{label}At WORK (max |Δ|={err:.3f} deg).")
            return

        pin_ok, pinned = pin_to_joint_target(
            robot,
            target,
            tol_deg=tol_deg,
            label=label,
            name="WORK",
        )
        if pin_ok and pinned is not None:
            pin_err = float(np.max(np.abs(np.asarray(pinned[:6]) - np.asarray(target))))
            print(f"{label}At WORK after pin (max |Δ|={pin_err:.3f} deg).")
            return

        err_now = _home_joint_error_deg(robot, target)
        last_err = f"stream/pin failed; max |Δ|={err_now:.2f} deg" if err_now is not None else "no joint reads"
        print(f"{label}[debug] WORK attempt {attempt + 1}/{attempts}: {last_err}")
        debug_robot_state(robot, step=f"WORK attempt {attempt + 1} failed", label=label, target_deg=target)

    raise RuntimeError(
        f"{label}Failed to reach WORK after {attempts} attempt(s) ({last_err}). "
        "Check the arm and serial link; avoid Ctrl+Z (stopped jobs hold the port)."
    )


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
        debug_robot_state(
            robot,
            step=f"stream failed: {segment}",
            label=stream_kw.get("label", ""),
            target_deg=HOME_JOINTS_DEG,
        )
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


def execute_on_robot(
    robot: RobotAsciiInterface,
    _trajectory_rad: np.ndarray,
    targets: GraspTargets,
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> None:
    del targets  # kept for API compatibility with callers
    exec_params = dict(params)
    meta = exec_params.get("plan_meta") or {}
    _populate_pre_grasp_joints_deg(exec_params, meta)

    segments = _motion_segments(meta)
    if segments.get("grasp") is None:
        raise RuntimeError(
            "execute_on_robot requires plan_meta['segments']['grasp'] from plan_approach_and_grasp."
        )

    pre_seg = segments.get("pre_grasp")
    grasp_seg = segments.get("grasp")
    lift_seg = segments.get("lift")
    home_seg = segments.get("home")

    open_gripper_at_pre = bool(params.get("open_gripper", True))
    close_at_grasp = bool(params.get("close_gripper", True))
    return_home = bool(meta.get("plan_return_home", cfg.approach.plan_return_home))

    from .plan_motion import work_joints_deg

    work_deg = work_joints_deg(cfg)
    cmd_mode = int(params.get("cmd_mode", 2))
    home_first = bool(params.get("home_first", True))
    joint_current_limit_a = float(
        params.get("joint_current_limit_a", cfg.execute.joint_current_limit_a)
    )
    if not ensure_robot_ready(
        robot,
        cmd_mode=cmd_mode,
        home_first=home_first,
        open_hand=False,
        joint_current_limit_a=joint_current_limit_a,
        label=label,
        start_joints_deg=work_deg,
    ):
        raise RuntimeError("Failed to reach WORK before grasp execution.")

    stream_kw = {
        "smooth": bool(params.get("smooth", True)),
        "send_interval_ms": float(params.get("send_interval_ms", 120.0)),
        "speed_scale": float(params.get("speed_scale", 0.85)),
        "label": label,
    }

    armed = home_first

    if pre_seg is not None:
        print(f"{label}Phase 1: WORK -> pre-grasp ({len(pre_seg)} planner samples).")
        ramp = float(params.get("approach_speed_scale", 0.65))
        _stream_segment_or_raise(
            robot,
            pre_seg,
            segment="WORK -> pre-grasp",
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
            sync_start_to_robot=True,
            ramp_speed_scale=min(ramp, float(stream_kw["speed_scale"])),
        )
        armed = True
        if open_gripper_at_pre:
            open_gripper(robot, label=label)

    if grasp_seg is not None:
        phase = "pre-grasp -> grasp" if pre_seg is not None else "WORK -> grasp"
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
            # Match physical pose after pre-insert hold so insert does not jerk off-plan.
            sync_start_to_robot=True,
        )
        armed = True
    elif open_gripper_at_pre and pre_seg is None:
        open_gripper(robot, label=label)

    if close_at_grasp:
        close_gripper(robot, label=label)

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

    if home_seg is not None:
        print(f"{label}Phase 4: return WORK ({len(home_seg)} planner samples).")
        home_ramp = params.get("return_home_speed_scale")
        _stream_segment_or_raise(
            robot,
            home_seg,
            segment="return WORK",
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
            sync_start_to_robot=True,
            ramp_speed_scale=float(home_ramp) if home_ramp is not None else None,
        )
        _log_home_error(robot, label=label, step="After return stream", target_joints_deg=work_deg)
    elif close_at_grasp and lift_seg is None:
        retreat_after_close_if_needed(robot, cfg, exec_params, label=label)

    if return_home:
        _confirm_return_home(
            robot,
            cmd_mode=cmd_mode,
            params=params,
            label=label,
            target_joints_deg=work_deg,
        )


def execute_put_sensor_on_robot(
    robot: RobotAsciiInterface,
    segments: Mapping[str, Any],
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> None:
    """Put sensor back using reversed grab_sensor segments; end at WORK open."""
    from high_level_jobs.session.motion_store import MotionRecord

    record = MotionRecord(
        name="put",
        kind="grab_sensor",
        segments={
            k: (np.asarray(v, dtype=np.float64) if v is not None else None)
            for k, v in segments.items()
        },
    )
    ordered = record.reversed_put_order()
    if not ordered:
        raise RuntimeError("put_sensor: no reversible segments in motion record.")

    from .plan_motion import work_joints_deg as _work_joints_deg

    work_deg = _work_joints_deg(cfg)
    cmd_mode = int(params.get("cmd_mode", 2))
    home_first = bool(params.get("home_first", True))
    joint_current_limit_a = float(
        params.get("joint_current_limit_a", cfg.execute.joint_current_limit_a)
    )
    if not ensure_robot_ready(
        robot,
        cmd_mode=cmd_mode,
        home_first=home_first,
        open_hand=False,
        joint_current_limit_a=joint_current_limit_a,
        label=label,
        start_joints_deg=work_deg,
    ):
        raise RuntimeError("Failed to reach WORK before put_sensor.")

    stream_kw = {
        "smooth": bool(params.get("smooth", True)),
        "send_interval_ms": float(params.get("send_interval_ms", 120.0)),
        "speed_scale": float(params.get("speed_scale", 0.75)),
        "label": label,
    }

    opened = False
    keys = [k for k, _, _ in ordered]
    release_after = "lift" if "lift" in keys else ("home" if "home" in keys else None)

    for i, (key, seg_label, traj) in enumerate(ordered):
        if (not opened) and key == "grasp" and release_after is None:
            open_gripper(robot, label=label)
            opened = True
        print(f"{label}put_sensor: {seg_label} ({len(traj)} samples).")
        _stream_segment_or_raise(
            robot,
            traj,
            segment=seg_label,
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
            sync_start_to_robot=(i == 0),
        )
        if (not opened) and key == release_after:
            open_gripper(robot, label=label)
            opened = True

    if not opened:
        open_gripper(robot, label=label)

    _confirm_return_home(
        robot,
        cmd_mode=cmd_mode,
        params=params,
        label=label,
        target_joints_deg=work_deg,
    )


def execute_return_segments_on_robot(
    robot: RobotAsciiInterface,
    segment_list: Sequence[tuple[str, np.ndarray]],
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> None:
    """Stream recorded return segments (lift/home or reversed place) to WORK."""
    from .plan_motion import work_joints_deg as _work_joints_deg

    if not segment_list:
        raise RuntimeError("return: empty segment list.")

    work_deg = _work_joints_deg(cfg)
    cmd_mode = int(params.get("cmd_mode", 2))
    joint_current_limit_a = float(
        params.get("joint_current_limit_a", cfg.execute.joint_current_limit_a)
    )
    if not ensure_robot_ready(
        robot,
        cmd_mode=cmd_mode,
        home_first=False,
        open_hand=False,
        joint_current_limit_a=joint_current_limit_a,
        label=label,
    ):
        raise RuntimeError("Failed to arm robot for return.")

    stream_kw = {
        "smooth": bool(params.get("smooth", True)),
        "send_interval_ms": float(params.get("send_interval_ms", 160.0)),
        "speed_scale": float(params.get("speed_scale", 0.55)),
        "label": label,
    }
    for i, (seg_label, traj) in enumerate(segment_list):
        print(f"{label}return: {seg_label} ({len(traj)} samples).")
        home_ramp = params.get("return_home_speed_scale") if i == len(segment_list) - 1 else None
        _stream_segment_or_raise(
            robot,
            np.asarray(traj, dtype=np.float64),
            segment=seg_label,
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
            sync_start_to_robot=(i == 0),
            ramp_speed_scale=float(home_ramp) if home_ramp is not None else None,
        )
    _confirm_return_home(
        robot,
        cmd_mode=cmd_mode,
        params=params,
        label=label,
        target_joints_deg=work_deg,
    )


def execute_outbound_segments_on_robot(
    robot: RobotAsciiInterface,
    segment_list: Sequence[tuple[str, np.ndarray]],
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> None:
    """Stream recorded outbound place segments (pre_grasp + grasp); stay at insert."""
    from .plan_motion import work_joints_deg as _work_joints_deg

    if not segment_list:
        raise RuntimeError("traj_replay: empty outbound segment list.")

    work_deg = _work_joints_deg(cfg)
    cmd_mode = int(params.get("cmd_mode", 2))
    home_first = bool(params.get("home_first", True))
    joint_current_limit_a = float(
        params.get("joint_current_limit_a", cfg.execute.joint_current_limit_a)
    )
    if not ensure_robot_ready(
        robot,
        cmd_mode=cmd_mode,
        home_first=home_first,
        open_hand=False,
        joint_current_limit_a=joint_current_limit_a,
        label=label,
        start_joints_deg=work_deg if home_first else None,
    ):
        raise RuntimeError("Failed to arm robot for traj_replay.")

    stream_kw = {
        "smooth": bool(params.get("smooth", True)),
        "send_interval_ms": float(params.get("send_interval_ms", 160.0)),
        "speed_scale": float(params.get("speed_scale", 0.55)),
        "label": label,
    }
    for i, (seg_label, traj) in enumerate(segment_list):
        print(f"{label}traj_replay: {seg_label} ({len(traj)} samples).")
        _stream_segment_or_raise(
            robot,
            np.asarray(traj, dtype=np.float64),
            segment=seg_label,
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
            sync_start_to_robot=(i == 0),
        )


def execute_standby_to_work(
    robot: RobotAsciiInterface,
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> dict[str, Any]:
    """Plan from current joints to WORK and stream (fresh plan, not a recording)."""
    from trajectory_run.run_planned_trajectory import read_physical_joints

    from .plan_motion import (
        build_scene,
        plan_return_home_segment,
        start_q_from_robot_joints_deg,
        work_joints_deg as _work_joints_deg,
    )

    work_deg = _work_joints_deg(cfg)
    cmd_mode = int(params.get("cmd_mode", 2))
    joint_current_limit_a = float(
        params.get("joint_current_limit_a", cfg.execute.joint_current_limit_a)
    )
    if not ensure_robot_ready(
        robot,
        cmd_mode=cmd_mode,
        home_first=False,
        open_hand=False,
        joint_current_limit_a=joint_current_limit_a,
        label=label,
    ):
        raise RuntimeError("Failed to arm robot for standby.")

    current = read_physical_joints(robot)
    if current is None:
        raise RuntimeError("standby: cannot read current joints.")

    scene = build_scene(cfg)
    try:
        start_q = start_q_from_robot_joints_deg(scene, current)
        home_traj, home_goal_q = plan_return_home_segment(
            scene,
            start_q,
            enable_collision=cfg.enable_collision,
            work_joints_deg_list=work_deg,
        )
        if home_traj is None or home_goal_q is None:
            raise RuntimeError("standby: failed to plan return to WORK.")
        print(f"{label}standby: streaming planned return ({len(home_traj)} samples).")
        stream_kw = {
            "smooth": bool(params.get("smooth", True)),
            "send_interval_ms": float(params.get("send_interval_ms", 120.0)),
            "speed_scale": float(params.get("speed_scale", 0.65)),
            "label": label,
        }
        _stream_segment_or_raise(
            robot,
            np.asarray(home_traj, dtype=np.float64),
            segment="standby -> WORK",
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
            sync_start_to_robot=True,
        )
        _confirm_return_home(
            robot,
            cmd_mode=cmd_mode,
            params=params,
            label=label,
            target_joints_deg=work_deg,
        )
        return {
            "trajectory_rad": np.asarray(home_traj, dtype=np.float64),
            "home_goal_q": home_goal_q,
            "start_joints_deg": list(current[:6]),
        }
    finally:
        from trajectory_planner.scene import disconnect_planning_scene

        disconnect_planning_scene(scene)
