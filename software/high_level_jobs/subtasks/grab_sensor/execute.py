from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from manual_control.mapping import HOME_JOINTS_DEG  # noqa: E402
from robot_ascii_interface import RobotAsciiInterface  # noqa: E402
from trajectory_run.run_planned_trajectory import (  # noqa: E402
    firmware_home,
    prepare_robot_trajectory,
    stream_waypoints_deg,
)

from .models import GrabSensorConfig, GraspTargets


def ensure_robot_ready(
    robot: RobotAsciiInterface,
    *,
    cmd_mode: int = 2,
    home_first: bool = True,
    label: str = "",
) -> bool:
    print(f"{label}!START -> {robot.start()}")
    time.sleep(0.2)
    print(f"{label}HAND_EN -> {robot.hand_enable()}")
    time.sleep(0.1)
    robot.hand_open()
    time.sleep(0.3)

    if not home_first:
        print(f"{label}#CMDMODE {cmd_mode} -> {robot.set_command_mode(cmd_mode)}")
        time.sleep(0.2)
        return True

    ok, _ = firmware_home(
        robot,
        HOME_JOINTS_DEG,
        cmd_mode=cmd_mode,
        label=label,
    )
    return ok


def execute_grasp_trajectory(
    robot: RobotAsciiInterface,
    trajectory_rad: np.ndarray,
    *,
    smooth: bool = True,
    send_interval_ms: float = 120.0,
    speed_scale: float = 0.85,
    label: str = "",
) -> bool:
    waypoints_deg, _, info = prepare_robot_trajectory(
        trajectory_rad,
        smooth=smooth,
        sample_dt_s=send_interval_ms / 1000.0,
        max_vel_deg_s=25.0,
        max_acc_deg_s2=80.0,
    )
    print(f"{label}Prepared {len(waypoints_deg)} robot waypoints ({info}).")
    return stream_waypoints_deg(
        robot,
        waypoints_deg,
        send_interval_ms=send_interval_ms,
        speed_scale=speed_scale,
        label=label,
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
    print(f"{label}Closing gripper...")
    robot.hand_close()
    time.sleep(0.5)

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
    )


def execute_on_robot(
    robot: RobotAsciiInterface,
    trajectory_rad: np.ndarray,
    targets: GraspTargets,
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> None:
    cmd_mode = int(params.get("cmd_mode", 2))
    home_first = bool(params.get("home_first", True))
    if not ensure_robot_ready(robot, cmd_mode=cmd_mode, home_first=home_first, label=label):
        raise RuntimeError("Failed to reach HOME before grasp execution.")

    ok = execute_grasp_trajectory(
        robot,
        trajectory_rad,
        smooth=bool(params.get("smooth", True)),
        send_interval_ms=float(params.get("send_interval_ms", 120.0)),
        speed_scale=float(params.get("speed_scale", 0.85)),
        label=label,
    )
    if not ok:
        raise RuntimeError("Trajectory streaming failed.")

    exec_params = dict(params)
    meta = exec_params.get("plan_meta") or {}
    pre_goal_q = meta.get("pre_goal_q")
    if pre_goal_q is not None:
        from trajectory_run.run_planned_trajectory import sim_trajectory_to_robot_deg

        pre_deg = sim_trajectory_to_robot_deg(np.asarray([pre_goal_q], dtype=np.float64))[0]
        exec_params["pre_grasp_joints_deg"] = pre_deg.tolist()

    if bool(params.get("close_gripper", True)):
        close_gripper_and_retreat(robot, targets, cfg, exec_params, label=label)
