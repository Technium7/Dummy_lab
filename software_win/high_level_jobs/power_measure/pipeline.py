"""Plan and optionally execute a top-down insert from a click target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from high_level_jobs.grab_sensor.execute import execute_on_robot
from high_level_jobs.grab_sensor.grasp import finalize_grasp_targets
from high_level_jobs.grab_sensor.ik_debug import show_ik_goal_debug_gui
from high_level_jobs.grab_sensor.plan_motion import plan_approach_and_grasp

from .config_loader import apply_runtime_calibrations
from .models import PowerMeasureConfig
from .target import InsertTarget, insert_target_to_grasp_targets

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_DEBUG_DIR = _PKG_DIR / "debug"


def _save_debug(cfg: PowerMeasureConfig, insert: InsertTarget, plan_meta: Mapping[str, Any]) -> None:
    debug_dir = cfg.save_debug_dir or _DEFAULT_DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "insert": {
            k: v
            for k, v in insert.debug.items()
            if not isinstance(v, np.ndarray)
        },
        "plan_meta": {
            k: (int(v) if isinstance(v, (np.integer,)) else v)
            for k, v in plan_meta.items()
            if k != "segments" and not isinstance(v, np.ndarray)
        },
    }
    (debug_dir / "power_measure.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[power_measure] debug saved: {debug_dir / 'power_measure.json'}")


def run_insert_pipeline(
    cfg: PowerMeasureConfig,
    insert: InsertTarget,
    params: Mapping[str, Any],
    *,
    robot=None,
    start_joints_deg: list[float] | None = None,
) -> dict[str, Any]:
    """Plan pre→insert→lift/home; optionally execute with gripper held closed."""
    cfg = apply_runtime_calibrations(cfg, params)
    grab_cfg = cfg.to_grab_sensor_config()
    targets = insert_target_to_grasp_targets(insert)

    print(
        f"[power_measure] insert RPY world deg={targets.debug.get('insert_rpy_world_deg')}  "
        f"mode={targets.debug.get('grasp_orientation_mode')}"
    )
    from high_level_jobs.grab_sensor.table_position import (
        format_robot_in_table_display,
        format_table_xyz_display,
    )

    print(f"[power_measure] robot in table frame: {format_robot_in_table_display(cfg.robot_in_table_frame)}")
    print(f"[power_measure] measured (table): {format_table_xyz_display(insert.measured_table_xyz_m)}")
    print(f"[power_measure] TCP (table): {format_table_xyz_display(insert.tcp_table_xyz_m)}")

    if start_joints_deg is None:
        from high_level_jobs.grab_sensor.plan_motion import work_joints_deg

        start_joints_deg = list(work_joints_deg(grab_cfg))
        print(f"[power_measure] start/return WORK joints deg={start_joints_deg}")

    # Match grab_sensor: finalize before IK GUI / planning so TCP XYZ+RPY are consistent.
    finalize_grasp_targets(grab_cfg, targets)
    print(
        f"[power_measure] finalized RPY world deg={targets.debug.get('grasp_rpy_world_deg')}  "
        f"TCP robot m={targets.grasp_pos_m.round(4).tolist()}"
    )

    if bool(params.get("ik_gui", False)):
        print("[power_measure] opening PyBullet IK GUI (--ik-gui); close window to exit.")
        show_ik_goal_debug_gui(grab_cfg, targets)
        return {
            "targets": targets,
            "insert": insert,
            "trajectory_rad": None,
            "plan_meta": {},
            "start_joints_deg": start_joints_deg,
        }

    trajectory, plan_meta = plan_approach_and_grasp(
        grab_cfg,
        start_joints_deg,
        targets,
        use_gui=bool(params.get("plan_gui", False)),
    )
    print(f"[power_measure] planned {plan_meta['n_total']} waypoints.")
    _save_debug(cfg, insert, plan_meta)

    if bool(params.get("plan_gui", False)):
        from high_level_jobs.grab_sensor.ik_debug import playback_grasp_plan_in_gui

        print("[power_measure] opening PyBullet plan playback (--plan-gui); close window to exit.")
        playback_grasp_plan_in_gui(grab_cfg, trajectory, targets)
    else:
        print(
            "[power_measure] No PyBullet window. "
            "Add --ik-gui or --plan-gui for robot preview."
        )

    result: dict[str, Any] = {
        "targets": targets,
        "insert": insert,
        "trajectory_rad": trajectory,
        "plan_meta": plan_meta,
        "start_joints_deg": start_joints_deg,
    }

    if bool(params.get("execute", False)):
        exec_params = dict(params)
        exec_params.setdefault("joint_current_limit_a", cfg.execute.joint_current_limit_a)
        exec_params.setdefault("home_confirm_attempts", cfg.execute.home_confirm_attempts)
        # Sensor already held — do not open/close gripper
        exec_params["open_gripper"] = False
        exec_params["close_gripper"] = False
        exec_params["plan_meta"] = plan_meta

        own_robot = False
        if robot is None:
            from robot_ascii_interface import RobotAsciiInterface
            from robot_config import default_robot_port

            port = str(params.get("port") or default_robot_port())
            robot = RobotAsciiInterface(
                port=port,
                baudrate=int(params.get("baudrate", 115200)),
            )
            if not robot.connect():
                raise RuntimeError(f"Failed to connect robot on {port}")
            own_robot = True
        try:
            mode = str(params.get("power_mode") or "full").strip().lower()
            use_power = bool(cfg.power_meter.enabled) and not bool(
                params.get("no_power_scan", False)
            )
            if mode in ("place", "place_only"):
                from .execute_power import execute_power_place_on_robot

                power_out = execute_power_place_on_robot(
                    robot,
                    trajectory,
                    targets,
                    cfg,
                    insert,
                    exec_params,
                )
                result["power_scan"] = power_out.get("power_scan")
                result["place_result"] = power_out
            elif use_power:
                from .execute_power import execute_power_measure_on_robot

                power_out = execute_power_measure_on_robot(
                    robot,
                    trajectory,
                    targets,
                    cfg,
                    insert,
                    exec_params,
                )
                result["power_scan"] = power_out.get("power_scan")
            else:
                execute_on_robot(robot, trajectory, targets, grab_cfg, exec_params)
        finally:
            if own_robot:
                from manual_control.robot_session import disconnect_robot

                disconnect_robot(robot)

    return result


def run_power_optimize(
    cfg: PowerMeasureConfig,
    insert: InsertTarget,
    params: Mapping[str, Any],
    *,
    robot=None,
) -> dict[str, Any]:
    """Run power search/climb from the current insert pose (session command)."""
    cfg = apply_runtime_calibrations(cfg, params)
    exec_params = dict(params)
    exec_params.setdefault("joint_current_limit_a", cfg.execute.joint_current_limit_a)

    own_robot = False
    if robot is None:
        from robot_ascii_interface import RobotAsciiInterface
        from robot_config import default_robot_port

        port = str(params.get("port") or default_robot_port())
        robot = RobotAsciiInterface(
            port=port,
            baudrate=int(params.get("baudrate", 115200)),
        )
        if not robot.connect():
            raise RuntimeError(f"Failed to connect robot on {port}")
        own_robot = True
    try:
        from .execute_power import execute_power_optimize_on_robot

        return execute_power_optimize_on_robot(
            robot,
            cfg,
            insert,
            exec_params,
        )
    finally:
        if own_robot:
            from manual_control.robot_session import disconnect_robot

            disconnect_robot(robot)
