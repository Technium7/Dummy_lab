from __future__ import annotations

import logging

from robot_config import default_robot_port
from typing import Any, Mapping

from high_level_jobs.subtasks.robot_actions import _get_or_create_robot, _v

from .config_loader import load_grab_sensor_config, resolve_config_path
from .execute import execute_on_robot
from .pipeline import run_detection, run_grab_sensor_pipeline
from .plan_motion import plan_approach_and_grasp

logger = logging.getLogger(__name__)


def _load_cfg(params: Mapping[str, Any], ctx: Mapping[str, Any]):
    path = resolve_config_path(params, ctx)
    cfg = load_grab_sensor_config(path)
    return cfg, path


def grab_sensor_detect(params: Mapping[str, Any], ctx: dict[str, Any]) -> None:
    """
    Capture with fixed D415, segment the table object, estimate grasp location in robot frame.

    Only fixed spatial config is ``camera_to_robot``. Object pose comes from vision each time.
    Stores ``ctx['grab_sensor_targets']`` and ``ctx['grab_sensor_config']``.
    Params: config, color_file, depth_file, camera_serial, width, height, ...
    """
    cfg, path = _load_cfg(params, ctx)
    ctx["grab_sensor_config_path"] = str(path)
    ctx["grab_sensor_config"] = cfg

    if ctx.get("dry_run"):
        _v(ctx, "[dry-run] grab_sensor_detect: would capture + segment + estimate pose")
        return

    _, targets = run_detection(cfg, params)[:2]
    ctx["grab_sensor_targets"] = targets
    _v(
        ctx,
        f"Grasp target (robot m) {targets.grasp_pos_m.round(4).tolist()}, "
        f"rpy deg {targets.debug.get('grasp_rpy_robot_deg')}",
    )


def grab_sensor_plan(params: Mapping[str, Any], ctx: dict[str, Any]) -> None:
    """
    Plan pre-grasp + grasp with iLQR. Requires ``grab_sensor_detect`` or prior targets in ctx.

    Stores ``ctx['grab_sensor_trajectory_rad']``.
    """
    cfg = ctx.get("grab_sensor_config")
    if cfg is None:
        cfg, path = _load_cfg(params, ctx)
        ctx["grab_sensor_config"] = cfg
        ctx["grab_sensor_config_path"] = str(path)

    targets = ctx.get("grab_sensor_targets")
    if targets is None:
        raise RuntimeError("grab_sensor_plan requires grab_sensor_targets (run grab_sensor_detect first).")

    if ctx.get("dry_run"):
        _v(ctx, "[dry-run] grab_sensor_plan: would run iLQR to pre-grasp + grasp")
        return

    start_deg = params.get("start_joints_deg")
    if start_deg is None and ctx.get("robot") is not None:
        from trajectory_run.run_planned_trajectory import read_physical_joints

        current = read_physical_joints(ctx["robot"])
        if current is not None:
            start_deg = current
    if start_deg is None:
        from manual_control.mapping import HOME_JOINTS_DEG

        start_deg = list(HOME_JOINTS_DEG)

    trajectory, meta = plan_approach_and_grasp(
        cfg,
        start_deg,
        targets,
        use_gui=bool(params.get("plan_gui", False)),
    )
    ctx["grab_sensor_trajectory_rad"] = trajectory
    ctx["grab_sensor_plan_meta"] = meta
    _v(ctx, f"Planned grasp trajectory ({meta['n_total']} samples).")


def grab_sensor_execute(params: Mapping[str, Any], ctx: dict[str, Any]) -> None:
    """
    Stream planned trajectory and close gripper. Requires robot connection + plan in ctx.
    """
    trajectory = ctx.get("grab_sensor_trajectory_rad")
    targets = ctx.get("grab_sensor_targets")
    cfg = ctx.get("grab_sensor_config")
    if trajectory is None or targets is None or cfg is None:
        raise RuntimeError(
            "grab_sensor_execute requires grab_sensor_trajectory_rad and grab_sensor_targets."
        )

    if ctx.get("dry_run"):
        _v(ctx, "[dry-run] grab_sensor_execute: would stream trajectory + close gripper")
        return

    port = str(params.get("port") or ctx.get("robot_port") or default_robot_port())
    baud = int(params.get("baudrate") or ctx.get("robot_baudrate") or 115200)
    robot = _get_or_create_robot(ctx, port, baud)
    if not robot.ser or not robot.ser.is_open:
        if not robot.connect():
            raise RuntimeError(f"Failed to connect to {port}")

    execute_on_robot(robot, trajectory, targets, cfg, {**params, "plan_meta": ctx.get("grab_sensor_plan_meta")})
    _v(ctx, "Grasp execution finished.")


def grab_sensor(params: Mapping[str, Any], ctx: dict[str, Any]) -> None:
    """Full pipeline: detect → plan → (optional) execute."""
    cfg, path = _load_cfg(params, ctx)
    ctx["grab_sensor_config"] = cfg
    ctx["grab_sensor_config_path"] = str(path)

    if ctx.get("dry_run"):
        _v(ctx, "[dry-run] grab_sensor: detect + plan + execute (no hardware)")
        return

    port = str(params.get("port") or ctx.get("robot_port") or default_robot_port())
    baud = int(params.get("baudrate") or ctx.get("robot_baudrate") or 115200)
    robot = ctx.get("robot")
    exec_params = dict(params)
    exec_params.setdefault("execute", True)

    if exec_params.get("execute") and robot is None:
        robot = _get_or_create_robot(ctx, port, baud)
        if not robot.ser or not robot.ser.is_open:
            if not robot.connect():
                raise RuntimeError(f"Failed to connect to {port}")

    result = run_grab_sensor_pipeline(cfg, exec_params, robot=robot)
    ctx["grab_sensor_targets"] = result["targets"]
    ctx["grab_sensor_trajectory_rad"] = result["trajectory_rad"]
    ctx["grab_sensor_plan_meta"] = result["plan_meta"]
    _v(ctx, "grab_sensor pipeline complete.")


GRAB_SENSOR_SUBTASKS = {
    "grab_sensor": grab_sensor,
    "grab_sensor_detect": grab_sensor_detect,
    "grab_sensor_plan": grab_sensor_plan,
    "grab_sensor_execute": grab_sensor_execute,
}
