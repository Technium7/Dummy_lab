"""End-to-end Windows flat grab_sensor pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .capture import (
    capture_realsense_frame,
    load_capture_from_files,
    save_debug_capture,
    save_grasp_overlay,
    validate_color_image_file,
)
from .config_loader import apply_runtime_calibrations
from .execute import execute_on_robot
from .grasp import build_grasp_targets_from_detections
from .ik_debug import show_ik_goal_debug_gui
from .models import GrabSensorConfig, GraspTargets
from .plan_motion import plan_approach_and_grasp
from .vision.infer import infer_scene, objects_to_debug

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_DEBUG_DIR = _PKG_DIR / "debug"


def _offline_capture_paths(cfg: GrabSensorConfig) -> tuple[Path | None, Path | None]:
    candidates: list[Path] = []
    if cfg.save_debug_dir is not None:
        candidates.append(cfg.save_debug_dir)
    candidates.append(_DEFAULT_DEBUG_DIR)
    for directory in candidates:
        color = directory / "grab_sensor.color.png"
        if not color.is_file():
            continue
        try:
            validate_color_image_file(color)
        except OSError as exc:
            print(f"[grab_sensor] Skipping corrupt offline color: {exc}")
            continue
        depth = directory / "grab_sensor.depth.npy"
        return color, depth if depth.is_file() else None
    return None, None


def run_detection(
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
) -> tuple[Any, GraspTargets, list[Any]]:
    cfg = apply_runtime_calibrations(cfg, params)
    color_file = params.get("color_file")
    depth_file = params.get("depth_file")
    offline = bool(params.get("offline", False))

    if color_file is None and offline:
        fallback_color, fallback_depth = _offline_capture_paths(cfg)
        if fallback_color is None:
            raise FileNotFoundError(
                f"offline mode requires grab_sensor.color.png under "
                f"{cfg.save_debug_dir or _DEFAULT_DEBUG_DIR}"
            )
        color_file = fallback_color
        depth_file = fallback_depth

    if color_file:
        capture = load_capture_from_files(
            Path(str(color_file)),
            Path(str(depth_file)) if depth_file else None,
        )
    else:
        try:
            camera_serial = str(params.get("camera_serial") or cfg.camera_serial or "").strip()
            capture = capture_realsense_frame(
                width=int(params.get("width", 640)),
                height=int(params.get("height", 480)),
                fps=int(params.get("fps", 30)),
                warmup=int(params.get("warmup", 15)),
                serial=camera_serial,
            )
        except (ImportError, RuntimeError) as exc:
            if not bool(params.get("allow_offline_fallback", True)):
                raise
            fallback_color, fallback_depth = _offline_capture_paths(cfg)
            if fallback_color is None:
                raise
            print(
                f"[grab_sensor] Live RealSense failed ({exc}). "
                f"Falling back to {fallback_color}"
            )
            capture = load_capture_from_files(fallback_color, fallback_depth)

    if cfg.save_debug_dir is not None:
        save_debug_capture(capture, cfg.save_debug_dir, stem="grab_sensor")

    objects = infer_scene(capture.color_rgb, capture.depth_m, cfg)
    vision_debug = objects_to_debug(objects)
    targets = build_grasp_targets_from_detections(cfg, objects, vision_debug=vision_debug)
    targets.debug["_depth_m"] = capture.depth_m

    print(
        f"[grab_sensor] detections: "
        f"{sum(1 for o in objects if o.class_name == 'sensor')} sensor, "
        f"{sum(1 for o in objects if o.class_name == 'post')} post, "
        f"{sum(1 for o in objects if o.class_name == 'other')} other"
    )
    print(f"[grab_sensor] sensor_uv={targets.debug.get('sensor_uv')} "
          f"axis_deg_image={targets.debug.get('sensor_axis_deg')} "
          f"axis_deg_table={targets.debug.get('holder_axis_table_deg')} "
          f"yaw_from={targets.debug.get('grasp_yaw_axis_source')}")

    if cfg.save_debug_dir is not None:
        cfg.save_debug_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "grasp_pos_robot_m": targets.grasp_pos_m.tolist(),
            "grasp_rpy_robot_deg": np.degrees(targets.grasp_rpy_rad).tolist(),
            "pre_grasp_pos_robot_m": targets.pre_grasp_pos_m.tolist(),
            "debug": {
                k: v
                for k, v in targets.debug.items()
                if k != "_depth_m" and not isinstance(v, np.ndarray)
            },
        }
        (cfg.save_debug_dir / "detection.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        save_grasp_overlay(
            capture, objects, targets, cfg.save_debug_dir / "grasp_overlay.png"
        )

    viz_path = params.get("visualize_out")
    if viz_path or params.get("visualize", False):
        out = Path(str(viz_path)) if viz_path else Path("grasp_overlay.png")
        if out.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            out = out / "grasp_overlay.png"
        save_grasp_overlay(capture, objects, targets, out)
        print(f"[grab_sensor] grasp overlay saved: {out.resolve()}")

    return capture, targets, objects


def run_grab_sensor_pipeline(
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    robot=None,
    start_joints_deg: list[float] | None = None,
) -> dict[str, Any]:
    cfg = apply_runtime_calibrations(cfg, params)
    _, targets, _objects = run_detection(cfg, params)

    print(
        f"[grab_sensor] grasp RPY world deg={targets.debug.get('grasp_rpy_world_deg')}  "
        f"mode={targets.debug.get('grasp_orientation_mode')}"
    )
    if cfg.grasp_position_from_table:
        from .table_position import format_robot_in_table_display, format_table_xyz_display

        print(f"[grab_sensor] robot in table frame: {format_robot_in_table_display(cfg.robot_in_table_frame)}")
        summary = targets.debug.get("grasp_frame_summary")
        if summary:
            print(
                f"[grab_sensor] grasp target (table): "
                f"{format_table_xyz_display([v / 1000.0 for v in summary['table_xyz_mm']])}"
            )

    if start_joints_deg is None:
        from .plan_motion import work_joints_deg

        start_joints_deg = list(work_joints_deg(cfg))
        print(f"[grab_sensor] start/return WORK joints deg={start_joints_deg}")

    if bool(params.get("ik_gui", False)):
        show_ik_goal_debug_gui(cfg, targets)
        return {
            "targets": targets,
            "trajectory_rad": None,
            "plan_meta": {},
            "start_joints_deg": start_joints_deg,
        }

    trajectory, plan_meta = plan_approach_and_grasp(
        cfg,
        start_joints_deg,
        targets,
        use_gui=bool(params.get("plan_gui", False)),
    )
    print(f"[grab_sensor] planned {plan_meta['n_total']} waypoints.")

    if bool(params.get("plan_gui", False)):
        from .ik_debug import playback_grasp_plan_in_gui

        playback_grasp_plan_in_gui(cfg, trajectory, targets)
    else:
        print(
            "[grab_sensor] No PyBullet window (--visualize is camera overlay only). "
            "Add --ik-gui or --plan-gui for robot preview."
        )

    result: dict[str, Any] = {
        "targets": targets,
        "trajectory_rad": trajectory,
        "plan_meta": plan_meta,
        "start_joints_deg": start_joints_deg,
    }

    if bool(params.get("execute", False)):
        exec_params = dict(params)
        exec_params.setdefault("joint_current_limit_a", cfg.execute.joint_current_limit_a)
        exec_params.setdefault("home_confirm_attempts", cfg.execute.home_confirm_attempts)
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
            execute_on_robot(robot, trajectory, targets, cfg, exec_params)
        finally:
            if own_robot:
                from manual_control.robot_session import disconnect_robot

                disconnect_robot(robot)

    return result
