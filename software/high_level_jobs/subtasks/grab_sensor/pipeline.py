from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .detection import (
    capture_realsense_frame,
    detect_table_object,
    detect_with_yolo_grasp,
    load_capture_from_files,
    save_debug_capture,
    save_grasp_overlay,
    segment_table_object,
    validate_color_image_file,
)
from .config_loader import apply_runtime_calibrations, load_grab_sensor_config
from .execute import execute_on_robot
from .grasp import build_grasp_targets
from .ik_debug import show_ik_goal_debug_gui
from .plan_motion import plan_approach_and_grasp
from .models import GrabSensorConfig, GraspTargets, VisionMode


def _vision_mode(cfg: GrabSensorConfig, params: Mapping[str, Any]) -> VisionMode:
    raw = params.get("vision_mode", cfg.vision_mode)
    mode = str(raw).strip().lower()
    if mode not in ("depth_segment", "yolo_grasp"):
        raise ValueError(f"Unknown vision_mode {raw!r}")
    return mode  # type: ignore[return-value]


def _run_vision(
    cfg: GrabSensorConfig,
    capture,
    params: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    mode = _vision_mode(cfg, params)
    extra: dict[str, Any] = {"vision_mode": mode}

    if mode == "yolo_grasp":
        if cfg.yolo is None or cfg.grasp_net is None:
            raise ValueError("yolo_grasp mode requires yolo and grasp_net in config.")
        result = detect_with_yolo_grasp(capture, cfg)
        extra.update(result.debug)
        return result.mask, result.detection, extra

    mask, bbox = segment_table_object(capture.depth_m, cfg.segmentation)
    detection = detect_table_object(mask, capture.depth_m, capture.depth_intrinsics, bbox)
    return mask, detection, extra


_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_DEBUG_DIR = _PKG_DIR / "debug"
_DEFAULT_OFFLINE_COLOR = _DEFAULT_DEBUG_DIR / "grab_sensor.color.png"
_DEFAULT_OFFLINE_DEPTH = _DEFAULT_DEBUG_DIR / "grab_sensor.depth.npy"


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
            print(f"[grab_sensor] Skipping corrupt offline color capture: {exc}")
            continue
        depth = directory / "grab_sensor.depth.npy"
        return color, depth if depth.is_file() else None
    return None, None


def run_detection(
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
) -> tuple[Any, GraspTargets, Any, Any]:
    cfg = apply_runtime_calibrations(cfg, params)
    color_file = params.get("color_file")
    depth_file = params.get("depth_file")
    offline = bool(params.get("offline", False))

    if color_file is None and offline:
        fallback_color, fallback_depth = _offline_capture_paths(cfg)
        if fallback_color is None:
            raise FileNotFoundError(
                "offline mode requires saved debug capture "
                f"(grab_sensor.color.png under {cfg.save_debug_dir or _DEFAULT_DEBUG_DIR})"
            )
        color_file = fallback_color
        depth_file = fallback_depth

    if color_file:
        capture = load_capture_from_files(
            Path(str(color_file)),
            Path(str(depth_file)) if depth_file else None,
            depth_scale=float(params.get("depth_scale", 0.001)),
        )
    else:
        try:
            camera_serial = str(
                params.get("camera_serial") or cfg.camera_serial or ""
            ).strip()
            capture = capture_realsense_frame(
                width=int(params.get("width", 640)),
                height=int(params.get("height", 480)),
                fps=int(params.get("fps", 30)),
                warmup=int(params.get("warmup", 15)),
                serial=camera_serial,
                frame_timeout_ms=int(params.get("frame_timeout_ms", 10000)),
                start_retries=int(params.get("camera_start_retries", 3)),
            )
        except (ImportError, RuntimeError) as exc:
            if not bool(params.get("allow_offline_fallback", True)):
                raise
            fallback_color, fallback_depth = _offline_capture_paths(cfg)
            if fallback_color is None:
                raise
            print(
                f"[grab_sensor] Live RealSense capture failed ({exc}). "
                f"Falling back to saved capture: {fallback_color}"
            )
            capture = load_capture_from_files(
                fallback_color,
                fallback_depth,
                depth_scale=float(params.get("depth_scale", 0.001)),
            )

    if cfg.save_debug_dir is not None:
        save_debug_capture(capture, cfg.save_debug_dir, stem="grab_sensor")

    mask, detection, vision_debug = _run_vision(cfg, capture, params)
    targets = build_grasp_targets(cfg, detection, vision_debug=vision_debug)

    if vision_debug.get("vision_mode") == "yolo_grasp":
        grasp_uv = vision_debug.get("grasp_uv")
        base_uv = vision_debug.get("base_uv")
        profile = vision_debug.get("detection_profile", "side")
        if profile == "top" or cfg.grasp_net and cfg.grasp_net.base_only:
            print(f"[grab_sensor] top_detection NN: base_uv={base_uv}")
        else:
            print(
                f"[grab_sensor] NN pixels: grasp_uv={grasp_uv} base_uv={base_uv}"
            )
        if offline:
            print(
                "[grab_sensor] offline mode — overlay uses saved debug capture, not the live "
                "camera frame. For a match with live_review, run without --offline or press O "
                "in live_review to save grasp_overlay.png from the current frame."
            )

    if cfg.save_debug_dir is not None:
        _save_detection_debug(cfg, capture, mask, detection, targets)
        save_grasp_overlay(
            capture, mask, detection, targets, cfg, cfg.save_debug_dir / "grasp_overlay.png"
        )

    viz_path = params.get("visualize_out")
    if viz_path or params.get("visualize", False):
        out = Path(str(viz_path)) if viz_path else Path("grasp_overlay.png")
        if out.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            out = out / "grasp_overlay.png"
        if cfg.save_debug_dir is None or out.resolve() != (cfg.save_debug_dir / "grasp_overlay.png").resolve():
            save_grasp_overlay(capture, mask, detection, targets, cfg, out)
        print(f"[grab_sensor] grasp overlay saved: {out.resolve()}")

    return capture, targets, mask, detection


def _save_detection_debug(cfg, capture, mask, detection, targets: GraspTargets) -> None:
    out = cfg.save_debug_dir
    assert out is not None
    out.mkdir(parents=True, exist_ok=True)

    overlay = capture.color_rgb.copy()
    overlay[mask] = (overlay[mask].astype(np.float32) * 0.5 + np.array([0, 255, 0]) * 0.5).astype(
        np.uint8
    )
    Image.fromarray(overlay).save(out / "segmentation_overlay.png")

    meta = {
        "centroid_cam_m": detection.centroid_cam_m.tolist(),
        "object_rpy_cam_deg": np.degrees(detection.object_rpy_cam_rad).tolist(),
        "object_yaw_cam_deg": float(np.degrees(detection.yaw_rad)),
        "n_points": detection.n_points,
        "bbox_xy": detection.bbox_xy,
        "grasp_pos_robot_m": targets.grasp_pos_m.tolist(),
        "grasp_rpy_robot_deg": np.degrees(targets.grasp_rpy_rad).tolist(),
        "pre_grasp_pos_robot_m": targets.pre_grasp_pos_m.tolist(),
        "pre_grasp_rpy_robot_deg": np.degrees(targets.pre_grasp_rpy_rad).tolist(),
        "grasp_orientation_mode": targets.debug.get("grasp_orientation_mode"),
        "debug": targets.debug,
    }
    (out / "detection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_grab_sensor_pipeline(
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
    *,
    robot=None,
    start_joints_deg: list[float] | None = None,
) -> dict[str, Any]:
    cfg = apply_runtime_calibrations(cfg, params)
    _, targets = run_detection(cfg, params)[:2]

    print(
        f"[grab_sensor] object yaw (cam) ~{targets.debug.get('object_yaw_cam_deg', 0):.1f} deg  "
        f"mode={targets.debug.get('grasp_orientation_mode')}"
    )
    if cfg.grasp_position_from_table_base:
        from .table_position import (
            format_robot_in_table_display,
            format_table_xyz_display,
        )

        print(f"[grab_sensor] robot in table frame: {format_robot_in_table_display(cfg.robot_in_table_frame)}")
        summary = targets.debug.get("grasp_frame_summary")
        if summary:
            print(
                f"[grab_sensor] grasp target (table): "
                f"{format_table_xyz_display([v / 1000.0 for v in summary['table_xyz_mm']])}"
            )

    if start_joints_deg is None:
        from manual_control.mapping import HOME_JOINTS_DEG

        start_joints_deg = list(HOME_JOINTS_DEG)

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
            "Add --ik-gui for viz-style grasp IK check, --plan-gui to play the path, or run "
            "python -m high_level_jobs.subtasks.grab_sensor.view_grasp_table for table/robot/camera/grasp."
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
            baud = int(params.get("baudrate") or 115200)
            robot = RobotAsciiInterface(port=port, baudrate=baud)
            if not robot.connect():
                raise RuntimeError(f"Failed to connect to {port}")
            own_robot = True
            print(f"[grab_sensor] Connected to {port} for execute.")

        try:
            execute_on_robot(robot, trajectory, targets, cfg, exec_params)
            result["executed"] = True
        finally:
            if own_robot and robot is not None:
                from manual_control.robot_session import disconnect_robot

                disconnect_robot(robot)

    return result
