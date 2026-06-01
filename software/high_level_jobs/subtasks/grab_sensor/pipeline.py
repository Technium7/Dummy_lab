from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .capture import capture_realsense_frame, load_capture_from_files, save_debug_capture
from .config_loader import load_grab_sensor_config
from .execute import execute_on_robot
from .grasp import compute_grasp_targets
from .plan_motion import plan_approach_and_grasp
from .pose import detect_table_object
from .segment import segment_table_object
from .types import GrabSensorConfig, GraspTargets, VisionMode
from .visualize import save_grasp_overlay
from .yolo_grasp import detect_with_yolo_grasp


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


def run_detection(
    cfg: GrabSensorConfig,
    params: Mapping[str, Any],
) -> tuple[Any, GraspTargets, Any, Any]:
    color_file = params.get("color_file")
    depth_file = params.get("depth_file")

    if color_file:
        capture = load_capture_from_files(
            Path(str(color_file)),
            Path(str(depth_file)) if depth_file else None,
            depth_scale=float(params.get("depth_scale", 0.001)),
        )
    else:
        capture = capture_realsense_frame(
            width=int(params.get("width", 640)),
            height=int(params.get("height", 480)),
            fps=int(params.get("fps", 30)),
            warmup=int(params.get("warmup", 15)),
            serial=str(params.get("camera_serial", "")),
        )

    if cfg.save_debug_dir is not None:
        save_debug_capture(capture, cfg.save_debug_dir, stem="grab_sensor")

    mask, detection, vision_debug = _run_vision(cfg, capture, params)
    targets = compute_grasp_targets(cfg, detection)
    targets.debug.update(vision_debug)

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
    _, targets = run_detection(cfg, params)[:2]
    print(
        f"[grab_sensor] object yaw (cam) ~{targets.debug.get('object_yaw_cam_deg', 0):.1f} deg  "
        f"mode={targets.debug.get('grasp_orientation_mode')}  "
        f"grasp rpy (robot) deg={targets.debug.get('grasp_rpy_robot_deg')}"
    )
    print(f"[grab_sensor] grasp position (robot m)={targets.grasp_pos_m.round(4).tolist()}")
    print(f"[grab_sensor] pre-grasp (robot m)={targets.pre_grasp_pos_m.round(4).tolist()}")

    if start_joints_deg is None:
        from manual_control.mapping import HOME_JOINTS_DEG

        start_joints_deg = list(HOME_JOINTS_DEG)
        if robot is not None:
            from trajectory_run.run_planned_trajectory import read_physical_joints

            current = read_physical_joints(robot)
            if current is not None:
                start_joints_deg = current

    trajectory, plan_meta = plan_approach_and_grasp(
        cfg,
        start_joints_deg,
        targets,
        use_gui=bool(params.get("plan_gui", False)),
    )
    print(f"[grab_sensor] planned {plan_meta['n_total']} waypoints.")

    result: dict[str, Any] = {
        "targets": targets,
        "trajectory_rad": trajectory,
        "plan_meta": plan_meta,
        "start_joints_deg": start_joints_deg,
    }

    if bool(params.get("execute", False)) and robot is not None:
        exec_params = dict(params)
        exec_params["plan_meta"] = plan_meta
        execute_on_robot(robot, trajectory, targets, cfg, exec_params)
        result["executed"] = True

    return result
