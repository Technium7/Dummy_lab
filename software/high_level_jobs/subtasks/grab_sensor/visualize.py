from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .capture import CaptureResult
from .grasp import camera_to_robot_matrix
from .table_position import (
    draw_base_table_annotation,
    draw_table_grid_overlay,
    infer_base_table_pose,
    infer_cam_point_table_pose,
    load_table_calib,
    resolve_table_calib_path,
    robot_gui_to_table_m,
    table_point_to_cam_m,
    table_point_to_pixel,
    table_pose_offset_mm,
    table_xy_to_grid_indices,
)
from .transforms import invert_transform, pose_to_matrix
from .models import GrabSensorConfig, GraspTargets, TableObjectDetection


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for grasp overlay. pip install opencv-python-headless") from exc
    return cv2


def cam_point_to_pixel(point_cam_m: np.ndarray, intrinsics: Any) -> Tuple[int, int] | None:
    x, y, z = float(point_cam_m[0]), float(point_cam_m[1]), float(point_cam_m[2])
    if z <= 1e-4:
        return None
    u = int(round(intrinsics.fx * x / z + intrinsics.ppx))
    v = int(round(intrinsics.fy * y / z + intrinsics.ppy))
    return u, v


def robot_point_to_cam(point_robot_m: np.ndarray, t_robot_cam: np.ndarray) -> np.ndarray:
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(point_robot_m, dtype=np.float64)[:3]
    t_cam_robot = invert_transform(t_robot_cam)
    return (t_cam_robot @ p)[:3]


def grasp_point_cam_m(
    targets: GraspTargets,
    cfg: GrabSensorConfig,
) -> np.ndarray:
    if targets.debug.get("grasp_position_source") == "table_base_xy_height":
        grasp_table = targets.debug.get("grasp_table_xyz_m")
        table_calib = resolve_table_calib_path(cfg.table_calib_path)
        if grasp_table and len(grasp_table) >= 3 and table_calib is not None:
            return table_point_to_cam_m(
                (float(grasp_table[0]), float(grasp_table[1]), float(grasp_table[2])),
                table_calib,
            )
    t_robot_cam = camera_to_robot_matrix(cfg.camera_to_robot)
    return robot_point_to_cam(np.asarray(targets.grasp_pos_m, dtype=np.float64), t_robot_cam)


def pre_grasp_point_cam_m(
    targets: GraspTargets,
    cfg: GrabSensorConfig,
) -> np.ndarray:
    if targets.debug.get("grasp_position_source") == "table_base_xy_height":
        table_calib = resolve_table_calib_path(cfg.table_calib_path)
        if table_calib is not None:
            pre_table = robot_gui_to_table_m(targets.pre_grasp_pos_m, cfg.robot_in_table_frame)
            return table_point_to_cam_m(
                (float(pre_table[0]), float(pre_table[1]), float(pre_table[2])),
                table_calib,
            )
    t_robot_cam = camera_to_robot_matrix(cfg.camera_to_robot)
    return robot_point_to_cam(np.asarray(targets.pre_grasp_pos_m, dtype=np.float64), t_robot_cam)


def render_grasp_overlay(
    capture: CaptureResult,
    mask: np.ndarray,
    detection: TableObjectDetection,
    targets: GraspTargets,
    cfg: GrabSensorConfig,
) -> np.ndarray:
    cv2 = _require_cv2()
    intr = capture.color_intrinsics

    img = capture.color_rgb.copy()
    tint = img.astype(np.float32)
    tint[mask] = tint[mask] * 0.45 + np.array([80, 220, 80], dtype=np.float32) * 0.55
    img = np.clip(tint, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    base_uv = targets.debug.get("base_uv")
    table_calib_path = resolve_table_calib_path(cfg.table_calib_path)
    table_pose: dict | None = None
    calib_data: dict | None = None
    if table_calib_path is not None:
        calib_data = load_table_calib(table_calib_path)
        near_xy = None
        if targets.debug.get("base_table_xyz_m"):
            xyz = targets.debug["base_table_xyz_m"]
            near_xy = (float(xyz[0]), float(xyz[1]))
        elif base_uv and len(base_uv) == 2 and targets.debug.get("base_cam_m"):
            try:
                table_pose = infer_base_table_pose(
                    np.asarray(targets.debug["base_cam_m"], dtype=np.float64),
                    table_calib_path,
                )
                near_xy = (table_pose["table_xyz_m"][0], table_pose["table_xyz_m"][1])
            except (KeyError, ValueError, OSError):
                table_pose = None
        draw_table_grid_overlay(img, calib_data, near_table_xy=near_xy)

    x, y, w, h = detection.bbox_xy
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 0), 2)

    object_px = cam_point_to_pixel(detection.centroid_cam_m, intr)
    grasp_px = cam_point_to_pixel(grasp_point_cam_m(targets, cfg), intr)
    pre_px = cam_point_to_pixel(pre_grasp_point_cam_m(targets, cfg), intr)

    grasp_uv = targets.debug.get("grasp_uv")
    nn_grasp_px = None
    if grasp_uv and len(grasp_uv) == 2:
        nn_grasp_px = (int(grasp_uv[0]), int(grasp_uv[1]))

    def _draw_target(bgr_img, px, color, label, radius=10):
        if px is None:
            return
        u, v = px
        h_img, w_img = bgr_img.shape[:2]
        if not (0 <= u < w_img and 0 <= v < h_img):
            return
        cv2.circle(bgr_img, (u, v), radius + 6, (255, 255, 255), 2)
        cv2.circle(bgr_img, (u, v), radius, color, -1)
        cv2.drawMarker(bgr_img, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(
            bgr_img,
            label,
            (u + 12, v - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    _draw_target(img, object_px, (255, 180, 0), "object center", radius=7)
    _draw_target(img, pre_px, (255, 120, 0), "pre-grasp", radius=8)
    if targets.debug.get("vision_mode") == "yolo_grasp" and nn_grasp_px is not None:
        _draw_target(img, nn_grasp_px, (0, 0, 255), "GRASP (NN)", radius=12)
        if grasp_px is not None and (
            abs(grasp_px[0] - nn_grasp_px[0]) > 8 or abs(grasp_px[1] - nn_grasp_px[1]) > 8
        ):
            _draw_target(img, grasp_px, (255, 0, 255), "robot IK", radius=8)
    else:
        _draw_target(img, grasp_px, (0, 0, 255), "GRASP", radius=12)

    if nn_grasp_px and targets.debug.get("vision_mode") != "yolo_grasp":
        gu, gv = int(grasp_uv[0]), int(grasp_uv[1])
        cv2.drawMarker(img, (gu, gv), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(
            img,
            "NN grasp px",
            (gu + 12, gv + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if base_uv and len(base_uv) == 2:
        bu, bv = int(base_uv[0]), int(base_uv[1])
        if table_calib_path is not None and calib_data is not None:
            if table_pose is None and targets.debug.get("base_cam_m"):
                try:
                    table_pose = infer_base_table_pose(
                        np.asarray(targets.debug["base_cam_m"], dtype=np.float64),
                        table_calib_path,
                    )
                except (KeyError, ValueError, OSError):
                    table_pose = None
            if table_pose is None and targets.debug.get("base_table_xyz_m"):
                xyz = targets.debug["base_table_xyz_m"]
                pitch_m = float(calib_data.get("grid_config", {}).get("pitch_m", 0.0254))
                tx, ty, tz = float(xyz[0]), float(xyz[1]), float(xyz[2])
                gcol, grow, col_f, row_f = table_xy_to_grid_indices(tx, ty, pitch_m)
                table_pose = {
                    "table_xyz_m": [tx, ty, tz],
                    "table_xy_mm": [tx * 1000.0, ty * 1000.0],
                    "table_xy_in": [tx / 0.0254, ty / 0.0254],
                    "grid_col": gcol,
                    "grid_row": grow,
                    "grid_col_f": col_f,
                    "grid_row_f": row_f,
                }
            table_surface_uv = None
            grasp_pose = None
            grasp_offset_mm = None
            grasp_uv_tuple = None
            if grasp_uv and len(grasp_uv) == 2:
                grasp_uv_tuple = (int(grasp_uv[0]), int(grasp_uv[1]))
            if table_pose is not None:
                table_surface_uv = table_point_to_pixel(
                    (table_pose["table_xyz_m"][0], table_pose["table_xyz_m"][1], 0.0),
                    calib_data,
                    intr,
                )
                if targets.debug.get("grasp_cam_m"):
                    try:
                        grasp_pose = infer_cam_point_table_pose(
                            np.asarray(targets.debug["grasp_cam_m"], dtype=np.float64),
                            table_calib_path,
                        )
                        grasp_offset_mm = table_pose_offset_mm(table_pose, grasp_pose)
                    except (KeyError, ValueError, OSError):
                        grasp_pose = None
                        grasp_offset_mm = None
                draw_base_table_annotation(
                    img,
                    base_uv=(bu, bv),
                    table_pose=table_pose,
                    table_surface_uv=table_surface_uv,
                    grasp_uv=grasp_uv_tuple,
                    grasp_pose=grasp_pose,
                    grasp_offset_mm=grasp_offset_mm,
                )
            else:
                _draw_target(img, (bu, bv), (0, 255, 0), "BASE (NN)", radius=10)
        else:
            _draw_target(img, (bu, bv), (0, 255, 0), "BASE (NN)", radius=10)

    if object_px and grasp_px:
        cv2.line(img, object_px, grasp_px, (0, 255, 255), 2, cv2.LINE_AA)

    # Object +X axis in image (grasp orientation hint)
    axis_len_m = 0.03
    rot = pose_to_matrix(detection.centroid_cam_m, detection.object_rpy_cam_rad)
    tip_cam = rot @ np.array([axis_len_m, 0.0, 0.0, 1.0])
    tip_px = cam_point_to_pixel(tip_cam[:3], intr)
    if object_px and tip_px:
        cv2.arrowedLine(img, object_px, tip_px, (200, 100, 255), 2, tipLength=0.25)

    if targets.debug.get("vision_mode") == "yolo_grasp":
        legend = [
            "Green tint = YOLO region",
            "Yellow box = YOLO bbox",
            "Red = NN grasp pixel",
            "Green X = sensor base on table (NN)",
            "Gray grid = optical table calibration",
        ]
        if table_calib_path is not None:
            legend.append(f"Table calib: {table_calib_path.name}")
    else:
        legend = [
            "Green = segmented table object",
            "Orange = object center (vision)",
            "Red = robot grasp target",
        ]
    y0 = 24
    if table_calib_path is not None:
        y0 = 88
    for i, line in enumerate(legend):
        cv2.putText(
            img,
            line,
            (10, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_grasp_overlay(
    capture: CaptureResult,
    mask: np.ndarray,
    detection: TableObjectDetection,
    targets: GraspTargets,
    cfg: GrabSensorConfig,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = render_grasp_overlay(capture, mask, detection, targets, cfg)
    Image.fromarray(overlay).save(out_path)
    return out_path
