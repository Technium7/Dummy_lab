"""Two-stage vision: YOLO full frame -> crop -> grasp-point CNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .capture import CaptureResult
from .grasp_point_net import crop_region, load_grasp_model, predict_grasp_and_base_uv
from .pose import backproject_mask_to_points_cam, estimate_object_pose_pca
from .models import GrabSensorConfig, TableObjectDetection
from .yolo_detect import YoloDetection, detect_sensor_bbox


@dataclass(frozen=True)
class YoloGraspResult:
    mask: np.ndarray
    detection: TableObjectDetection
    grasp_uv: tuple[int, int]
    yolo: YoloDetection
    crop_origin: tuple[int, int]
    debug: dict[str, Any]
    base_uv: tuple[int, int] | None = None


def pixel_to_cam_m(
    u: int,
    v: int,
    depth_m: np.ndarray,
    intrinsics: Any,
    *,
    patch_radius_px: int = 3,
) -> np.ndarray:
    h, w = depth_m.shape
    r = max(0, int(patch_radius_px))
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[patch > 1e-4]
    if valid.size == 0:
        raise RuntimeError(f"No valid depth at grasp pixel ({u}, {v}).")
    z = float(np.median(valid))
    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.ppx)
    cy = float(intrinsics.ppy)
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def crop_with_padding(
    color_rgb: np.ndarray,
    bbox_xy: tuple[int, int, int, int],
    padding_px: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    x, y, bw, bh = bbox_xy
    pad = max(0, int(padding_px))
    img_h, img_w = color_rgb.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w, x + bw + pad)
    y1 = min(img_h, y + bh + pad)
    return color_rgb[y0:y1, x0:x1].copy(), (x0, y0)


def map_crop_uv_to_full(
    grasp_uv_crop: tuple[int, int],
    crop_origin: tuple[int, int],
    full_shape: tuple[int, int],
) -> tuple[int, int]:
    u = int(grasp_uv_crop[0] + crop_origin[0])
    v = int(grasp_uv_crop[1] + crop_origin[1])
    h, w = full_shape
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    return u, v


def bbox_mask(
    bbox_xy: tuple[int, int, int, int],
    shape: tuple[int, int],
    depth_m: np.ndarray | None = None,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    x, y, bw, bh = bbox_xy
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    x = max(0, x)
    y = max(0, y)
    region = mask[y:y2, x:x2]
    region[:] = True
    if depth_m is not None:
        mask &= depth_m > 1e-4
    return mask


def _estimate_orientation(
    cfg: GrabSensorConfig,
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Any,
    grasp_cam_m: np.ndarray,
) -> tuple[np.ndarray, float]:
    if cfg.grasp_orientation_mode != "follow_object":
        return np.zeros(3, dtype=np.float64), 0.0

    try:
        points = backproject_mask_to_points_cam(mask, depth_m, intrinsics)
        if len(points) < 20:
            raise RuntimeError("too few points")
        _, rpy, yaw = estimate_object_pose_pca(points)
        return rpy, yaw
    except RuntimeError:
        return np.zeros(3, dtype=np.float64), 0.0


def detect_with_yolo_grasp(capture: CaptureResult, cfg: GrabSensorConfig) -> YoloGraspResult:
    """YOLO full frame -> RGB+depth crop -> grasp net -> 3D grasp point."""
    yolo = detect_sensor_bbox(capture.color_rgb, cfg.yolo)
    crop_rgb, crop_origin = crop_region(
        capture.color_rgb,
        yolo.bbox_xy,
        cfg.yolo.crop_padding_px,
    )
    crop_depth, _ = crop_region(
        capture.depth_m,
        yolo.bbox_xy,
        cfg.yolo.crop_padding_px,
    )

    grasp_model, model_meta = load_grasp_model(cfg.grasp_net.model_path)
    img_h, img_w = capture.color_rgb.shape[:2]

    grasp_uv_crop, base_uv_crop = predict_grasp_and_base_uv(
        grasp_model,
        crop_rgb,
        crop_depth,
        model_meta=model_meta,
        depth_min_m=cfg.grasp_net.depth_min_m,
        depth_max_m=cfg.grasp_net.depth_max_m,
    )
    grasp_uv = map_crop_uv_to_full(grasp_uv_crop, crop_origin, (img_h, img_w))
    base_uv = (
        map_crop_uv_to_full(base_uv_crop, crop_origin, (img_h, img_w))
        if base_uv_crop is not None
        else None
    )
    grasp_source = "crop_rgbd"

    grasp_cam_m = pixel_to_cam_m(
        grasp_uv[0],
        grasp_uv[1],
        capture.depth_m,
        capture.color_intrinsics,
        patch_radius_px=cfg.grasp_net.depth_patch_radius_px,
    )
    mask = bbox_mask(yolo.bbox_xy, (img_h, img_w), capture.depth_m)
    object_rpy, yaw = _estimate_orientation(
        cfg,
        mask,
        capture.depth_m,
        capture.color_intrinsics,
        grasp_cam_m,
    )

    detection = TableObjectDetection(
        centroid_cam_m=grasp_cam_m,
        yaw_rad=yaw,
        object_rpy_cam_rad=object_rpy,
        mask=mask,
        n_points=int(mask.sum()),
        bbox_xy=yolo.bbox_xy,
    )

    debug = {
        "vision_mode": "yolo_grasp",
        "yolo_confidence": yolo.confidence,
        "yolo_class": yolo.class_name,
        "yolo_bbox_xy": yolo.bbox_xy,
        "grasp_uv": list(grasp_uv),
        "grasp_cam_m": grasp_cam_m.round(4).tolist(),
        "crop_origin": list(crop_origin),
        "grasp_uv_crop": list(grasp_uv_crop),
        "grasp_source": grasp_source,
    }
    if base_uv is not None:
        try:
            base_cam_m = pixel_to_cam_m(
                base_uv[0],
                base_uv[1],
                capture.depth_m,
                capture.color_intrinsics,
                patch_radius_px=cfg.grasp_net.depth_patch_radius_px,
            )
            debug["base_uv"] = list(base_uv)
            debug["base_uv_crop"] = list(base_uv_crop) if base_uv_crop else None
            debug["base_cam_m"] = base_cam_m.round(4).tolist()
            from .table_position import (
                infer_base_table_pose,
                infer_cam_point_table_pose,
                resolve_table_calib_path,
                table_pose_offset_mm,
            )

            table_calib = resolve_table_calib_path(cfg.table_calib_path)
            if table_calib is not None:
                pose = infer_base_table_pose(base_cam_m, table_calib)
                debug["base_table_xyz_m"] = [round(v, 4) for v in pose["table_xyz_m"]]
                debug["base_table_xy_mm"] = [round(v, 2) for v in pose["table_xy_mm"]]
                debug["table_calib_path"] = pose["table_calib_path"]
                grasp_pose = infer_cam_point_table_pose(grasp_cam_m, table_calib)
                offset = table_pose_offset_mm(pose, grasp_pose)
                debug["grasp_table_xyz_m"] = [round(v, 4) for v in grasp_pose["table_xyz_m"]]
                debug["grasp_table_xy_mm"] = [round(v, 2) for v in grasp_pose["table_xy_mm"]]
                debug["grasp_offset_from_base_mm"] = {
                    k: round(v, 2) for k, v in offset.items()
                }
        except RuntimeError:
            debug["base_uv"] = list(base_uv)
            debug["base_uv_crop"] = list(base_uv_crop) if base_uv_crop else None
    return YoloGraspResult(
        mask=mask,
        detection=detection,
        grasp_uv=grasp_uv,
        yolo=yolo,
        crop_origin=crop_origin,
        debug=debug,
        base_uv=base_uv,
    )


detect_yolo_grasp = detect_with_yolo_grasp
