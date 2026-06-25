"""Two-stage top-view vision: YOLO full frame -> crop -> base point CNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..base_ellipse import BaseEllipse
from ..grasp_point_net import crop_region, load_grasp_model, predict_grasp_uv
from ..yolo_detect import YoloDetection, detect_sensor_bbox
from ...models import GrabSensorConfig


@dataclass(frozen=True)
class YoloBaseInference:
    """YOLO + base point on a full RGBD frame (top/overhead view)."""

    yolo: YoloDetection
    base_uv: tuple[int, int]
    crop_origin: tuple[int, int]
    base_uv_crop: tuple[int, int]
    source: str = "base_point_net"


def map_crop_uv_to_full(
    uv_crop: tuple[int, int],
    crop_origin: tuple[int, int],
    full_shape: tuple[int, int],
) -> tuple[int, int]:
    ox, oy = crop_origin
    h, w = full_shape
    u = int(np.clip(uv_crop[0] + ox, 0, w - 1))
    v = int(np.clip(uv_crop[1] + oy, 0, h - 1))
    return u, v


def predict_base_on_crop(
    crop_rgb: np.ndarray,
    crop_depth: np.ndarray,
    *,
    base_model_path: Path,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[tuple[int, int], str]:
    """Predict base center pixel in a YOLO crop."""
    base_model, base_meta = load_grasp_model(base_model_path)
    base_uv = predict_grasp_uv(
        base_model,
        crop_rgb,
        crop_depth,
        model_meta=base_meta,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    source = (
        "base_point_net"
        if int(base_meta.get("num_outputs", 0)) <= 2
        or base_meta.get("target") == "base_point"
        else "base_ellipse_net"
    )
    return base_uv, source


def infer_yolo_base_points(
    color_rgb: np.ndarray,
    depth_m: np.ndarray,
    cfg: GrabSensorConfig,
) -> YoloBaseInference:
    """YOLO full frame -> crop -> base point CNN."""
    if cfg.yolo is None or cfg.grasp_net is None:
        raise ValueError("yolo_base requires yolo and grasp_net.base_model_path in config.")
    base_path = cfg.grasp_net.base_model_path
    if base_path is None or not base_path.is_file():
        raise FileNotFoundError(f"Base model not found: {base_path}")

    yolo = detect_sensor_bbox(color_rgb, cfg.yolo)
    crop_rgb, crop_origin = crop_region(
        color_rgb,
        yolo.bbox_xy,
        cfg.yolo.crop_padding_px,
    )
    crop_depth, _ = crop_region(
        depth_m,
        yolo.bbox_xy,
        cfg.yolo.crop_padding_px,
    )
    base_uv_crop, source = predict_base_on_crop(
        crop_rgb,
        crop_depth,
        base_model_path=base_path,
        depth_min_m=cfg.grasp_net.depth_min_m,
        depth_max_m=cfg.grasp_net.depth_max_m,
    )
    img_h, img_w = color_rgb.shape[:2]
    base_uv = map_crop_uv_to_full(base_uv_crop, crop_origin, (img_h, img_w))
    return YoloBaseInference(
        yolo=yolo,
        base_uv=base_uv,
        crop_origin=crop_origin,
        base_uv_crop=base_uv_crop,
        source=source,
    )


def infer_to_yolo_grasp_compat(
    color_rgb: np.ndarray,
    depth_m: np.ndarray,
    cfg: GrabSensorConfig,
) -> Any:
    """Adapt top-view inference to the shared YoloGraspInference shape (grasp = base)."""
    from ..yolo_grasp import YoloGraspInference

    nn = infer_yolo_base_points(color_rgb, depth_m, cfg)
    base_ellipse = BaseEllipse.from_center_uv(nn.base_uv)
    return YoloGraspInference(
        yolo=nn.yolo,
        grasp_uv=nn.base_uv,
        base_uv=nn.base_uv,
        base_ellipse=base_ellipse,
        crop_origin=nn.crop_origin,
        grasp_uv_crop=nn.base_uv_crop,
        base_ellipse_crop=BaseEllipse.from_center_uv(nn.base_uv_crop),
        grasp_source=nn.source,
    )
