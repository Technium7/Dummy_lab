"""Capture helpers for Windows grab_sensor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from camera.realsense import RealSenseStreamer


@dataclass
class CaptureResult:
    color_rgb: np.ndarray
    depth_m: np.ndarray
    depth_scale: float
    color_intrinsics: Any
    device_name: str = ""
    device_serial: str = ""


def validate_color_image_file(color_path: Path) -> None:
    path = color_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Color capture not found: {path}")
    if path.stat().st_size < 1024:
        raise OSError(f"Color capture looks truncated ({path.stat().st_size} bytes): {path}")
    with Image.open(path) as im:
        im.load()
        if im.size[0] < 32 or im.size[1] < 32:
            raise OSError(f"Color capture has invalid size {im.size}: {path}")


def capture_realsense_frame(
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup: int = 15,
    serial: str = "",
) -> CaptureResult:
    with RealSenseStreamer(
        width=width, height=height, fps=fps, serial=serial, warmup=warmup
    ) as streamer:
        frame = streamer.read(with_depth=True)
        return CaptureResult(
            color_rgb=frame.color_rgb,
            depth_m=frame.depth_m,
            depth_scale=frame.depth_scale,
            color_intrinsics=frame.color_intrinsics,
            device_name=frame.device_name,
            device_serial=frame.device_serial,
        )


def load_capture_from_files(
    color_path: Path,
    depth_path: Path | None = None,
    *,
    depth_scale: float = 0.001,
) -> CaptureResult:
    validate_color_image_file(color_path)
    color_rgb = np.asarray(Image.open(color_path).convert("RGB"))
    if depth_path is not None and depth_path.is_file():
        depth_m = np.load(depth_path)
    else:
        depth_m = np.zeros(color_rgb.shape[:2], dtype=np.float64)
    return CaptureResult(
        color_rgb=color_rgb,
        depth_m=depth_m,
        depth_scale=depth_scale,
        color_intrinsics=None,
    )


def save_debug_capture(capture: CaptureResult, out_dir: Path, *, stem: str = "grab_sensor") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(capture.color_rgb).save(out_dir / f"{stem}.color.png")
    np.save(out_dir / f"{stem}.depth.npy", capture.depth_m.astype(np.float32))
    meta = {
        "device_name": capture.device_name,
        "device_serial": capture.device_serial,
        "depth_scale": capture.depth_scale,
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_grasp_overlay(
    capture: CaptureResult,
    objects: list[Any],
    targets: Any,
    out_path: Path,
) -> None:
    import cv2

    bgr = cv2.cvtColor(capture.color_rgb, cv2.COLOR_RGB2BGR)
    from high_level_jobs.grab_sensor.vision.live_label import _draw_object

    for obj in objects:
        drawn = {
            "class": getattr(obj, "class_name", "other"),
            "bbox_xyxy": getattr(obj, "bbox_xyxy", None),
            "kpts": getattr(obj, "kpts", []),
            "axis_deg": getattr(obj, "axis_deg", None),
        }
        _draw_object(cv2, bgr, drawn)

    guv = targets.debug.get("sensor_uv")
    if guv:
        cv2.circle(bgr, (int(guv[0]), int(guv[1])), 8, (0, 0, 255), 2)
        cv2.putText(
            bgr,
            "grasp",
            (int(guv[0]) + 10, int(guv[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    axis = targets.debug.get("sensor_axis_deg")
    if guv and axis is not None:
        kpts = targets.debug.get("sensor_kpts") or []
        if len(kpts) >= 2:
            from high_level_jobs.grab_sensor.vision.sensor_geometry import contact_edge_geometry

            bbox = targets.debug.get("sensor_bbox")
            geo = contact_edge_geometry(kpts, bbox_xyxy=bbox)
            cv2.arrowedLine(
                bgr,
                (int(geo.contact_uv[0]), int(geo.contact_uv[1])),
                (int(geo.corner_uv[0]), int(geo.corner_uv[1])),
                (0, 0, 255),
                2,
                tipLength=0.3,
            )
        else:
            import math

            ang = math.radians(float(axis))
            end = (guv[0] + 50 * math.cos(ang), guv[1] + 50 * math.sin(ang))
            cv2.arrowedLine(
                bgr,
                (int(guv[0]), int(guv[1])),
                (int(end[0]), int(end[1])),
                (0, 0, 255),
                2,
                tipLength=0.3,
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)
