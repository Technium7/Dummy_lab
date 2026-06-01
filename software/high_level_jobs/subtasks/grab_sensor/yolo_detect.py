"""YOLO object detection wrapper for table sensor localization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .types import YoloConfig


@dataclass(frozen=True)
class YoloDetection:
    bbox_xy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLO detection. Install with: pip install ultralytics"
        ) from exc
    return YOLO


@lru_cache(maxsize=4)
def _load_yolo(model_path: str):
    YOLO = _require_ultralytics()
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {path}\n"
            "Export labels and train first:\n"
            "  python -m high_level_jobs.subtasks.grab_sensor.export_yolo_dataset\n"
            "  python -m high_level_jobs.subtasks.grab_sensor.train_yolo_model"
        )
    return YOLO(str(path))


def detect_sensor_bbox(color_rgb: np.ndarray, cfg: YoloConfig) -> YoloDetection:
    """Run one full-frame YOLO inference and return the best sensor detection."""
    import cv2

    model = _load_yolo(str(cfg.model_path.resolve()))
    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(
        bgr,
        conf=float(cfg.confidence),
        iou=float(cfg.iou),
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO returned no results.")

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError(
            f"No sensor detected (conf >= {cfg.confidence}). "
            "Adjust scene, retrain YOLO, or lower yolo.confidence in config."
        )

    names = result.names or {}
    best: YoloDetection | None = None
    for box in boxes:
        cls_id = int(box.cls.item())
        name = str(names.get(cls_id, str(cls_id)))
        if cfg.class_name and name != cfg.class_name:
            continue
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x0 = int(max(0, np.floor(x1)))
        y0 = int(max(0, np.floor(y1)))
        x1i = int(min(color_rgb.shape[1], np.ceil(x2)))
        y1i = int(min(color_rgb.shape[0], np.ceil(y2)))
        w = max(1, x1i - x0)
        h = max(1, y1i - y0)
        det = YoloDetection(
            bbox_xy=(x0, y0, w, h),
            confidence=conf,
            class_id=cls_id,
            class_name=name,
        )
        if best is None or det.confidence > best.confidence:
            best = det

    if best is None:
        raise RuntimeError(
            f"No '{cfg.class_name}' detection above confidence {cfg.confidence}."
        )
    return best
