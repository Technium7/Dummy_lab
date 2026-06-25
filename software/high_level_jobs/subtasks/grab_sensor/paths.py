"""Production vision paths (top/overhead detection)."""

from __future__ import annotations

from pathlib import Path

_PKG = Path(__file__).resolve().parent

TOP_DETECTION_ROOT = _PKG / "detection" / "top_detection"
TOP_DATA_DIR = TOP_DETECTION_ROOT / "dataset" / "base_keypoints"
TOP_MODEL_DIR = TOP_DATA_DIR / "model"
TOP_YOLO_MODEL = TOP_MODEL_DIR / "sensor_yolo.pt"
TOP_BASE_MODEL = TOP_MODEL_DIR / "base_point.pt"

SIDE_DETECTION_ROOT = _PKG / "detection" / "side_detection"
SIDE_DATA_DIR = SIDE_DETECTION_ROOT / "dataset" / "grasp_keypoints"
SIDE_MODEL_DIR = SIDE_DATA_DIR / "model"
