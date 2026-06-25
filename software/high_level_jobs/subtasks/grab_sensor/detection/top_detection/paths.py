"""Default paths for top/overhead view labeling and training."""

from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = PKG / "dataset" / "base_keypoints"
DEFAULT_YOLO_EXPORT_DIR = PKG / "dataset" / "sensor_yolo"
DEFAULT_YOLO_MODEL = DEFAULT_DATA_DIR / "model" / "sensor_yolo.pt"
DEFAULT_BASE_MODEL = DEFAULT_DATA_DIR / "model" / "base_point.pt"
