"""Default paths for side-view (oblique D415) labeling and training."""

from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = PKG / "dataset" / "grasp_keypoints"
DEFAULT_YOLO_EXPORT_DIR = PKG / "dataset" / "sensor_yolo"
DEFAULT_YOLO_MODEL = DEFAULT_DATA_DIR / "model" / "sensor_yolo.pt"
DEFAULT_GRASP_MODEL = DEFAULT_DATA_DIR / "model" / "grasp_point.pt"
DEFAULT_BASE_MODEL = DEFAULT_DATA_DIR / "model" / "base_point.pt"
