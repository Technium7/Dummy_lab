"""Path helpers for Windows grab_sensor."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SOFTWARE_WIN_DIR = PACKAGE_DIR.parents[1]
REPO_ROOT = PACKAGE_DIR.parents[2]
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "default_grasp.yaml"
DEFAULT_ROBOT_IN_TABLE_YAML = REPO_ROOT / "physical_env" / "robot_in_table.yaml"
DEBUG_DIR = PACKAGE_DIR / "debug"
VISION_DIR = PACKAGE_DIR / "vision"
DATASET_DIR = VISION_DIR / "dataset"
MODEL_DIR = DATASET_DIR / "model"
