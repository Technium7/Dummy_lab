from __future__ import annotations

from typing import Tuple

import numpy as np

from .types import SegmentationConfig


def _morph_close_open(mask: np.ndarray, kernel_px: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for segmentation morphology. "
            "Install with: pip install opencv-python"
        ) from exc

    k = max(1, int(kernel_px))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened.astype(bool)


def _largest_component(mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required. pip install opencv-python") from exc

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        raise RuntimeError("No foreground blob found after depth segmentation.")

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    comp = labels == best
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    return comp, (x, y, w, h)


def segment_table_object(depth_m: np.ndarray, cfg: SegmentationConfig) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    band = valid & (depth_m >= cfg.min_depth_m) & (depth_m <= cfg.max_depth_m)
    if int(band.sum()) < cfg.min_area_px:
        raise RuntimeError(
            f"Depth band [{cfg.min_depth_m:.2f}, {cfg.max_depth_m:.2f}] m has only "
            f"{int(band.sum())} px (min_area_px={cfg.min_area_px}). Adjust segmentation limits."
        )

    cleaned = _morph_close_open(band, cfg.morph_kernel_px)
    mask, bbox = _largest_component(cleaned)
    if int(mask.sum()) < cfg.min_area_px:
        raise RuntimeError(
            f"Largest blob area {int(mask.sum())} px < min_area_px={cfg.min_area_px}."
        )
    return mask, bbox


segment_sensor_mask = segment_table_object
