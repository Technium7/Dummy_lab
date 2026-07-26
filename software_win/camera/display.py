"""OpenCV display helpers shared by Zelux / Infinity live viewers."""

from __future__ import annotations

from typing import Optional

import numpy as np


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for camera live view. Install with:\n"
            "  pip install opencv-python"
        ) from exc
    return cv2


def to_bgr_u8(frame: np.ndarray, *, rgb: bool = False) -> np.ndarray:
    """Normalize mono / RGB / BGR arrays to contiguous HxWx3 uint8 BGR.

    Args:
        frame: Raw camera array.
        rgb: If True, treat 3-channel input as RGB and convert to BGR.
    """
    if frame is None:
        raise ValueError("frame is None")
    arr = np.asarray(frame)
    if arr.ndim == 2:
        u8 = _scale_to_u8(arr)
        return np.stack([u8, u8, u8], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported frame shape: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = _scale_to_u8(arr)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if rgb:
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def _scale_to_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    flat = arr.astype(np.float32)
    lo = float(np.percentile(flat, 1.0))
    hi = float(np.percentile(flat, 99.5))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    scaled = np.clip((flat - lo) * (255.0 / (hi - lo)), 0, 255)
    return scaled.astype(np.uint8)


def label_frame(cv2, frame_bgr: np.ndarray, text: str) -> np.ndarray:
    out = frame_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def resize_max(cv2, frame_bgr: np.ndarray, max_width: Optional[int]) -> np.ndarray:
    if not max_width or frame_bgr.shape[1] <= max_width:
        return frame_bgr
    scale = max_width / float(frame_bgr.shape[1])
    new_size = (int(frame_bgr.shape[1] * scale), int(frame_bgr.shape[0] * scale))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def wait_quit_key(cv2, delay_ms: int = 1) -> bool:
    """Return True if user pressed Q or Esc."""
    key = cv2.waitKey(delay_ms) & 0xFF
    return key in (ord("q"), ord("Q"), 27)
