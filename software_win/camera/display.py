"""OpenCV display helpers shared by Zelux / Infinity live viewers."""

from __future__ import annotations

from dataclasses import dataclass
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


def create_resizable_window(cv2, title: str, *, width: int | None = None, height: int | None = None) -> None:
    """Create a freely resizable HighGUI window (no locked aspect ratio)."""
    flags = int(cv2.WINDOW_NORMAL)
    freeratio = getattr(cv2, "WINDOW_FREERATIO", None)
    if freeratio is not None:
        flags |= int(freeratio)
    cv2.namedWindow(title, flags)
    try:
        cv2.setWindowProperty(
            title,
            cv2.WND_PROP_ASPECT_RATIO,
            float(getattr(cv2, "WINDOW_FREERATIO", 1.0)),
        )
    except Exception:
        pass
    if width is not None and height is not None and width > 0 and height > 0:
        try:
            cv2.resizeWindow(title, int(width), int(height))
        except Exception:
            pass


def create_controls_window(cv2, title: str, *, width: int = 420) -> None:
    """Separate trackbar host so image windows are not width-locked by Win32 trackbars."""
    create_resizable_window(cv2, title, width=width, height=80)
    # Trackbars need an image buffer in the window before createTrackbar.
    cv2.imshow(title, np.zeros((40, int(width), 3), dtype=np.uint8))


def controls_title_for(image_title: str) -> str:
    return f"{image_title} controls"


@dataclass
class WindowSizeState:
    """Remembered image-window size so imshow cannot snap width/height back."""

    width: int = 0
    height: int = 0
    seeded: bool = False


def _read_window_size(cv2, title: str) -> Optional[tuple[int, int]]:
    try:
        _x, _y, w, h = cv2.getWindowImageRect(title)
        if w > 1 and h > 1:
            return int(w), int(h)
    except Exception:
        pass
    return None


def show_resizable(
    cv2,
    title: str,
    frame_bgr: np.ndarray,
    state: WindowSizeState,
    *,
    max_width: Optional[int] = None,
) -> None:
    """imshow a capped frame, then re-apply the user window size.

    Win32 HighGUI resets the HWND to the Mat size on every imshow. Keeping the
    Mat size stable and calling resizeWindow afterward lets free horizontal /
    vertical dragging stick. Adopt a new size when the window differs from what
    we last forced (user drag between frames).
    """
    frame = resize_max(cv2, frame_bgr, max_width)
    fh, fw = frame.shape[:2]

    if not state.seeded:
        state.width, state.height = fw, fh
        state.seeded = True
    else:
        cur = _read_window_size(cv2, title)
        if cur is not None:
            cw, ch = cur
            if abs(cw - state.width) > 2 or abs(ch - state.height) > 2:
                state.width, state.height = cw, ch

    cv2.imshow(title, frame)
    try:
        cv2.resizeWindow(title, int(state.width), int(state.height))
    except Exception:
        pass


def wait_quit_key(cv2, delay_ms: int = 1) -> bool:
    """Return True if user pressed Q or Esc."""
    key = cv2.waitKey(delay_ms) & 0xFF
    return key in (ord("q"), ord("Q"), 27)
