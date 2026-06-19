"""Elliptical sensor-base counter: label format, geometry, and drawing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Default ellipse when only center is labeled (legacy point labels).
DEFAULT_AXIS_A_PX = 22.0
DEFAULT_AXIS_B_PX = 14.0
DEFAULT_ANGLE_DEG = 0.0
NUM_OUTPUTS_BASE_ELLIPSE = 5

NN_BASE_BGR = (0, 220, 0)
EDIT_BASE_BGR = (0, 255, 0)
ELLIPSE_BGR = (0, 200, 0)


@dataclass(frozen=True)
class BaseEllipse:
    """Ellipse in full-image pixels: center + OpenCV ellipse params."""

    center_uv: tuple[int, int]
    axis_a_px: float
    axis_b_px: float
    angle_deg: float

    @property
    def center_u(self) -> int:
        return int(self.center_uv[0])

    @property
    def center_v(self) -> int:
        return int(self.center_uv[1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "center_uv": [int(self.center_uv[0]), int(self.center_uv[1])],
            "axis_a_px": round(float(self.axis_a_px), 2),
            "axis_b_px": round(float(self.axis_b_px), 2),
            "angle_deg": round(float(self.angle_deg), 2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseEllipse | None:
        if not data:
            return None
        center = data.get("center_uv")
        if center is None or len(center) < 2:
            cu = data.get("center_u")
            cv = data.get("center_v")
            if cu is None or cv is None:
                return None
            center = [cu, cv]
        return cls(
            center_uv=(int(center[0]), int(center[1])),
            axis_a_px=float(data.get("axis_a_px", DEFAULT_AXIS_A_PX)),
            axis_b_px=float(data.get("axis_b_px", DEFAULT_AXIS_B_PX)),
            angle_deg=float(data.get("angle_deg", DEFAULT_ANGLE_DEG)),
        )

    @classmethod
    def from_center_uv(
        cls,
        center_uv: tuple[int, int],
        *,
        axis_a_px: float = DEFAULT_AXIS_A_PX,
        axis_b_px: float = DEFAULT_AXIS_B_PX,
        angle_deg: float = DEFAULT_ANGLE_DEG,
    ) -> BaseEllipse:
        return cls(
            center_uv=(int(center_uv[0]), int(center_uv[1])),
            axis_a_px=float(axis_a_px),
            axis_b_px=float(axis_b_px),
            angle_deg=float(angle_deg),
        )


def parse_base_ellipse(data: dict[str, Any]) -> BaseEllipse | None:
    """Read base ellipse from a label JSON (new or legacy center-only)."""
    if "base_ellipse" in data and data["base_ellipse"]:
        ell = BaseEllipse.from_dict(data["base_ellipse"])
        if ell is not None:
            return ell
    if "base_uv" in data and len(data["base_uv"]) >= 2:
        return BaseEllipse.from_center_uv((int(data["base_uv"][0]), int(data["base_uv"][1])))
    if "base_u" in data and "base_v" in data:
        return BaseEllipse.from_center_uv((int(data["base_u"]), int(data["base_v"])))
    return None


def ellipse_to_norm_crop(
    ellipse: BaseEllipse,
    crop_origin: tuple[int, int],
    crop_shape: tuple[int, int],
    letterbox_meta: Any,
) -> np.ndarray:
    """5-dim training target in letterbox-normalized crop space."""
    from .grasp_point_net import uv_to_norm_letterbox

    ox, oy = crop_origin
    cu = int(ellipse.center_uv[0]) - ox
    cv = int(ellipse.center_uv[1]) - oy
    cx_norm, cy_norm = uv_to_norm_letterbox((cu, cv), letterbox_meta)
    crop_h, crop_w = crop_shape
    max_dim = float(max(crop_w, crop_h, 1))
    return np.array(
        [
            cx_norm,
            cy_norm,
            min(0.5, float(ellipse.axis_a_px) / max_dim),
            min(0.5, float(ellipse.axis_b_px) / max_dim),
            float(ellipse.angle_deg) / 180.0,
        ],
        dtype=np.float32,
    )


def norm_crop_to_ellipse(
    pred: np.ndarray,
    crop_origin: tuple[int, int],
    crop_shape: tuple[int, int],
    letterbox_meta: Any,
) -> BaseEllipse:
    """Decode 5-dim network output to ellipse in full-image pixels."""
    from .grasp_point_net import norm_to_uv_letterbox

    crop_h, crop_w = crop_shape
    max_dim = float(max(crop_w, crop_h, 1))
    cu, cv = norm_to_uv_letterbox((float(pred[0]), float(pred[1])), letterbox_meta)
    ox, oy = crop_origin
    return BaseEllipse(
        center_uv=(int(cu) + ox, int(cv) + oy),
        axis_a_px=max(4.0, float(pred[2]) * max_dim),
        axis_b_px=max(3.0, float(pred[3]) * max_dim),
        angle_deg=float(pred[4]) * 180.0,
    )


def map_ellipse_crop_to_full(
    ellipse: BaseEllipse,
    crop_origin: tuple[int, int],
    full_shape: tuple[int, int],
) -> BaseEllipse:
    """Shift ellipse center from crop coords to full frame."""
    ox, oy = crop_origin
    h, w = full_shape
    u = int(np.clip(ellipse.center_uv[0] + ox, 0, w - 1))
    v = int(np.clip(ellipse.center_uv[1] + oy, 0, h - 1))
    return BaseEllipse(
        center_uv=(u, v),
        axis_a_px=ellipse.axis_a_px,
        axis_b_px=ellipse.axis_b_px,
        angle_deg=ellipse.angle_deg,
    )


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for base ellipse overlay") from exc
    return cv2


# Labeling GUIs: thin, semi-transparent overlays so the image stays visible.
LABEL_OVERLAY_ALPHA = 0.45
LABEL_GRASP_BGR = (0, 0, 255)
LABEL_BASE_BGR = (0, 255, 0)


def _blend_layer(bgr: np.ndarray, layer: np.ndarray, alpha: float) -> None:
    """Alpha-blend colored pixels from layer onto bgr."""
    mask = layer.max(axis=2) > 0
    if not np.any(mask):
        return
    blended = (
        alpha * layer.astype(np.float32) + (1.0 - alpha) * bgr.astype(np.float32)
    ).astype(np.uint8)
    bgr[mask] = blended[mask]


def draw_label_grasp(bgr: np.ndarray, u: int, v: int, *, alpha: float = LABEL_OVERLAY_ALPHA) -> None:
    cv2 = _require_cv2()

    def _draw(layer: np.ndarray) -> None:
        cv2.drawMarker(layer, (u, v), LABEL_GRASP_BGR, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        cv2.circle(layer, (u, v), 4, LABEL_GRASP_BGR, 1, cv2.LINE_AA)

    layer = np.zeros_like(bgr)
    _draw(layer)
    _blend_layer(bgr, layer, alpha)


def draw_base_ellipse(
    bgr: np.ndarray,
    ellipse: BaseEllipse,
    *,
    color: tuple[int, int, int] = ELLIPSE_BGR,
    center_color: tuple[int, int, int] = NN_BASE_BGR,
    thickness: int = 2,
    draw_center: bool = True,
) -> None:
    """Draw partially-visible counter ellipse with center cross."""
    cv2 = _require_cv2()
    u, v = int(ellipse.center_uv[0]), int(ellipse.center_uv[1])
    h_img, w_img = bgr.shape[:2]
    if not (0 <= u < w_img and 0 <= v < h_img):
        return
    axes = (
        max(2, int(round(ellipse.axis_a_px))),
        max(2, int(round(ellipse.axis_b_px))),
    )
    cv2.ellipse(
        bgr,
        (u, v),
        axes,
        float(ellipse.angle_deg),
        0,
        360,
        color,
        thickness,
        cv2.LINE_AA,
    )
    if draw_center:
        cv2.drawMarker(bgr, (u, v), center_color, cv2.MARKER_CROSS, 16, 2)
        cv2.circle(bgr, (u, v), 6, center_color, 1, cv2.LINE_AA)
