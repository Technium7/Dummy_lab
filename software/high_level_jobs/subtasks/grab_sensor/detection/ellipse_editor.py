"""PowerPoint-style ellipse editor: drag to create, handles to move/resize/rotate."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base_ellipse import (
    BaseEllipse,
    LABEL_BASE_BGR,
    LABEL_OVERLAY_ALPHA,
    _blend_layer,
    _require_cv2,
)

HandleKind = Literal["center", "corner", "rotate", "body"]
DragMode = Literal["create", "move", "rotate", "corner", "body"]

HANDLE_HIT_RADIUS = 10
MIN_AXIS_PX = 4.0
ROTATION_HANDLE_SCALE = 1.25

HANDLE_CORNER_BGR = (0, 220, 255)
HANDLE_CENTER_BGR = (0, 255, 0)
HANDLE_ROTATE_BGR = (255, 200, 0)
BOX_BGR = (180, 180, 180)


def _rotate_uv(cx: float, cy: float, lx: float, ly: float, angle_deg: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return cx + lx * c - ly * s, cy + lx * s + ly * c


def _image_to_local(cx: float, cy: float, u: float, v: float, angle_deg: float) -> tuple[float, float]:
    dx, dy = u - cx, v - cy
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return dx * c + dy * s, -dx * s + dy * c


def ellipse_from_axis_box(p0: tuple[int, int], p1: tuple[int, int], *, min_axis_px: float = MIN_AXIS_PX) -> BaseEllipse:
    """Create axis-aligned ellipse inscribed in the drag rectangle."""
    x0, y0 = p0
    x1, y1 = p1
    cx = int(round((x0 + x1) * 0.5))
    cy = int(round((y0 + y1) * 0.5))
    axis_a = max(min_axis_px, abs(x1 - x0) * 0.5)
    axis_b = max(min_axis_px, abs(y1 - y0) * 0.5)
    return BaseEllipse(center_uv=(cx, cy), axis_a_px=axis_a, axis_b_px=axis_b, angle_deg=0.0)


def ellipse_handles(ellipse: BaseEllipse) -> dict[str, tuple[float, float]]:
    cx, cy = float(ellipse.center_uv[0]), float(ellipse.center_uv[1])
    a, b = float(ellipse.axis_a_px), float(ellipse.axis_b_px)
    angle = float(ellipse.angle_deg)
    return {
        "center": (cx, cy),
        "rotate": _rotate_uv(cx, cy, a * ROTATION_HANDLE_SCALE, 0.0, angle),
        "corner_se": _rotate_uv(cx, cy, a, b, angle),
        "corner_sw": _rotate_uv(cx, cy, -a, b, angle),
        "corner_nw": _rotate_uv(cx, cy, -a, -b, angle),
        "corner_ne": _rotate_uv(cx, cy, a, -b, angle),
    }


def _dist_sq(u: float, v: float, px: float, py: float) -> float:
    dx, dy = u - px, v - py
    return dx * dx + dy * dy


def hit_test_ellipse(ellipse: BaseEllipse, uv: tuple[int, int], *, radius: int = HANDLE_HIT_RADIUS) -> tuple[HandleKind, str] | None:
    u, v = float(uv[0]), float(uv[1])
    r2 = float(radius * radius)
    handles = ellipse_handles(ellipse)

    rotate_u, rotate_v = handles["rotate"]
    if _dist_sq(u, v, rotate_u, rotate_v) <= r2 * 1.4:
        return "rotate", "rotate"

    for key in ("corner_se", "corner_sw", "corner_nw", "corner_ne"):
        hu, hv = handles[key]
        if _dist_sq(u, v, hu, hv) <= r2:
            return "corner", key

    cx, cy = handles["center"]
    if _dist_sq(u, v, cx, cy) <= r2:
        return "center", "center"

    lx, ly = _image_to_local(cx, cy, u, v, ellipse.angle_deg)
    if (lx / max(ellipse.axis_a_px, 1e-3)) ** 2 + (ly / max(ellipse.axis_b_px, 1e-3)) ** 2 <= 1.0:
        return "body", "body"

    return None


@dataclass
class EllipseDrag:
    mode: DragMode
    handle: str
    start_uv: tuple[int, int]
    start_ellipse: BaseEllipse
    create_anchor: tuple[int, int] | None = None


def apply_ellipse_drag(drag: EllipseDrag, current_uv: tuple[int, int]) -> BaseEllipse:
    start = drag.start_ellipse
    su, sv = drag.start_uv
    u, v = current_uv
    cx0, cy0 = float(start.center_uv[0]), float(start.center_uv[1])

    if drag.mode == "create" and drag.create_anchor is not None:
        return ellipse_from_axis_box(drag.create_anchor, current_uv)

    if drag.mode in ("move", "body"):
        du, dv = u - su, v - sv
        return BaseEllipse(
            center_uv=(int(round(cx0 + du)), int(round(cy0 + dv))),
            axis_a_px=start.axis_a_px,
            axis_b_px=start.axis_b_px,
            angle_deg=start.angle_deg,
        )

    if drag.mode == "rotate":
        angle = math.degrees(math.atan2(v - cy0, u - cx0))
        return BaseEllipse(
            center_uv=start.center_uv,
            axis_a_px=start.axis_a_px,
            axis_b_px=start.axis_b_px,
            angle_deg=angle,
        )

    if drag.mode == "corner":
        lx, ly = _image_to_local(cx0, cy0, float(u), float(v), start.angle_deg)
        return BaseEllipse(
            center_uv=start.center_uv,
            axis_a_px=max(MIN_AXIS_PX, abs(lx)),
            axis_b_px=max(MIN_AXIS_PX, abs(ly)),
            angle_deg=start.angle_deg,
        )

    return start


def draw_ellipse_editor(
    bgr: np.ndarray,
    ellipse: BaseEllipse | None,
    *,
    preview_ellipse: BaseEllipse | None = None,
    alpha: float = LABEL_OVERLAY_ALPHA,
) -> None:
    """Draw ellipse, rotated bounding box, center, and PPT-style handles."""
    cv2 = _require_cv2()
    draw_ell = preview_ellipse or ellipse
    if draw_ell is None:
        return

    layer = np.zeros_like(bgr)
    cx, cy = int(draw_ell.center_uv[0]), int(draw_ell.center_uv[1])
    axes = (
        max(2, int(round(draw_ell.axis_a_px))),
        max(2, int(round(draw_ell.axis_b_px))),
    )
    cv2.ellipse(
        layer,
        (cx, cy),
        axes,
        float(draw_ell.angle_deg),
        0,
        360,
        LABEL_BASE_BGR,
        1,
        cv2.LINE_AA,
    )
    cv2.drawMarker(layer, (cx, cy), HANDLE_CENTER_BGR, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), 3, HANDLE_CENTER_BGR, 1, cv2.LINE_AA)

    handles = ellipse_handles(draw_ell)
    corner_pts = np.array(
        [
            handles["corner_se"],
            handles["corner_sw"],
            handles["corner_nw"],
            handles["corner_ne"],
        ],
        dtype=np.int32,
    )
    cv2.polylines(layer, [corner_pts], True, BOX_BGR, 1, cv2.LINE_AA)

    ru, rv = handles["rotate"]
    cv2.line(layer, (cx, cy), (int(round(ru)), int(round(rv))), BOX_BGR, 1, cv2.LINE_AA)

    for key in ("corner_se", "corner_sw", "corner_nw", "corner_ne"):
        hu, hv = handles[key]
        cv2.rectangle(
            layer,
            (int(round(hu)) - 4, int(round(hv)) - 4),
            (int(round(hu)) + 4, int(round(hv)) + 4),
            HANDLE_CORNER_BGR,
            1,
            cv2.LINE_AA,
        )

    cv2.circle(layer, (int(round(ru)), int(round(rv))), 5, HANDLE_ROTATE_BGR, 1, cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), 5, HANDLE_CENTER_BGR, 1, cv2.LINE_AA)

    _blend_layer(bgr, layer, alpha)


class EllipseEditor:
    """Left-button ellipse editor with PPT-style handles."""

    def __init__(self) -> None:
        self.ellipse: BaseEllipse | None = None
        self._drag: EllipseDrag | None = None
        self._preview: BaseEllipse | None = None

    def clear(self) -> None:
        self.ellipse = None
        self._drag = None
        self._preview = None

    def set_ellipse(self, ellipse: BaseEllipse | None) -> None:
        self.ellipse = ellipse
        self._drag = None
        self._preview = None

    @property
    def preview_ellipse(self) -> BaseEllipse | None:
        if self._drag is not None:
            return self._preview
        return self.ellipse

    def is_valid(self) -> bool:
        ell = self.ellipse
        return (
            ell is not None
            and ell.axis_a_px >= MIN_AXIS_PX
            and ell.axis_b_px >= MIN_AXIS_PX
        )

    def on_mouse(self, event: int, uv: tuple[int, int], cv2) -> str | None:
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = hit_test_ellipse(self.ellipse, uv) if self.ellipse is not None else None
            if hit is not None:
                kind, handle = hit
                mode: DragMode = "move" if kind == "center" else kind  # type: ignore[assignment]
                self._drag = EllipseDrag(
                    mode=mode,
                    handle=handle,
                    start_uv=uv,
                    start_ellipse=deepcopy(self.ellipse),
                )
                self._preview = self.ellipse
                return None

            self._drag = EllipseDrag(
                mode="create",
                handle="create",
                start_uv=uv,
                start_ellipse=BaseEllipse.from_center_uv(uv, axis_a_px=0, axis_b_px=0),
                create_anchor=uv,
            )
            self._preview = ellipse_from_axis_box(uv, uv)
            return "Drag box to draw ellipse"

        if event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            self._preview = apply_ellipse_drag(self._drag, uv)
            return None

        if event == cv2.EVENT_LBUTTONUP and self._drag is not None:
            updated = apply_ellipse_drag(self._drag, uv)
            if self._drag.mode == "create":
                if updated.axis_a_px >= MIN_AXIS_PX and updated.axis_b_px >= MIN_AXIS_PX:
                    self.ellipse = updated
                    msg = (
                        f"Ellipse center ({updated.center_uv[0]}, {updated.center_uv[1]}) "
                        f"a={updated.axis_a_px:.0f} b={updated.axis_b_px:.0f}"
                    )
                else:
                    msg = "Ellipse too small — drag a larger box"
            else:
                self.ellipse = updated
                msg = (
                    f"Ellipse center ({updated.center_uv[0]}, {updated.center_uv[1]}) "
                    f"angle={updated.angle_deg:.0f}°"
                )
            self._drag = None
            self._preview = self.ellipse
            return msg

        return None
