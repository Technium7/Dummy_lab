"""Sensor geometry: holder–sensor contact + directed holder orientation.

Label order (2 clicks):
  kpts[0] = holder–sensor contact
  kpts[1] = direction tip along the holder (toward grasp / free space)

axis_deg = atan2(dir - contact).
Grasp = contact + along_holder * offset (table mm preferred; px for overlay).
Head predicts contact (u,v) + (cos θ, sin θ).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

SENSOR_NUM_KPTS = 2
SENSOR_CONTACT_IDX = 0
SENSOR_DIR_IDX = 1

# Back-compat aliases
SENSOR_GRASP_IDX = SENSOR_CONTACT_IDX  # legacy name; first pt is contact
SENSOR_CORNER_IDX = SENSOR_DIR_IDX
SENSOR_HOLDER_IDX = SENSOR_CONTACT_IDX
SENSOR_TIP_IDX = SENSOR_CONTACT_IDX
SENSOR_RECT_KPTS = 0

DEFAULT_GRASP_OFFSET_PX = 18.0


@dataclass(frozen=True)
class ContactAxisGeometry:
    contact_uv: tuple[float, float]
    dir_uv: tuple[float, float]
    axis_deg: float
    along_uv: tuple[float, float]

    @property
    def grasp_uv(self) -> tuple[float, float]:
        """Alias used by older overlays — contact itself (offset applied separately)."""
        return self.contact_uv

    @property
    def corner_uv(self) -> tuple[float, float]:
        return self.dir_uv

    @property
    def outward_uv(self) -> tuple[float, float]:
        """Along holder (directed by second click) — used as grasp offset direction."""
        return self.along_uv

    @property
    def inward_uv(self) -> tuple[float, float]:
        return (-self.along_uv[0], -self.along_uv[1])


# Back-compat name
GraspAxisGeometry = ContactAxisGeometry


def _unit(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return 1.0, 0.0
    return dx / n, dy / n


def contact_uv(kpts: Sequence) -> tuple[float, float]:
    p = kpts[SENSOR_CONTACT_IDX]
    return float(p[0]), float(p[1])


def dir_uv(kpts: Sequence) -> tuple[float, float]:
    p = kpts[SENSOR_DIR_IDX]
    return float(p[0]), float(p[1])


def grasp_uv(kpts: Sequence) -> tuple[float, float]:
    """First keypoint (contact). Prefer grasp_uv_from_sensor_kpts for offset grasp."""
    return contact_uv(kpts)


def corner_uv(kpts: Sequence) -> tuple[float, float]:
    return dir_uv(kpts)


def holder_uv(kpts: Sequence) -> tuple[float, float]:
    return contact_uv(kpts)


def tip_uv(kpts: Sequence) -> tuple[float, float]:
    return contact_uv(kpts)


def rect_center_uv(kpts: Sequence) -> tuple[float, float]:
    return contact_uv(kpts) if kpts else (0.0, 0.0)


def axis_deg_from_contact_dir(kpts: Sequence) -> float:
    if len(kpts) < SENSOR_NUM_KPTS:
        return 0.0
    c = contact_uv(kpts)
    d = dir_uv(kpts)
    return float(math.degrees(math.atan2(d[1] - c[1], d[0] - c[0])))


def cos_sin_from_kpts(kpts: Sequence) -> tuple[float, float]:
    if len(kpts) < SENSOR_NUM_KPTS:
        return 1.0, 0.0
    c = contact_uv(kpts)
    d = dir_uv(kpts)
    return _unit(d[0] - c[0], d[1] - c[1])


def contact_axis_geometry(kpts: Sequence) -> ContactAxisGeometry:
    if len(kpts) < SENSOR_NUM_KPTS:
        raise ValueError("Need 2 keypoints (contact + direction along holder)")
    c = contact_uv(kpts)
    d = dir_uv(kpts)
    along = _unit(d[0] - c[0], d[1] - c[1])
    axis_deg = float(math.degrees(math.atan2(along[1], along[0])))
    return ContactAxisGeometry(
        contact_uv=c,
        dir_uv=d,
        axis_deg=axis_deg,
        along_uv=along,
    )


def grasp_axis_geometry(kpts: Sequence) -> ContactAxisGeometry:
    return contact_axis_geometry(kpts)


def contact_edge_geometry(
    kpts: Sequence,
    *,
    body_hint_uv: tuple[float, float] | None = None,
    bbox_xyxy: Sequence[float] | None = None,
) -> ContactAxisGeometry:
    _ = body_hint_uv, bbox_xyxy
    return contact_axis_geometry(kpts)


def holder_edge_geometry(
    kpts: Sequence,
    *,
    body_hint_uv: tuple[float, float] | None = None,
    bbox_xyxy: Sequence[float] | None = None,
) -> ContactAxisGeometry:
    return contact_edge_geometry(kpts, body_hint_uv=body_hint_uv, bbox_xyxy=bbox_xyxy)


def axis_deg_from_sensor_kpts(
    kpts: Sequence,
    *,
    bbox_xyxy: Sequence[float] | None = None,
) -> float:
    _ = bbox_xyxy
    if len(kpts) >= SENSOR_NUM_KPTS:
        return axis_deg_from_contact_dir(kpts)
    if len(kpts) >= 2:
        p0, p1 = kpts[0], kpts[1]
        return float(
            math.degrees(math.atan2(float(p1[1]) - float(p0[1]), float(p1[0]) - float(p0[0])))
        )
    return 0.0


def grasp_uv_from_sensor_kpts(
    kpts: Sequence,
    *,
    offset_px: float = DEFAULT_GRASP_OFFSET_PX,
    bbox_xyxy: Sequence[float] | None = None,
    body_hint_uv: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Image-space grasp preview: contact + along_holder * offset_px."""
    _ = bbox_xyxy, body_hint_uv
    if not kpts:
        raise ValueError("No keypoints")
    if len(kpts) < SENSOR_NUM_KPTS or abs(float(offset_px)) < 1e-6:
        return float(kpts[0][0]), float(kpts[0][1])
    geo = contact_axis_geometry(kpts)
    hx, hy = geo.contact_uv
    nx, ny = geo.along_uv
    d = float(offset_px)
    return hx + d * nx, hy + d * ny


def dir_tip_from_contact_axis(
    contact: tuple[float, float],
    axis_deg: float,
    *,
    length_px: float = 40.0,
) -> tuple[float, float]:
    rad = math.radians(float(axis_deg))
    return (
        float(contact[0]) + float(length_px) * math.cos(rad),
        float(contact[1]) + float(length_px) * math.sin(rad),
    )


def kpts_from_grasp_axis(
    grasp: tuple[float, float],
    axis_deg: float,
    *,
    length_px: float = 40.0,
) -> list[list[float]]:
    """Build [contact, dir_tip] from contact UV + axis (name kept for infer)."""
    tip = dir_tip_from_contact_axis(grasp, axis_deg, length_px=length_px)
    return [[float(grasp[0]), float(grasp[1])], [float(tip[0]), float(tip[1])]]


def normalize_sensor_kpts(kpts: Sequence) -> list[list[float]]:
    return [[float(p[0]), float(p[1])] for p in kpts[:SENSOR_NUM_KPTS]]
