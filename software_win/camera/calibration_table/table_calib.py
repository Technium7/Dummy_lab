"""Build and save table-grid calibration JSON from a solved grid observation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .grid_config import TableGridConfig, table_frame_grid_indices
from .grid_detect import GridObservation, invert_transform, rvec_tvec_to_matrix


def _matrix_to_rpy_deg(rot: np.ndarray) -> np.ndarray:
    sy = math.sqrt(float(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    if sy > 1e-8:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def _decompose_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return transform[:3, 3].copy(), _matrix_to_rpy_deg(transform[:3, :3])


def _hole_to_dict(hole) -> dict[str, Any]:
    table_col, table_row = table_frame_grid_indices(hole.grid_col, hole.grid_row)
    return {
        "grid_col": table_col,
        "grid_row": table_row,
        "image_px": list(hole.center_px),
        "radius_px": hole.radius_px,
        "table_xy_m": list(hole.table_xy_m),
        "is_origin": hole.is_origin,
    }


@dataclass
class TableCalibrationResult:
    grid_config: TableGridConfig
    observation: GridObservation
    transform_cam_table: np.ndarray
    transform_table_cam: np.ndarray
    camera_in_table_translation_m: np.ndarray
    camera_in_table_rpy_deg: np.ndarray
    table_origin_in_camera_m: np.ndarray
    table_origin_in_camera_rpy_deg: np.ndarray
    captured_utc: str
    multi_shot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        method = "optical_table_grid_multi_shot_pnp" if self.multi_shot else "optical_table_grid_pnp"
        out = {
            "method": method,
            "description": (
                "Camera extrinsics relative to optical table grid. "
                "Table origin is the chosen hole (0,0); +X/+Y follow grid column/row "
                "(same sense as robot base XY); +Z is up from the table. "
                "Use transform_table_cam / transform_cam_table for geometry; rpy_deg is XYZ Euler decomposition only."
            ),
            "captured_utc": self.captured_utc,
            "grid_config": self.grid_config.to_dict(),
            "metrics": {
                "holes_detected": len(self.observation.holes),
                "reprojection_error_px": float(self.observation.reprojection_error_px),
                "pitch_px": float(self.observation.pitch_px)
                if np.isfinite(self.observation.pitch_px)
                else None,
            },
            "table_to_camera": {
                "translation_m": self.table_origin_in_camera_m.tolist(),
                "rpy_deg": self.table_origin_in_camera_rpy_deg.tolist(),
                "transform_cam_table": self.transform_cam_table.tolist(),
                "rvec": self.observation.rvec.tolist() if self.observation.rvec is not None else None,
                "tvec": self.observation.tvec.tolist() if self.observation.tvec is not None else None,
            },
            "camera_in_table_frame": {
                "translation_m": self.camera_in_table_translation_m.tolist(),
                "rpy_deg": self.camera_in_table_rpy_deg.tolist(),
                "transform_table_cam": self.transform_table_cam.tolist(),
            },
            "holes": [_hole_to_dict(h) for h in self.observation.holes],
        }
        if self.multi_shot is not None:
            out["multi_shot"] = self.multi_shot
        return out


def build_calibration_result(
    observation: GridObservation,
    grid_config: TableGridConfig,
    *,
    multi_shot: dict[str, Any] | None = None,
) -> TableCalibrationResult:
    if not observation.found or observation.rvec is None or observation.tvec is None:
        raise ValueError("Cannot build calibration from failed grid observation")

    transform_cam_table = rvec_tvec_to_matrix(observation.rvec, observation.tvec)
    transform_table_cam = invert_transform(transform_cam_table)

    table_t, table_rpy = _decompose_transform(transform_cam_table)
    cam_t, cam_rpy = _decompose_transform(transform_table_cam)

    return TableCalibrationResult(
        grid_config=grid_config,
        observation=observation,
        transform_cam_table=transform_cam_table,
        transform_table_cam=transform_table_cam,
        camera_in_table_translation_m=cam_t,
        camera_in_table_rpy_deg=cam_rpy,
        table_origin_in_camera_m=table_t,
        table_origin_in_camera_rpy_deg=table_rpy,
        captured_utc=datetime.now(timezone.utc).isoformat(),
        multi_shot=multi_shot,
    )


def save_calibration_result(
    result: TableCalibrationResult,
    out_dir: Path,
    *,
    color_bgr: np.ndarray | None = None,
    annotated_bgr: np.ndarray | None = None,
    stem: str = "table_calib",
    camera_settings: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if camera_settings:
        payload["camera"] = camera_settings
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if color_bgr is not None:
        _require_cv2().imwrite(str(out_dir / f"{stem}.color.png"), color_bgr)
    if annotated_bgr is not None:
        _require_cv2().imwrite(str(out_dir / f"{stem}.overlay.png"), annotated_bgr)
    return json_path


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for table calibration. "
            "Install with: pip install -r software_win/requirements.txt"
        ) from exc
    return cv2
