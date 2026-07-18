"""Dense overhead depth → table-frame XYZ maps (no detection / YOLO)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from high_level_jobs.subtasks.grab_sensor.table_position import (
    load_transform_table_cam,
    resolve_table_calib_path,
)

# Matches grab_sensor D415 defaults at 640x480 (episodes do not store intrinsics).
DEFAULT_D415_INTRINSICS: dict[str, float] = {
    "width": 640.0,
    "height": 480.0,
    "fx": 610.814697265625,
    "fy": 610.5267944335938,
    "ppx": 314.0621337890625,
    "ppy": 234.38771057128906,
}


@dataclass(frozen=True)
class TableXyzBounds:
    xy_min_m: float = -0.40
    xy_max_m: float = 0.40
    z_min_m: float = -0.05
    z_max_m: float = 0.35


@dataclass
class TableXyzContext:
    """Cached calib + intrinsics for repeated frame conversion."""

    transform_table_cam: np.ndarray
    fx: float
    fy: float
    ppx: float
    ppy: float
    bounds: TableXyzBounds
    calib_path: Path

    @staticmethod
    def from_config(
        *,
        table_calib_path: str | Path | None = None,
        intrinsics: dict[str, float] | None = None,
        xy_min_m: float = -0.40,
        xy_max_m: float = 0.40,
        z_min_m: float = -0.05,
        z_max_m: float = 0.35,
    ) -> TableXyzContext:
        configured = Path(table_calib_path).resolve() if table_calib_path else None
        if configured is not None and not configured.is_file():
            configured = None
        path = resolve_table_calib_path(configured)
        if path is None:
            raise FileNotFoundError(
                "No valid table_calib.json found. Set table_calib_path or run "
                "software/camera/calibration_table."
            )
        intr = dict(DEFAULT_D415_INTRINSICS)
        if intrinsics:
            intr.update({k: float(v) for k, v in intrinsics.items()})
        return TableXyzContext(
            transform_table_cam=load_transform_table_cam(path),
            fx=float(intr["fx"]),
            fy=float(intr["fy"]),
            ppx=float(intr["ppx"]),
            ppy=float(intr["ppy"]),
            bounds=TableXyzBounds(
                xy_min_m=float(xy_min_m),
                xy_max_m=float(xy_max_m),
                z_min_m=float(z_min_m),
                z_max_m=float(z_max_m),
            ),
            calib_path=path,
        )

    def with_intrinsics(self, intrinsics: Any) -> TableXyzContext:
        """Prefer live RealSense intrinsics at deploy when available."""
        try:
            fx = float(intrinsics.fx)
            fy = float(intrinsics.fy)
            ppx = float(intrinsics.ppx)
            ppy = float(intrinsics.ppy)
        except (AttributeError, TypeError, ValueError):
            return self
        return TableXyzContext(
            transform_table_cam=self.transform_table_cam,
            fx=fx,
            fy=fy,
            ppx=ppx,
            ppy=ppy,
            bounds=self.bounds,
            calib_path=self.calib_path,
        )


def depth_to_table_xyz(
    depth_m: np.ndarray,
    ctx: TableXyzContext,
) -> np.ndarray:
    """
    Back-project depth to camera XYZ, then table frame.

    Returns ``(3, H, W)`` float32 table XYZ in meters. Invalid depth → 0.
    Call on the unflipped depth; if augmenting with a horizontal flip, flip the
    returned map the same way as RGB-D (``xyz[:, :, ::-1]``).
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    h, w = depth.shape[:2]
    vs, us = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    valid = depth > 1e-4
    z = depth
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    x[valid] = (us[valid] - ctx.ppx) * z[valid] / max(ctx.fx, 1e-6)
    y[valid] = (vs[valid] - ctx.ppy) * z[valid] / max(ctx.fy, 1e-6)

    cam = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    flat = cam.reshape(-1, 3)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([flat.astype(np.float64), ones], axis=1)
    table = (ctx.transform_table_cam @ hom.T).T[:, :3].astype(np.float32)
    table = table.reshape(h, w, 3)
    table[~valid] = 0.0
    return np.transpose(table, (2, 0, 1))  # (3, H, W)


def normalize_table_xyz(
    xyz_chw: np.ndarray,
    bounds: TableXyzBounds,
    *,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Resize to policy resolution and map meters → [0, 1]. Invalid stays 0."""
    import cv2

    xyz = np.asarray(xyz_chw, dtype=np.float32)
    if xyz.shape[1] != out_h or xyz.shape[2] != out_w:
        resized = np.zeros((3, out_h, out_w), dtype=np.float32)
        for c in range(3):
            resized[c] = cv2.resize(
                xyz[c], (out_w, out_h), interpolation=cv2.INTER_NEAREST
            )
        xyz = resized

    out = np.zeros_like(xyz, dtype=np.float32)
    # Valid if any channel nonzero (invalid depth cleared to 0 upstream).
    valid = np.any(np.abs(xyz) > 1e-8, axis=0)

    xy_span = max(bounds.xy_max_m - bounds.xy_min_m, 1e-6)
    z_span = max(bounds.z_max_m - bounds.z_min_m, 1e-6)
    out[0][valid] = np.clip((xyz[0][valid] - bounds.xy_min_m) / xy_span, 0.0, 1.0)
    out[1][valid] = np.clip((xyz[1][valid] - bounds.xy_min_m) / xy_span, 0.0, 1.0)
    out[2][valid] = np.clip((xyz[2][valid] - bounds.z_min_m) / z_span, 0.0, 1.0)
    return out
