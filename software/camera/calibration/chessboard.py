from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .card_config import CalibrationCardConfig


@dataclass
class BoardObservation:
    found: bool
    corners: np.ndarray | None
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    reprojection_error_px: float
    annotated_bgr: np.ndarray | None


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for calibration. "
            "Install with: pip install -r requirements-vision.txt"
        ) from exc
    return cv2


def object_points_3d(cfg: CalibrationCardConfig) -> np.ndarray:
    cols, rows = cfg.pattern_size
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid.astype(np.float64) * cfg.square_size_m
    return objp


def camera_matrix_from_intrinsics(intr: Any) -> np.ndarray:
    return np.array(
        [
            [float(intr.fx), 0.0, float(intr.ppx)],
            [0.0, float(intr.fy), float(intr.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_from_intrinsics(intr: Any) -> np.ndarray:
    coeffs = getattr(intr, "coeffs", None)
    if coeffs is None:
        return np.zeros(5, dtype=np.float64)
    return np.asarray(list(coeffs), dtype=np.float64)[:5]


def detect_board_pose(
    color_bgr: np.ndarray,
    cfg: CalibrationCardConfig,
    intrinsics: Any,
) -> BoardObservation:
    cv2 = _require_cv2()
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    pattern = cfg.pattern_size
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    annotated = color_bgr.copy()
    if not found or corners is None:
        cv2.putText(
            annotated,
            "Checkerboard not found",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return BoardObservation(
            found=False,
            corners=None,
            rvec=None,
            tvec=None,
            reprojection_error_px=float("inf"),
            annotated_bgr=annotated,
        )

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    objp = object_points_3d(cfg)
    cam_mat = camera_matrix_from_intrinsics(intrinsics)
    dist = distortion_from_intrinsics(intrinsics)
    ok, rvec, tvec = cv2.solvePnP(objp, corners_refined, cam_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        cv2.drawChessboardCorners(annotated, pattern, corners_refined, found)
        return BoardObservation(
            found=False,
            corners=corners_refined,
            rvec=None,
            tvec=None,
            reprojection_error_px=float("inf"),
            annotated_bgr=annotated,
        )

    projected, _ = cv2.projectPoints(objp, rvec, tvec, cam_mat, dist)
    err = float(np.linalg.norm(projected.reshape(-1, 2) - corners_refined.reshape(-1, 2), axis=1).mean())
    cv2.drawChessboardCorners(annotated, pattern, corners_refined, found)
    cv2.putText(
        annotated,
        f"reproj {err:.2f}px",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return BoardObservation(
        found=True,
        corners=corners_refined,
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
        reprojection_error_px=err,
        annotated_bgr=annotated,
    )


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    rot, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = tvec.reshape(3)
    return mat
