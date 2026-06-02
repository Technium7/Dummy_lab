from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .transforms import matrix_to_rpy_rad
from .models import TableObjectDetection


def backproject_mask_to_points_cam(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Any,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("Empty segmentation mask.")

    z = depth_m[ys, xs]
    keep = z > 0.0
    xs = xs[keep]
    ys = ys[keep]
    z = z[keep]
    if len(z) < 20:
        raise RuntimeError(f"Too few valid depth points in mask ({len(z)}).")

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.ppx)
    cy = float(intrinsics.ppy)

    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def estimate_object_pose_pca(points_cam_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate table-object pose in the camera frame from its point cloud.

    Returns (centroid_m, object_rpy_cam_rad, yaw_rad_in_camera_xy).
    """
    centroid = points_cam_m.mean(axis=0)
    centered = points_cam_m - centroid
    cov = (centered.T @ centered) / max(len(points_cam_m), 1)
    evals, evecs = np.linalg.eigh(cov)

    order = np.argsort(evals)
    normal = evecs[:, order[0]]
    major = evecs[:, order[2]]

    if normal[2] > 0:
        normal = -normal

    major_xy = major.copy()
    major_xy[2] = 0.0
    if np.linalg.norm(major_xy) < 1e-6:
        major_xy = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    major_xy /= np.linalg.norm(major_xy)

    y_axis = np.cross(normal, major_xy)
    y_axis /= max(np.linalg.norm(y_axis), 1e-9)
    x_axis = np.cross(y_axis, normal)
    x_axis /= max(np.linalg.norm(x_axis), 1e-9)

    rot = np.column_stack([x_axis, y_axis, normal])
    rpy = matrix_to_rpy_rad(rot)
    yaw = float(np.arctan2(x_axis[1], x_axis[0]))
    return centroid, rpy, yaw


def detect_table_object(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Any,
    bbox_xy: Tuple[int, int, int, int],
) -> TableObjectDetection:
    points = backproject_mask_to_points_cam(mask, depth_m, intrinsics)
    centroid, rpy, yaw = estimate_object_pose_pca(points)
    return TableObjectDetection(
        centroid_cam_m=centroid,
        yaw_rad=yaw,
        object_rpy_cam_rad=rpy,
        mask=mask,
        n_points=len(points),
        bbox_xy=bbox_xy,
    )


detect_sensor_pose = detect_table_object
