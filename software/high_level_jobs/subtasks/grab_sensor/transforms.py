from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

try:
    import pybullet as p
except ImportError:  # pragma: no cover - planning path requires pybullet
    p = None  # type: ignore


def rpy_rad_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    if p is not None:
        q = p.getQuaternionFromEuler(list(rpy))
        mat = np.array(p.getMatrixFromQuaternion(q), dtype=np.float64).reshape(3, 3)
        return mat
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy_rad(rot: np.ndarray) -> np.ndarray:
    sy = math.sqrt(float(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    if sy > 1e-8:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def pose_to_matrix(translation_m: Sequence[float], rpy_rad: Sequence[float]) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rpy_rad_to_matrix(rpy_rad)
    t[:3, 3] = np.asarray(translation_m, dtype=np.float64)[:3]
    return t


def decompose_matrix(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = transform[:3, 3].copy()
    rpy = matrix_to_rpy_rad(transform[:3, :3])
    return t, rpy


def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = [x, y, z]
    return m


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    t = transform[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ t
    return inv


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    for m in transforms:
        out = out @ m
    return out


def offset_in_local_frame(
    base_transform: np.ndarray,
    local_xyz_m: Sequence[float],
) -> np.ndarray:
    delta = translation_matrix(float(local_xyz_m[0]), float(local_xyz_m[1]), float(local_xyz_m[2]))
    return base_transform @ delta
