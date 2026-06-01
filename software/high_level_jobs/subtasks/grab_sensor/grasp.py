from __future__ import annotations

import math
from typing import Any

import numpy as np

from .transforms import compose_transforms, decompose_matrix, offset_in_local_frame, pose_to_matrix
from .types import CameraToRobot, GrabSensorConfig, GraspTargets, TableObjectDetection


def camera_to_robot_matrix(mount: CameraToRobot) -> np.ndarray:
    rpy_rad = np.radians(np.asarray(mount.rpy_deg, dtype=np.float64))
    return pose_to_matrix(mount.translation_m, rpy_rad)


def object_pose_robot(detection: TableObjectDetection, t_robot_cam: np.ndarray) -> np.ndarray:
    """Table object pose in robot frame from camera detection."""
    t_cam_object = pose_to_matrix(detection.centroid_cam_m, detection.object_rpy_cam_rad)
    return t_robot_cam @ t_cam_object


def grasp_offset_on_object(cfg: GrabSensorConfig) -> np.ndarray:
    g = cfg.grasp_on_object
    rpy_rad = np.radians(np.asarray(g.rpy_deg, dtype=np.float64))
    t_object_grasp = pose_to_matrix(g.position_m(), rpy_rad)
    if abs(g.approach_offset_mm) > 1e-9:
        t_object_grasp = offset_in_local_frame(
            t_object_grasp,
            (0.0, 0.0, g.approach_offset_mm * 1e-3),
        )
    return t_object_grasp


def _grasp_pose_follow_object(
    cfg: GrabSensorConfig,
    detection: TableObjectDetection,
    t_robot_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full chain: robot <- camera <- object (vision) <- grasp offset (taught)."""
    t_robot_object = object_pose_robot(detection, t_robot_cam)
    t_object_grasp = grasp_offset_on_object(cfg)
    t_robot_grasp = compose_transforms(t_robot_object, t_object_grasp)

    grasp_pos, grasp_rpy = decompose_matrix(t_robot_grasp)
    object_pos, object_rpy = decompose_matrix(t_robot_object)
    return grasp_pos, grasp_rpy, object_pos, object_rpy


def _grasp_pose_fixed_robot(
    cfg: GrabSensorConfig,
    detection: TableObjectDetection,
    t_robot_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vision supplies grasp LOCATION (x,y,z); gripper orientation comes from config.

    Position = robot-frame object centroid + optional mm offset from grasp_on_object
    (applied in robot frame after transforming the offset vector).
    """
    t_robot_object = object_pose_robot(detection, t_robot_cam)
    object_pos, object_rpy = decompose_matrix(t_robot_object)

    offset_m = cfg.grasp_on_object.position_m()
    grasp_pos = object_pos + t_robot_object[:3, :3] @ offset_m

    grasp_rpy = np.radians(np.asarray(cfg.fixed_grasp_rpy_robot_deg, dtype=np.float64))
    return grasp_pos, grasp_rpy, object_pos, object_rpy


def compute_grasp_targets(cfg: GrabSensorConfig, detection: TableObjectDetection) -> GraspTargets:
    t_robot_cam = camera_to_robot_matrix(cfg.camera_to_robot)

    if cfg.grasp_orientation_mode == "fixed_robot":
        grasp_pos, grasp_rpy, object_pos, object_rpy = _grasp_pose_fixed_robot(
            cfg, detection, t_robot_cam
        )
    else:
        grasp_pos, grasp_rpy, object_pos, object_rpy = _grasp_pose_follow_object(
            cfg, detection, t_robot_cam
        )

    t_robot_grasp = pose_to_matrix(grasp_pos, grasp_rpy)
    pre_offset_m = cfg.approach.pre_grasp_offset_mm * 1e-3
    t_robot_pre = offset_in_local_frame(t_robot_grasp, (0.0, 0.0, -pre_offset_m))
    pre_pos, pre_rpy = decompose_matrix(t_robot_pre)

    debug: dict[str, Any] = {
        "object_yaw_cam_deg": float(math.degrees(detection.yaw_rad)),
        "grasp_orientation_mode": cfg.grasp_orientation_mode,
        "n_points": detection.n_points,
        "bbox_xy": detection.bbox_xy,
        "object_pos_robot_m": object_pos.round(4).tolist(),
        "grasp_pos_robot_m": grasp_pos.round(4).tolist(),
        "grasp_rpy_robot_deg": [float(math.degrees(v)) for v in grasp_rpy],
        "pre_grasp_rpy_deg": [float(math.degrees(v)) for v in pre_rpy],
    }

    return GraspTargets(
        pre_grasp_pos_m=pre_pos,
        pre_grasp_rpy_rad=pre_rpy,
        grasp_pos_m=grasp_pos,
        grasp_rpy_rad=grasp_rpy,
        object_pos_m=object_pos,
        object_rpy_rad=object_rpy,
        debug=debug,
    )
