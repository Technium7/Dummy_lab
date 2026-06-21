"""Pretrained hole-base keypoint in robot frame (weak observation for insertion policy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from high_level_jobs.subtasks.grab_sensor.config_loader import (
    apply_runtime_calibrations,
    load_grab_sensor_config,
)
from high_level_jobs.subtasks.grab_sensor.detection.yolo_grasp import infer_yolo_grasp_points, pixel_to_cam_m
from high_level_jobs.subtasks.grab_sensor.grasp import camera_to_robot_matrix
from high_level_jobs.subtasks.grab_sensor.models import GrabSensorConfig
from high_level_jobs.subtasks.grab_sensor.table_position import (
    infer_pixel_table_pose,
    resolve_table_calib_path,
    table_point_to_planner_world_m,
)


@dataclass(frozen=True)
class BaseKeypointObservation:
    """Hole / holder base location from pretrained base CNN + table calib."""

    robot_xyz_m: np.ndarray
    image_uv: tuple[int, int] | None
    image_uv_norm: tuple[float, float]
    ellipse_norm: tuple[float, float, float]
    valid: bool
    confidence: float
    debug: dict[str, Any]

    def vector(self) -> np.ndarray:
        """Flat keypoint vector for the policy encoder (8D)."""
        u_norm, v_norm = self.image_uv_norm
        a_norm, b_norm, ang_norm = self.ellipse_norm
        valid_f = 1.0 if self.valid else 0.0
        return np.array(
            [
                float(self.robot_xyz_m[0]),
                float(self.robot_xyz_m[1]),
                float(self.robot_xyz_m[2]),
                u_norm,
                v_norm,
                a_norm,
                b_norm,
                ang_norm * valid_f,
            ],
            dtype=np.float32,
        )


def _ellipse_to_norm(ellipse: Any, *, width: int, height: int) -> tuple[float, float, float]:
    max_dim = max(width, height, 1)
    return (
        float(ellipse.axis_a_px) / max_dim,
        float(ellipse.axis_b_px) / max_dim,
        float(ellipse.angle_deg) / 180.0,
    )


def detect_base_keypoint(
    color_rgb: np.ndarray,
    depth_m: np.ndarray,
    cfg: GrabSensorConfig | None = None,
    *,
    intrinsics: Any | None = None,
) -> BaseKeypointObservation:
    """Run YOLO + base_point CNN and map base center to robot/planner frame."""
    if cfg is None:
        cfg = apply_runtime_calibrations(load_grab_sensor_config())

    img_h, img_w = color_rgb.shape[:2]

    try:
        nn = infer_yolo_grasp_points(color_rgb, depth_m, cfg)
    except Exception as exc:
        return BaseKeypointObservation(
            robot_xyz_m=np.zeros(3, dtype=np.float64),
            image_uv=None,
            image_uv_norm=(0.0, 0.0),
            ellipse_norm=(0.0, 0.0, 0.0),
            valid=False,
            confidence=0.0,
            debug={"reason": "infer_failed", "error": str(exc)},
        )

    if nn.base_uv is None or nn.base_ellipse is None:
        return BaseKeypointObservation(
            robot_xyz_m=np.zeros(3, dtype=np.float64),
            image_uv=nn.base_uv,
            image_uv_norm=(0.0, 0.0),
            ellipse_norm=(0.0, 0.0, 0.0),
            valid=False,
            confidence=float(nn.yolo.confidence),
            debug={"reason": "no_base_uv", "grasp_source": nn.grasp_source},
        )

    if intrinsics is None:
        from high_level_jobs.subtasks.grab_sensor.detection.capture import (
            _DEFAULT_D415_INTRINSICS,
            _dict_to_intrinsics,
        )

        intrinsics = _dict_to_intrinsics(_DEFAULT_D415_INTRINSICS)

    base_uv = nn.base_uv
    base_ellipse = nn.base_ellipse
    debug: dict[str, Any] = {
        "base_uv": list(base_uv),
        "base_ellipse": base_ellipse.as_dict(),
        "yolo_confidence": float(nn.yolo.confidence),
        "grasp_source": nn.grasp_source,
    }

    grasp_net = cfg.grasp_net
    try:
        base_cam_m = pixel_to_cam_m(
            base_uv[0],
            base_uv[1],
            depth_m,
            intrinsics,
            patch_radius_px=grasp_net.depth_patch_radius_px if grasp_net else 3,
            depth_min_m=grasp_net.depth_min_m if grasp_net else 0.25,
            depth_max_m=grasp_net.depth_max_m if grasp_net else 0.85,
        )
        debug["base_cam_m"] = base_cam_m.round(4).tolist()
    except RuntimeError as exc:
        return BaseKeypointObservation(
            robot_xyz_m=np.zeros(3, dtype=np.float64),
            image_uv=base_uv,
            image_uv_norm=(base_uv[0] / max(img_w - 1, 1), base_uv[1] / max(img_h - 1, 1)),
            ellipse_norm=_ellipse_to_norm(base_ellipse, width=img_w, height=img_h),
            valid=False,
            confidence=float(nn.yolo.confidence),
            debug={**debug, "reason": "depth_failed", "error": str(exc)},
        )

    table_calib = resolve_table_calib_path(cfg.table_calib_path)
    if table_calib is not None:
        pose = infer_pixel_table_pose(base_uv, table_calib, z_m=0.0)
        if pose is not None:
            table_xyz = tuple(float(v) for v in pose["table_xyz_m"])
            robot_xyz = table_point_to_planner_world_m(table_xyz, cfg.robot_in_table_frame)
            debug["base_table_xyz_m"] = list(table_xyz)
            debug["base_pose_source"] = "table_homography"
        else:
            t_cam_robot = camera_to_robot_matrix(cfg.camera_to_robot)
            p = np.array([*base_cam_m, 1.0], dtype=np.float64)
            robot_xyz = (t_cam_robot @ p)[:3]
            debug["base_pose_source"] = "camera_to_robot_fallback"
    else:
        t_cam_robot = camera_to_robot_matrix(cfg.camera_to_robot)
        p = np.array([*base_cam_m, 1.0], dtype=np.float64)
        robot_xyz = (t_cam_robot @ p)[:3]
        debug["base_pose_source"] = "camera_to_robot_only"

    debug["base_robot_m"] = robot_xyz.round(4).tolist()
    u_norm = base_uv[0] / max(img_w - 1, 1)
    v_norm = base_uv[1] / max(img_h - 1, 1)
    return BaseKeypointObservation(
        robot_xyz_m=robot_xyz.astype(np.float64),
        image_uv=base_uv,
        image_uv_norm=(float(u_norm), float(v_norm)),
        ellipse_norm=_ellipse_to_norm(base_ellipse, width=img_w, height=img_h),
        valid=True,
        confidence=float(nn.yolo.confidence),
        debug=debug,
    )
