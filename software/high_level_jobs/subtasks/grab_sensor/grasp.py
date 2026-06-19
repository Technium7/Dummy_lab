from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .transforms import compose_transforms, decompose_matrix, offset_in_local_frame, pose_to_matrix
from .models import CameraToRobot, GrabSensorConfig, GraspTargets, TableObjectDetection


def _tip_pre_grasp_pos_m(
    tip_pos_m: np.ndarray,
    approach_offset_mm: float,
    *,
    along_xy: bool = True,
    robot_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Tip pose before final grasp.

    Default: retreat toward the robot along the base→grasp XY direction (Z unchanged).
    ``along_xy=False``: legacy offset along world +Z.
    """
    offset_m = float(approach_offset_mm) * 1e-3
    if offset_m <= 0.0:
        return np.asarray(tip_pos_m, dtype=np.float64)[:3].copy()

    tip = np.asarray(tip_pos_m, dtype=np.float64)[:3].copy()
    if not along_xy:
        tip[2] += offset_m
        return tip

    origin = np.asarray(robot_origin_m, dtype=np.float64)[:3]
    xy = tip[:2] - origin[:2]
    dist_xy = float(np.linalg.norm(xy))
    if dist_xy < 1e-9:
        return tip
    unit_xy = xy / dist_xy
    tip[0] -= unit_xy[0] * offset_m
    tip[1] -= unit_xy[1] * offset_m
    return tip


def _pre_grasp_pos_m(
    grasp_pos_m: np.ndarray,
    approach_offset_mm: float,
    *,
    along_xy: bool = True,
) -> np.ndarray:
    """Pre-grasp tip pose (vision path, before end-link offset in planning)."""
    return _tip_pre_grasp_pos_m(grasp_pos_m, approach_offset_mm, along_xy=along_xy)


def record_grasp_tcp_targets(cfg: GrabSensorConfig, targets: GraspTargets) -> None:
    """Annotate targets: positions are TCP (``grasp_tcp_link`` in URDF)."""
    tip = np.asarray(targets.grasp_pos_m, dtype=np.float64)
    rpy = np.asarray(targets.grasp_rpy_rad, dtype=np.float64)

    targets.debug["grasp_tip_pos_m"] = tip.round(4).tolist()
    targets.debug["tcp_link"] = cfg.tcp_link

    tip_pre = _tip_pre_grasp_pos_m(
        tip,
        cfg.approach.pre_grasp_offset_mm,
        along_xy=cfg.approach.pre_grasp_along_xy,
    )
    targets.debug["pre_grasp_tip_pos_m"] = tip_pre.round(4).tolist()
    targets.pre_grasp_pos_m = tip_pre
    targets.pre_grasp_rpy_rad = rpy.copy()

    lift_mm = float(cfg.approach.post_grasp_lift_mm)
    if lift_mm > 0.0:
        lift_pos = tip.copy()
        lift_pos[2] += lift_mm * 1e-3
        targets.debug["post_lift_tip_pos_m"] = lift_pos.round(4).tolist()
        targets.debug["post_grasp_lift_mm"] = lift_mm


def _normalize_yaw_deg(yaw_deg: float) -> float:
    return (float(yaw_deg) + 180.0) % 360.0 - 180.0


def _grasp_xy_heading_deg(x_m: float, y_m: float) -> float:
    """Heading term (deg) for world yaw: ``atan2(x, y)`` in robot base XY."""
    return float(math.degrees(math.atan2(x_m, y_m)))


def grasp_abc_input_to_world_rpy_deg(
    abc_input_deg: Sequence[float],
    grasp_pos_m: np.ndarray | Sequence[float] | None,
    *,
    grasp_c_from_xy: bool = True,
    roll_offset_deg: float = 15.0,
    yaw_base_deg: float = 180.0,
) -> tuple[float, float, float]:
    """Map taught input ``[a, b, c]`` (deg) to planner world RPY (deg).

    ``world = [-a + roll_offset, -b, yaw_base + c - atan2(x, y)]`` when ``grasp_c_from_xy``.
    """
    a, b, c = (float(abc_input_deg[i]) for i in range(3))
    roll = -a + float(roll_offset_deg)
    pitch = -b
    yaw = float(yaw_base_deg) + c
    if grasp_c_from_xy and grasp_pos_m is not None:
        pos = np.asarray(grasp_pos_m, dtype=np.float64)[:3]
        yaw -= _grasp_xy_heading_deg(float(pos[0]), float(pos[1]))
    return roll, pitch, _normalize_yaw_deg(yaw)


def grasp_abc_to_world_rpy_deg(
    cfg: GrabSensorConfig,
    abc_input_deg: Sequence[float],
    grasp_pos_m: np.ndarray | Sequence[float] | None,
) -> tuple[float, float, float]:
    """``grasp_abc_input_to_world_rpy_deg`` using ``GrabSensorConfig`` orientation fields."""
    return grasp_abc_input_to_world_rpy_deg(
        abc_input_deg,
        grasp_pos_m,
        grasp_c_from_xy=cfg.grasp_c_from_xy,
        roll_offset_deg=cfg.grasp_rpy_roll_offset_deg,
        yaw_base_deg=cfg.grasp_rpy_yaw_base_deg,
    )


def apply_fixed_grasp_orientation(cfg: GrabSensorConfig, targets: GraspTargets) -> None:
    """Set grasp/pre-grasp world RPY for fixed_robot mode (called before planning)."""
    if cfg.grasp_orientation_mode != "fixed_robot":
        return

    abc = [float(v) for v in cfg.fixed_grasp_abc_input_deg]
    pos = np.asarray(targets.grasp_pos_m, dtype=np.float64)[:3]
    world_deg = grasp_abc_to_world_rpy_deg(cfg, abc, pos)
    if cfg.grasp_c_from_xy:
        targets.debug["grasp_heading_deg"] = _grasp_xy_heading_deg(float(pos[0]), float(pos[1]))
    targets.grasp_rpy_rad = np.radians(np.asarray(world_deg, dtype=np.float64))
    targets.debug["grasp_abc_input_deg"] = abc
    targets.debug["grasp_rpy_world_deg"] = [float(v) for v in world_deg]


def prepare_fixed_robot_grasp_targets(cfg: GrabSensorConfig, targets: GraspTargets) -> None:
    """Apply fixed orientation and TCP target annotations (shared by viz and ik-gui)."""
    apply_fixed_grasp_orientation(cfg, targets)
    record_grasp_tcp_targets(cfg, targets)


def finalize_grasp_targets(cfg: GrabSensorConfig, targets: GraspTargets) -> None:
    """
    Stable production path: table-base XYZ override (if enabled), fixed RPY, TCP pre-grasp/lift.

    Call after vision ``compute_grasp_targets`` and ``targets.debug`` are populated.
    Safe to call again before planning (idempotent).
    """
    if cfg.grasp_position_from_table_base:
        apply_table_base_grasp_position(cfg, targets, targets.debug)
    prepare_fixed_robot_grasp_targets(cfg, targets)


def build_grasp_targets(
    cfg: GrabSensorConfig,
    detection: TableObjectDetection,
    *,
    vision_debug: dict[str, Any] | None = None,
) -> GraspTargets:
    """Vision detection → finalized robot grasp targets (single entry for pipeline and overlays)."""
    targets = compute_grasp_targets(cfg, detection)
    if vision_debug:
        targets.debug.update(vision_debug)
    finalize_grasp_targets(cfg, targets)
    return targets


def load_grasp_targets_from_vision(
    cfg: GrabSensorConfig,
    *,
    offline: bool = True,
    grasp_pos_m: np.ndarray | Sequence[float] | None = None,
) -> GraspTargets:
    """Vision grasp XYZ + fixed_robot orientation (same inputs for viz and ik-gui)."""
    if grasp_pos_m is not None:
        pos = np.asarray(grasp_pos_m, dtype=np.float64)[:3]
        targets = GraspTargets(
            pre_grasp_pos_m=pos.copy(),
            pre_grasp_rpy_rad=np.zeros(3),
            grasp_pos_m=pos,
            grasp_rpy_rad=np.zeros(3),
            object_pos_m=pos.copy(),
            object_rpy_rad=np.zeros(3),
            debug={},
        )
    else:
        from .pipeline import run_detection

        _, targets, _, _ = run_detection(
            cfg,
            {
                "offline": offline,
                "allow_offline_fallback": not offline,
            },
        )
        return targets

    finalize_grasp_targets(cfg, targets)
    return targets


def apply_table_base_grasp_position(
    cfg: GrabSensorConfig,
    targets: GraspTargets,
    vision_debug: dict[str, Any],
) -> bool:
    """Set grasp XYZ from detected base table X/Y and fixed height above the surface."""
    if not cfg.grasp_position_from_table_base:
        return False

    base_xyz = vision_debug.get("base_table_xyz_m")
    if not base_xyz or len(base_xyz) < 2:
        raise RuntimeError(
            "grasp_position_from_table_base requires base_table_xyz_m from vision "
            "(label base ellipses, train base_point.pt, and ensure table calib is loaded)."
        )

    from .table_position import table_grasp_frame_summary, table_point_to_robot_gui_m

    height_m = float(cfg.grasp_motion.grasp_height_above_table_mm) * 1e-3
    grasp_table = (float(base_xyz[0]), float(base_xyz[1]), height_m)
    grasp_pos = table_point_to_robot_gui_m(grasp_table, cfg.robot_in_table_frame)

    targets.grasp_pos_m = np.asarray(grasp_pos, dtype=np.float64)
    targets.debug.update(
        {
            "grasp_position_source": "table_base_xy_height",
            "grasp_table_xyz_m": [round(v, 4) for v in grasp_table],
            "grasp_tip_pos_robot_m": grasp_pos.round(4).tolist(),
            "grasp_height_above_table_mm": float(cfg.grasp_motion.grasp_height_above_table_mm),
            "robot_in_table_in": list(cfg.robot_in_table_frame.translation_in),
            "grasp_frame_summary": table_grasp_frame_summary(grasp_table, cfg.robot_in_table_frame),
        }
    )
    return True


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
    """Vision supplies grasp XYZ; orientation resolved later from config."""
    t_robot_object = object_pose_robot(detection, t_robot_cam)
    object_pos, object_rpy = decompose_matrix(t_robot_object)

    offset_m = cfg.grasp_on_object.position_m()
    grasp_pos = object_pos + t_robot_object[:3, :3] @ offset_m
    grasp_rpy = np.zeros(3, dtype=np.float64)
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

    grasp_pos_m = np.asarray(grasp_pos, dtype=np.float64)
    pre_grasp_pos_m = _pre_grasp_pos_m(
        grasp_pos_m,
        cfg.approach.pre_grasp_offset_mm,
        along_xy=cfg.approach.pre_grasp_along_xy,
    )

    debug: dict[str, Any] = {
        "object_yaw_cam_deg": float(math.degrees(detection.yaw_rad)),
        "grasp_orientation_mode": cfg.grasp_orientation_mode,
        "n_points": detection.n_points,
        "bbox_xy": detection.bbox_xy,
        "object_pos_robot_m": object_pos.round(4).tolist(),
        "grasp_pos_robot_m": grasp_pos_m.round(4).tolist(),
        "grasp_rpy_robot_deg": [float(math.degrees(v)) for v in grasp_rpy],
        "pre_grasp_rpy_deg": [float(math.degrees(v)) for v in grasp_rpy],
    }

    return GraspTargets(
        pre_grasp_pos_m=pre_grasp_pos_m,
        pre_grasp_rpy_rad=grasp_rpy,
        grasp_pos_m=grasp_pos_m,
        grasp_rpy_rad=grasp_rpy,
        object_pos_m=object_pos,
        object_rpy_rad=object_rpy,
        debug=debug,
    )
