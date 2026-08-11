"""Build flat (perpendicular) grasp targets from cascade detections."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .models import GrabSensorConfig, GraspTargets
from .table_position import (
    infer_pixel_table_pose,
    table_grasp_frame_summary,
    table_point_to_planner_world_m,
)


def _tip_pre_grasp_pos_m(
    grasp_pos_m: np.ndarray,
    approach_offset_mm: float,
    *,
    along_xy: bool,
) -> np.ndarray:
    tip = np.asarray(grasp_pos_m, dtype=np.float64)[:3].copy()
    offset_m = float(approach_offset_mm) * 1e-3
    if along_xy:
        xy = tip[:2].copy()
        n = float(np.linalg.norm(xy))
        if n < 1e-9:
            tip[2] += offset_m
        else:
            tip[:2] = tip[:2] - (xy / n) * offset_m
    else:
        tip[2] += offset_m
    return tip


def _normalize_yaw_deg(yaw_deg: float) -> float:
    return (float(yaw_deg) + 180.0) % 360.0 - 180.0


def flat_grasp_world_rpy_deg(
    cfg: GrabSensorConfig,
    *,
    sensor_axis_deg: float,
) -> tuple[float, float, float]:
    """End-link perpendicular to table: fixed A/B → roll/pitch; yaw from holder axis.

    Maps taught semantic-style ``fixed_grasp_ab_deg`` and holder ``axis_deg`` (prefer
    table-plane angle) into planner world RPY (deg). Default A=-90 → roll=+90;
    ``grasp_rpy_roll_offset_deg`` (~-15.6) corrects URDF TCP (0,-100,-28) mm so
    wrist→tip is world -Z, not link6 -Y.
    """
    a, b = (float(cfg.fixed_grasp_ab_deg[0]), float(cfg.fixed_grasp_ab_deg[1]))
    roll = -a + float(cfg.grasp_rpy_roll_offset_deg)
    pitch = -b
    if cfg.yaw_from_sensor_axis:
        yaw = float(cfg.grasp_rpy_yaw_base_deg) + float(sensor_axis_deg) + float(cfg.grasp_c_offset_deg)
    else:
        yaw = float(cfg.grasp_rpy_yaw_base_deg) + float(cfg.grasp_c_offset_deg)
    return roll, pitch, _normalize_yaw_deg(yaw)


def apply_flat_grasp_orientation(cfg: GrabSensorConfig, targets: GraspTargets) -> None:
    # Prefer table-plane holder angle (camera tilt makes image axis_deg wrong for yaw).
    axis = targets.debug.get("holder_axis_table_deg")
    axis_src = "table"
    if axis is None:
        axis = targets.debug.get("sensor_axis_deg")
        axis_src = "image"
    if axis is None:
        axis = 0.0
        axis_src = "default"
    world_deg = flat_grasp_world_rpy_deg(cfg, sensor_axis_deg=float(axis))
    targets.grasp_rpy_rad = np.radians(np.asarray(world_deg, dtype=np.float64))
    targets.pre_grasp_rpy_rad = targets.grasp_rpy_rad.copy()
    a, b = cfg.fixed_grasp_ab_deg
    c = float(axis) + float(cfg.grasp_c_offset_deg)
    targets.debug["grasp_abc_input_deg"] = [float(a), float(b), float(c)]
    targets.debug["grasp_rpy_world_deg"] = [float(v) for v in world_deg]
    targets.debug["grasp_orientation_mode"] = "flat_perpendicular"
    targets.debug["grasp_yaw_axis_deg"] = float(axis)
    targets.debug["grasp_yaw_axis_source"] = axis_src
    if targets.debug.get("sensor_axis_deg") is None and axis_src == "image":
        targets.debug["sensor_axis_deg"] = float(axis)


def record_grasp_tcp_targets(cfg: GrabSensorConfig, targets: GraspTargets) -> None:
    tip_pre = _tip_pre_grasp_pos_m(
        targets.grasp_pos_m,
        cfg.approach.pre_grasp_offset_mm,
        along_xy=cfg.approach.pre_grasp_along_xy,
    )
    targets.debug["pre_grasp_tip_pos_m"] = tip_pre.round(4).tolist()
    targets.pre_grasp_pos_m = tip_pre
    targets.pre_grasp_rpy_rad = np.asarray(targets.grasp_rpy_rad, dtype=np.float64).copy()
    lift_mm = float(cfg.approach.post_grasp_lift_mm)
    targets.debug["post_grasp_lift_mm"] = lift_mm


def apply_table_grasp_position(cfg: GrabSensorConfig, targets: GraspTargets) -> bool:
    if not cfg.grasp_position_from_table:
        return False
    base_xyz = targets.debug.get("sensor_table_xyz_m")
    if not base_xyz or len(base_xyz) < 2:
        if cfg.table_calib_path is None:
            print("[grab_sensor] warning: no table calib — leaving grasp XYZ unset/zeros")
            return False
        raise RuntimeError(
            "grasp_position_from_table requires sensor_table_xyz_m from vision "
            "(table calib + sensor keypoints)."
        )
    height_m = float(cfg.grasp_motion.grasp_height_above_table_mm) * 1e-3
    grasp_table = (float(base_xyz[0]), float(base_xyz[1]), height_m)
    # Planner world = table frame
    grasp_pos = table_point_to_planner_world_m(grasp_table, cfg.robot_in_table_frame)
    targets.grasp_pos_m = np.asarray(grasp_pos, dtype=np.float64)
    targets.object_pos_m = targets.grasp_pos_m.copy()
    targets.debug.update(
        {
            "grasp_position_source": "table_contact_holder_offset",
            "grasp_table_xyz_m": [round(v, 4) for v in grasp_table],
            "grasp_tip_pos_m": grasp_pos.round(4).tolist(),
            "grasp_tip_pos_robot_m": grasp_pos.round(4).tolist(),
            "grasp_height_above_table_mm": float(cfg.grasp_motion.grasp_height_above_table_mm),
            "grasp_offset_from_holder_mm": float(cfg.grasp_offset_from_holder_mm),
            "grasp_frame_summary": table_grasp_frame_summary(grasp_table, cfg.robot_in_table_frame),
        }
    )
    return True


def finalize_grasp_targets(cfg: GrabSensorConfig, targets: GraspTargets) -> None:
    if cfg.grasp_position_from_table:
        apply_table_grasp_position(cfg, targets)
    apply_flat_grasp_orientation(cfg, targets)
    record_grasp_tcp_targets(cfg, targets)


def _grasp_uv_and_axis(
    cfg: GrabSensorConfig,
    sensor: Any,
) -> tuple[tuple[float, float], float, dict[str, Any]]:
    """Contact + holder axis → image grasp preview (px offset along holder)."""
    from high_level_jobs.grab_sensor.vision.sensor_geometry import (
        SENSOR_NUM_KPTS,
        axis_deg_from_sensor_kpts,
        contact_axis_geometry,
        grasp_uv_from_sensor_kpts,
    )

    kpts = list(sensor.kpts or [])
    bbox = list(sensor.bbox_xyxy) if getattr(sensor, "bbox_xyxy", None) else None
    meta: dict[str, Any] = {"grasp_schema": "contact_holder_table_offset"}
    axis_deg = 0.0

    if len(kpts) >= SENSOR_NUM_KPTS:
        geo = contact_axis_geometry(kpts)
        axis_deg = float(geo.axis_deg)
        meta["contact_uv"] = [float(geo.contact_uv[0]), float(geo.contact_uv[1])]
        meta["holder_dir_uv"] = [float(geo.dir_uv[0]), float(geo.dir_uv[1])]
        meta["holder_along_uv"] = [float(geo.along_uv[0]), float(geo.along_uv[1])]
        center = grasp_uv_from_sensor_kpts(
            kpts,
            offset_px=float(cfg.grasp_offset_from_holder_px),
            bbox_xyxy=bbox,
        )
    elif sensor.center_uv is not None:
        center = (float(sensor.center_uv[0]), float(sensor.center_uv[1]))
        if sensor.axis_deg is not None:
            axis_deg = float(sensor.axis_deg)
    elif kpts:
        center = (float(kpts[0][0]), float(kpts[0][1]))
        if sensor.axis_deg is not None:
            axis_deg = float(sensor.axis_deg)
    else:
        raise RuntimeError("Sensor detection missing center UV / keypoints")

    if sensor.axis_deg is not None and len(kpts) < SENSOR_NUM_KPTS:
        axis_deg = float(sensor.axis_deg)
    elif sensor.axis_deg is not None and len(kpts) >= SENSOR_NUM_KPTS:
        # Prefer head axis if present and kpts are synthetic tips
        pass

    meta["grasp_uv"] = [float(center[0]), float(center[1])]
    return (float(center[0]), float(center[1])), float(axis_deg), meta


def _holder_dir_on_table(
    cfg: GrabSensorConfig,
    kpts: Sequence,
) -> tuple[list[float], list[float], float] | None:
    """Contact table XY, unit holder dir in table XY, and atan2 yaw (deg) in table frame."""
    from high_level_jobs.grab_sensor.vision.sensor_geometry import (
        SENSOR_NUM_KPTS,
        contact_axis_geometry,
    )

    if cfg.table_calib_path is None or len(kpts) < SENSOR_NUM_KPTS:
        return None
    geo = contact_axis_geometry(kpts)
    hx, hy = geo.contact_uv
    nx, ny = geo.along_uv
    contact_pose = infer_pixel_table_pose(
        (int(round(hx)), int(round(hy))),
        cfg.table_calib_path,
        z_m=0.0,
    )
    if contact_pose is None:
        return None
    # Probe along holder in image → table-plane direction (handles camera tilt / V-flip).
    probe_uv = (hx + 25.0 * nx, hy + 25.0 * ny)
    probe_pose = infer_pixel_table_pose(
        (int(round(probe_uv[0])), int(round(probe_uv[1]))),
        cfg.table_calib_path,
        z_m=0.0,
    )
    hx_t, hy_t = float(contact_pose["table_xyz_m"][0]), float(contact_pose["table_xyz_m"][1])
    if probe_pose is not None:
        dx = float(probe_pose["table_xyz_m"][0]) - hx_t
        dy = float(probe_pose["table_xyz_m"][1]) - hy_t
        n = math.hypot(dx, dy)
        if n > 1e-9:
            dx, dy = dx / n, dy / n
        else:
            dx, dy = float(nx), float(ny)
    else:
        dx, dy = float(nx), float(ny)
    axis_table_deg = float(math.degrees(math.atan2(dy, dx)))
    return [hx_t, hy_t, 0.0], [dx, dy], axis_table_deg


def _table_xy_from_contact_offset(
    cfg: GrabSensorConfig,
    kpts: Sequence,
    *,
    bbox_xyxy: Sequence[float] | None = None,
) -> tuple[list[float], dict[str, Any]] | None:
    """Project contact → table, offset along holder by grasp_offset_from_holder_mm."""
    _ = bbox_xyxy
    packed = _holder_dir_on_table(cfg, kpts)
    if packed is None:
        return None
    contact_xyz, dir_xy, axis_table_deg = packed
    hx_t, hy_t = float(contact_xyz[0]), float(contact_xyz[1])
    dx, dy = float(dir_xy[0]), float(dir_xy[1])
    off_m = float(cfg.grasp_offset_from_holder_mm) * 1e-3
    grasp_xyz = [hx_t + off_m * dx, hy_t + off_m * dy, 0.0]
    meta = {
        "holder_dir_table_xy": [dx, dy],
        "holder_axis_table_deg": axis_table_deg,
        "contact_table_xyz_m": [hx_t, hy_t, 0.0],
    }
    return grasp_xyz, meta


def build_grasp_targets_from_detections(
    cfg: GrabSensorConfig,
    objects: Sequence[Any],
    *,
    vision_debug: dict[str, Any] | None = None,
) -> GraspTargets:
    """Pick best sensor detection → grasp targets; stash posts in debug for SCP."""
    sensors = [o for o in objects if getattr(o, "class_name", None) == "sensor"]
    posts = [o for o in objects if getattr(o, "class_name", None) == "post"]
    others = [o for o in objects if getattr(o, "class_name", None) == "other"]

    if not sensors:
        raise RuntimeError("No sensor detections — cannot build grasp targets.")

    sensor = max(sensors, key=lambda o: float(getattr(o, "score", 0.0)))
    center, axis_deg, geo_meta = _grasp_uv_and_axis(cfg, sensor)

    debug: dict[str, Any] = {
        "sensor_uv": [float(center[0]), float(center[1])],
        "sensor_axis_deg": axis_deg,
        "sensor_kpts": sensor.kpts,
        "sensor_bbox": sensor.bbox_xyxy,
        "sensor_score": float(sensor.score),
        "n_posts": len(posts),
        "n_others": len(others),
        "posts": [],
        "others": [],
        **geo_meta,
    }
    if vision_debug:
        debug.update(vision_debug)

    if cfg.table_calib_path is not None:
        packed = _table_xy_from_contact_offset(
            cfg, list(sensor.kpts or []), bbox_xyxy=sensor.bbox_xyxy
        )
        table_xyz = None
        if packed is not None:
            table_xyz, holder_meta = packed
            debug.update(holder_meta)
        if table_xyz is None:
            pose = infer_pixel_table_pose(
                (int(round(center[0])), int(round(center[1]))),
                cfg.table_calib_path,
                z_m=0.0,
            )
            if pose is not None:
                table_xyz = pose["table_xyz_m"]
                debug["sensor_table_pose"] = pose
        if table_xyz is not None:
            debug["sensor_table_xyz_m"] = table_xyz
            if "sensor_table_pose" not in debug:
                debug["sensor_table_pose"] = {
                    "table_xyz_m": table_xyz,
                    "uv": [float(center[0]), float(center[1])],
                }

        for post in posts:
            cuv = post.center_uv or (post.kpts[0] if post.kpts else None)
            if cuv is None:
                continue
            ppose = infer_pixel_table_pose(
                (int(round(cuv[0])), int(round(cuv[1]))),
                cfg.table_calib_path,
                z_m=0.0,
            )
            entry = {
                "uv": [float(cuv[0]), float(cuv[1])],
                "score": float(post.score),
                "table_xyz_m": ppose["table_xyz_m"] if ppose else None,
            }
            debug["posts"].append(entry)

        for other in others:
            cuv = other.center_uv or (other.kpts[0] if other.kpts else None)
            if cuv is None:
                continue
            debug["others"].append({"uv": [float(cuv[0]), float(cuv[1])], "score": float(other.score)})

    # Placeholder position until finalize applies table height
    pos = np.zeros(3, dtype=np.float64)
    if "sensor_table_xyz_m" in debug:
        height_m = float(cfg.grasp_motion.grasp_height_above_table_mm) * 1e-3
        gt = (
            float(debug["sensor_table_xyz_m"][0]),
            float(debug["sensor_table_xyz_m"][1]),
            height_m,
        )
        pos = table_point_to_planner_world_m(gt, cfg.robot_in_table_frame)

    targets = GraspTargets(
        pre_grasp_pos_m=pos.copy(),
        pre_grasp_rpy_rad=np.zeros(3),
        grasp_pos_m=pos.copy(),
        grasp_rpy_rad=np.zeros(3),
        object_pos_m=pos.copy(),
        object_rpy_rad=np.zeros(3),
        debug=debug,
    )
    finalize_grasp_targets(cfg, targets)
    return targets
