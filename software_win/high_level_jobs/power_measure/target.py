"""Click UV → measured table point → TCP insert pose (radial-left + height)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from high_level_jobs.grab_sensor.models import GraspTargets
from high_level_jobs.grab_sensor.table_position import (
    build_table_plane_homography,
    infer_pixel_table_pose,
    load_table_calib,
    table_grasp_frame_summary,
    table_point_to_planner_world_m,
)

from .models import PowerMeasureConfig

_INCH_TO_M = 0.0254


def _normalize_yaw_deg(yaw_deg: float) -> float:
    return (float(yaw_deg) + 180.0) % 360.0 - 180.0


def table_xy_to_pixel(
    table_xy_m: tuple[float, float],
    table_calib_path: Path,
) -> tuple[float, float] | None:
    """Project table-surface XY (m) to image UV via calib hole homography."""
    data = load_table_calib(table_calib_path)
    homography = build_table_plane_homography(data)
    if homography is None:
        return None
    x, y = float(table_xy_m[0]), float(table_xy_m[1])
    projected = homography @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(projected[2]) < 1e-9:
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


@dataclass
class InsertTarget:
    """Measured click + TCP insert geometry in table / robot frames."""

    click_uv: tuple[int, int]
    measured_table_xyz_m: tuple[float, float, float]
    tcp_table_xyz_m: tuple[float, float, float]
    tcp_robot_xyz_m: np.ndarray
    tcp_rpy_rad: np.ndarray
    pre_insert_robot_xyz_m: np.ndarray
    radial_unit_xy: tuple[float, float]
    left_unit_xy: tuple[float, float]
    radial_deg: float
    used_radial_fallback: bool = False
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def measured_uv(self) -> tuple[int, int]:
        return self.click_uv


def robot_xy_in_table_m(cfg: PowerMeasureConfig) -> tuple[float, float]:
    tx, ty, _ = cfg.robot_in_table_frame.translation_m()
    return float(tx), float(ty)


def compute_radial_left(
    measured_xy_m: tuple[float, float],
    robot_xy_m: tuple[float, float],
    *,
    min_radial_m: float = 1e-3,
) -> tuple[tuple[float, float], tuple[float, float], float, bool]:
    """Return (radial_unit, left_unit, radial_deg, used_fallback).

    Left = +90° CCW about +Z from robot→measured (left when facing along the radial).
    """
    dx = float(measured_xy_m[0]) - float(robot_xy_m[0])
    dy = float(measured_xy_m[1]) - float(robot_xy_m[1])
    norm = math.hypot(dx, dy)
    if norm < float(min_radial_m):
        # Fallback: treat "forward" as +X, left as -Y? Plan says table -X as left.
        # left = (−1, 0); invent a radial of (0, −1) so left = (−û_y, û_x) consistency:
        # û=(0,-1) → left=(1,0). Plan: fall back to table −X as left → (−1, 0).
        radial = (0.0, -1.0)
        left = (-1.0, 0.0)
        return radial, left, math.degrees(math.atan2(radial[1], radial[0])), True
    radial = (dx / norm, dy / norm)
    left = (-radial[1], radial[0])  # CCW 90°
    radial_deg = math.degrees(math.atan2(radial[1], radial[0]))
    return radial, left, radial_deg, False


def insert_world_rpy_deg(cfg: PowerMeasureConfig, *, radial_deg: float) -> tuple[float, float, float]:
    """World RPY from A/B + radial yaw (CW offset).

    Default A=-90, B=+90 → pitch=-90 so link local X (longer red axis in
    ``--ik-gui``) aligns with table +Z; yaw still spins the tip in the table XY.
    """
    a, b = (float(cfg.fixed_grasp_ab_deg[0]), float(cfg.fixed_grasp_ab_deg[1]))
    roll = -a + float(cfg.grasp_rpy_roll_offset_deg)
    pitch = -b
    yaw = (
        float(cfg.grasp_rpy_yaw_base_deg)
        + float(radial_deg)
        - float(cfg.yaw_cw_from_radial_deg)
        + float(cfg.grasp_c_offset_deg)
    )
    return roll, pitch, _normalize_yaw_deg(yaw)


def build_insert_target_from_uv(
    cfg: PowerMeasureConfig,
    uv: tuple[int, int],
) -> InsertTarget:
    if cfg.table_calib_path is None:
        raise RuntimeError("power_measure requires table_calib_path (no valid table calib found)")

    measured_z_m = float(cfg.measured_height_in) * _INCH_TO_M
    pose = infer_pixel_table_pose(
        (int(uv[0]), int(uv[1])),
        cfg.table_calib_path,
        z_m=measured_z_m,
    )
    if pose is None:
        raise RuntimeError(f"Failed to map click UV {uv} to table XY")

    mx, my, mz = (float(v) for v in pose["table_xyz_m"])
    robot_xy = robot_xy_in_table_m(cfg)
    radial, left, radial_deg, fallback = compute_radial_left(
        (mx, my),
        robot_xy,
        min_radial_m=cfg.min_radial_m,
    )
    if fallback:
        print(
            "[power_measure] warning: click too close to robot XY; "
            "using table -X as left offset direction"
        )

    left_m = float(cfg.tcp_left_of_radial_in) * _INCH_TO_M
    above_m = float(cfg.tcp_above_measured_in) * _INCH_TO_M
    tcp_x = mx + left_m * left[0]
    tcp_y = my + left_m * left[1]
    tcp_z = mz + above_m
    tcp_table = (tcp_x, tcp_y, tcp_z)
    tcp_robot = table_point_to_planner_world_m(tcp_table, cfg.robot_in_table_frame)

    world_deg = insert_world_rpy_deg(cfg, radial_deg=radial_deg)
    rpy_rad = np.radians(np.asarray(world_deg, dtype=np.float64))

    pre_offset_m = float(cfg.approach.pre_grasp_offset_mm) * 1e-3
    pre_robot = tcp_robot.copy()
    if cfg.approach.pre_grasp_along_xy:
        xy = pre_robot[:2].copy()
        n = float(np.linalg.norm(xy))
        if n < 1e-9:
            pre_robot[2] += pre_offset_m
        else:
            pre_robot[:2] = pre_robot[:2] - (xy / n) * pre_offset_m
    else:
        pre_robot[2] += pre_offset_m

    # Seed fields so grab_sensor.finalize_grasp_targets reproduces this TCP/yaw:
    #   grasp XY from sensor_table_xyz_m; Z from grasp_height_above_table_mm;
    #   yaw = yaw_base + sensor_axis_deg + c_offset with sensor_axis = radial - yaw_cw.
    sensor_axis_deg = float(radial_deg) - float(cfg.yaw_cw_from_radial_deg)
    debug: dict[str, Any] = {
        "click_uv": [int(uv[0]), int(uv[1])],
        "measured_table_xyz_m": [mx, my, mz],
        "measured_table_xyz_in": [mx / _INCH_TO_M, my / _INCH_TO_M, mz / _INCH_TO_M],
        "tcp_table_xyz_m": [tcp_x, tcp_y, tcp_z],
        "tcp_table_xyz_in": [tcp_x / _INCH_TO_M, tcp_y / _INCH_TO_M, tcp_z / _INCH_TO_M],
        "tcp_robot_xyz_m": tcp_robot.round(4).tolist(),
        "pre_insert_robot_xyz_m": pre_robot.round(4).tolist(),
        "radial_unit_xy": list(radial),
        "left_unit_xy": list(left),
        "radial_deg": float(radial_deg),
        "used_radial_fallback": bool(fallback),
        "insert_rpy_world_deg": [float(v) for v in world_deg],
        "grasp_rpy_world_deg": [float(v) for v in world_deg],
        "grasp_orientation_mode": "power_measure_link_x_along_table_z",
        "grasp_frame_summary": table_grasp_frame_summary(tcp_table, cfg.robot_in_table_frame),
        "robot_xy_table_m": list(robot_xy),
        "table_pose": pose,
        "sensor_table_xyz_m": [tcp_x, tcp_y, tcp_z],
        "sensor_axis_deg": sensor_axis_deg,
        "posts": [],
    }
    tcp_uv = table_xy_to_pixel((tcp_x, tcp_y), cfg.table_calib_path)
    if tcp_uv is not None:
        debug["tcp_uv"] = [float(tcp_uv[0]), float(tcp_uv[1])]
    robot_uv = table_xy_to_pixel(robot_xy, cfg.table_calib_path)
    if robot_uv is not None:
        debug["robot_uv"] = [float(robot_uv[0]), float(robot_uv[1])]

    return InsertTarget(
        click_uv=(int(uv[0]), int(uv[1])),
        measured_table_xyz_m=(mx, my, mz),
        tcp_table_xyz_m=tcp_table,
        tcp_robot_xyz_m=tcp_robot,
        tcp_rpy_rad=rpy_rad,
        pre_insert_robot_xyz_m=pre_robot,
        radial_unit_xy=radial,
        left_unit_xy=left,
        radial_deg=float(radial_deg),
        used_radial_fallback=bool(fallback),
        debug=debug,
    )


def insert_target_to_grasp_targets(insert: InsertTarget) -> GraspTargets:
    """Adapt InsertTarget into grab_sensor GraspTargets for plan/execute."""
    grasp_pos = np.asarray(insert.tcp_robot_xyz_m, dtype=np.float64)
    rpy = np.asarray(insert.tcp_rpy_rad, dtype=np.float64)
    pre = np.asarray(insert.pre_insert_robot_xyz_m, dtype=np.float64)
    return GraspTargets(
        pre_grasp_pos_m=pre,
        pre_grasp_rpy_rad=rpy.copy(),
        grasp_pos_m=grasp_pos,
        grasp_rpy_rad=rpy.copy(),
        object_pos_m=grasp_pos.copy(),
        object_rpy_rad=rpy.copy(),
        debug=dict(insert.debug),
    )


def draw_insert_overlay(
    bgr: np.ndarray,
    insert: InsertTarget,
    *,
    panel_h: int = 0,
) -> np.ndarray:
    """Draw click, robot→click radial, left offset, and TCP on a BGR image."""
    import cv2

    out = bgr
    y0 = int(panel_h)
    cu, cv = insert.click_uv
    cv2.drawMarker(out, (cu, cv + y0), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)

    robot_uv = insert.debug.get("robot_uv")
    tcp_uv = insert.debug.get("tcp_uv")
    if robot_uv is not None:
        ru, rv = int(round(float(robot_uv[0]))), int(round(float(robot_uv[1])))
        cv2.circle(out, (ru, rv + y0), 6, (255, 128, 0), 2)
        cv2.line(out, (ru, rv + y0), (cu, cv + y0), (255, 180, 80), 1, cv2.LINE_AA)
        cv2.putText(
            out,
            "robot",
            (ru + 6, rv + y0 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 128, 0),
            1,
        )

    if tcp_uv is not None:
        tu, tv = int(round(float(tcp_uv[0]))), int(round(float(tcp_uv[1])))
        cv2.drawMarker(out, (tu, tv + y0), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.arrowedLine(
            out,
            (cu, cv + y0),
            (tu, tv + y0),
            (0, 200, 0),
            2,
            tipLength=0.2,
        )
        cv2.putText(
            out,
            "TCP",
            (tu + 6, tv + y0 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    cv2.putText(
        out,
        "click",
        (cu + 8, cv + y0 + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    return out
