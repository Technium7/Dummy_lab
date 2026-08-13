"""Fixed table-frame rough TCP → GraspTargets for epoxy pre_apply planning."""

from __future__ import annotations

from typing import Any

import numpy as np

from high_level_jobs.grab_sensor.models import GraspTargets
from high_level_jobs.grab_sensor.table_position import (
    table_grasp_frame_summary,
    table_point_to_planner_world_m,
)

from .models import EpoxyApproachConfig


def build_rough_approach_targets(cfg: EpoxyApproachConfig) -> GraspTargets:
    """Build GraspTargets at the safe rough TCP (table inches → planner m)."""
    tcp_in = cfg.effective_tcp_table_xyz_in()
    tcp_m = cfg.effective_tcp_table_xyz_m()
    nominal_in = tuple(float(v) for v in cfg.nominal_tcp_table_xyz_in)

    tcp_robot = table_point_to_planner_world_m(tcp_m, cfg.robot_in_table_frame)

    # Seed so finalize_grasp_targets reproduces this TCP XYZ.
    # Yaw/C is not taught (ik_ignore_yaw); A/B come from fixed_grasp_ab_deg.
    # pre_grasp_* fields are grab_sensor API only; epoxy has no mid-air waypoint.
    debug: dict[str, Any] = {
        "nominal_tcp_table_xyz_in": list(nominal_in),
        "safety_offset_y_in": float(cfg.safety_offset_y_in),
        "safety_offset_z_in": float(cfg.safety_offset_z_in),
        "tcp_table_xyz_in": list(tcp_in),
        "tcp_table_xyz_m": list(tcp_m),
        "tcp_robot_xyz_m": tcp_robot.round(4).tolist(),
        "sensor_table_xyz_m": [tcp_m[0], tcp_m[1], tcp_m[2]],
        "sensor_axis_deg": 0.0,
        "grasp_orientation_mode": "epoxy_pre_apply",
        "ik_ignore_yaw": bool(cfg.ik_ignore_yaw),
        "grasp_frame_summary": table_grasp_frame_summary(tcp_m, cfg.robot_in_table_frame),
        "posts": [],
        "epoxy_phase": "pre_apply",
    }

    rpy = np.zeros(3, dtype=np.float64)  # filled by finalize_grasp_targets
    return GraspTargets(
        pre_grasp_pos_m=tcp_robot.copy(),
        pre_grasp_rpy_rad=rpy.copy(),
        grasp_pos_m=tcp_robot,
        grasp_rpy_rad=rpy.copy(),
        object_pos_m=tcp_robot.copy(),
        object_rpy_rad=rpy.copy(),
        debug=debug,
    )
