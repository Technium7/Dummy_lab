from __future__ import annotations

import math
from typing import Sequence

JOINT_SIGNS = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
JOINT_OFFSETS_DEG = [0.0, 75.0, 180.0, 0.0, 0.0, 0.0]
HOME_JOINTS_DEG = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]

# Matches firmware CtrlStepMotor reductions for J1..J6 (dummy_ref_core DummyRobot ctor).
JOINT_REDUCTION = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
DOMINANT_REDUCTION = max(JOINT_REDUCTION)
MIN_SPEED = 10.0
MAX_SPEED = 30.0


def sim_rad_to_transformed_deg(joint_idx: int, sim_rad: float) -> float:
    sign = JOINT_SIGNS[joint_idx] if joint_idx < len(JOINT_SIGNS) else 1.0
    offset_deg = JOINT_OFFSETS_DEG[joint_idx] if joint_idx < len(JOINT_OFFSETS_DEG) else 0.0
    return sign * (offset_deg - math.degrees(sim_rad))


def transformed_deg_to_sim_rad(joint_idx: int, transformed_deg: float) -> float:
    sign = JOINT_SIGNS[joint_idx] if joint_idx < len(JOINT_SIGNS) else 1.0
    offset_deg = JOINT_OFFSETS_DEG[joint_idx] if joint_idx < len(JOINT_OFFSETS_DEG) else 0.0
    return math.radians(offset_deg - sign * transformed_deg)


def sim_vel_rad_to_transformed_deg_s(joint_idx: int, sim_vel_rad_s: float) -> float:
    sign = JOINT_SIGNS[joint_idx] if joint_idx < len(JOINT_SIGNS) else 1.0
    return -sign * math.degrees(sim_vel_rad_s)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def deg_speed_to_ascii_speed(v_deg_per_s_list: Sequence[float]) -> float:
    demand = max(abs(v_deg_per_s_list[j]) * JOINT_REDUCTION[j] for j in range(6)) / 36.0
    min_ascii = MIN_SPEED * DOMINANT_REDUCTION / 36.0
    max_ascii = MAX_SPEED * DOMINANT_REDUCTION / 36.0
    return clamp(demand, min_ascii, max_ascii)

