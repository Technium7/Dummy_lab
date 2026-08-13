from __future__ import annotations

import math
from typing import Sequence

JOINT_SIGNS = [-1.0, -1.0, 1.0, 1.0, 1.0, -1.0]
JOINT_OFFSETS_DEG = [0.0, 75.0, 180.0, 0.0, 0.0, 0.0]
HOME_JOINTS_DEG = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]
# Fallback if physical_env/robot_in_table.yaml is missing (matches that file's work pose).
_DEFAULT_WORK_JOINTS_DEG = [0.0, -55.0, 145.0, 0.0, 0.0, 0.0]
_WORK_JOINTS_CACHE: list[float] | None = None


def work_joints_deg() -> list[float]:
    """Firmware-frame WORK pose for ``T`` (``physical_env/robot_in_table.yaml``)."""
    global _WORK_JOINTS_CACHE
    if _WORK_JOINTS_CACHE is not None:
        return list(_WORK_JOINTS_CACHE)
    vals = list(_DEFAULT_WORK_JOINTS_DEG)
    try:
        from pathlib import Path

        import yaml

        yaml_path = Path(__file__).resolve().parents[2] / "physical_env" / "robot_in_table.yaml"
        if yaml_path.is_file():
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            raw = data.get("work_position_deg")
            if raw is not None:
                vals = [float(v) for v in raw][:6]
                while len(vals) < 6:
                    vals.append(0.0)
    except Exception:
        pass
    _WORK_JOINTS_CACHE = vals
    return list(vals)


# Matches firmware CtrlStepMotor reductions for J1..J6 (dummy_ref_core DummyRobot ctor).
JOINT_REDUCTION = [50.0, 50.0, 50.0, 50.0, 50.0, 30.0]
DOMINANT_REDUCTION = max(JOINT_REDUCTION)
MIN_SPEED = 5.0
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


FIRMWARE_JOINT_LIMITS_DEG = [
    (-180.0, 180.0),  # J1
    (-75.0, 90.0),  # J2
    (35.0, 180.0),  # J3
    (-180.0, 180.0),  # J4
    (-120.0, 120.0),  # J5
    (-180.0, 180.0),  # J6
]

# Joints that are full-turn continuous in firmware space (±180). Values outside
# the interval are equivalent modulo 360 and must be wrapped, not hard-clamped
# (clamping e.g. 241° → 180° destroys the pose and scrambles streamed paths).
_CONTINUOUS_FIRMWARE_JOINTS = frozenset({0, 3, 5})


def shortest_deg_delta(from_deg: float, to_deg: float) -> float:
    """Signed shortest delta on a circle (result in (-180, 180])."""
    return ((float(to_deg) - float(from_deg) + 180.0) % 360.0) - 180.0


def continuous_firmware_joint(joint_idx: int) -> bool:
    return int(joint_idx) in _CONTINUOUS_FIRMWARE_JOINTS


def wrap_to_interval_deg(value: float, lower: float, upper: float) -> float:
    """Map ``value`` into ``[lower, upper]`` by adding/subtracting 360° when possible."""
    lo = float(lower)
    hi = float(upper)
    v = float(value)
    width = hi - lo
    if width <= 0.0:
        return v
    if width >= 360.0 - 1e-9:
        # Full-circle principal range (typically [-180, 180]).
        return ((v - lo) % 360.0) + lo
    # Limited joint: try ±360 k until inside, else leave for clamp.
    for _ in range(8):
        if lo <= v <= hi:
            return v
        if v > hi:
            v -= 360.0
        elif v < lo:
            v += 360.0
    return v


def normalize_firmware_joint_deg(joint_idx: int, value_deg: float) -> float:
    """Wrap into firmware range when an equivalent angle exists; else clamp."""
    lo, hi = FIRMWARE_JOINT_LIMITS_DEG[joint_idx]
    v = float(value_deg)
    if joint_idx in _CONTINUOUS_FIRMWARE_JOINTS:
        return ((v + 180.0) % 360.0) - 180.0
    wrapped = wrap_to_interval_deg(v, lo, hi)
    if lo <= wrapped <= hi:
        return wrapped
    # Prefer nearest ±360 candidate to the interval midpoint, then clamp.
    mid = 0.5 * (lo + hi)
    best = min((v + 360.0 * k for k in range(-3, 4)), key=lambda c: abs(c - mid))
    return clamp(best, lo, hi)


def normalize_firmware_joints_deg(joints: Sequence[float]) -> list[float]:
    out: list[float] = []
    for i in range(min(6, len(joints))):
        out.append(normalize_firmware_joint_deg(i, float(joints[i])))
    while len(out) < 6:
        out.append(0.0)
    return out


def clamp_firmware_joints_deg(joints: Sequence[float]) -> list[float]:
    """Bring joints into firmware limits (wrap continuous / ±360 when equivalent)."""
    return normalize_firmware_joints_deg(joints)


def unwrap_firmware_trajectory_deg(
    waypoints_deg: Sequence[Sequence[float]],
    *,
    reference_deg: Sequence[float] | None = None,
) -> list[list[float]]:
    """Remove ±360 jumps on continuous joints so interpolators take the short path.

    Limited joints are only normalized into range (no unwrap across a hard stop).
    Absolute values may leave ±180 after unwrap so callers that stream to firmware
    should re-normalize per sample (or rely on ``normalize`` at send time).
    """
    pts = [[float(v) for v in row[:6]] for row in waypoints_deg]
    if not pts:
        return pts
    for i, row in enumerate(pts):
        while len(row) < 6:
            row.append(0.0)
        pts[i] = row

    prev = (
        normalize_firmware_joints_deg(reference_deg)
        if reference_deg is not None and len(reference_deg) >= 6
        else normalize_firmware_joints_deg(pts[0])
    )
    out: list[list[float]] = []
    for i, row in enumerate(pts):
        q = normalize_firmware_joints_deg(row)
        if i == 0 and reference_deg is None:
            out.append(q)
            prev = q
            continue
        unwrapped = list(q)
        for j in _CONTINUOUS_FIRMWARE_JOINTS:
            # Shift by ±360 so step from prev is in (-180, 180].
            d = q[j] - prev[j]
            d = ((d + 180.0) % 360.0) - 180.0
            unwrapped[j] = prev[j] + d
        out.append(unwrapped)
        prev = unwrapped
    return out


def firmware_joint_limits_sim_rad() -> tuple[list[float], list[float]]:
    """Sim-rad joint limits corresponding to ``FIRMWARE_JOINT_LIMITS_DEG``."""
    lowers: list[float] = []
    uppers: list[float] = []
    for j, (lo, hi) in enumerate(FIRMWARE_JOINT_LIMITS_DEG):
        a = transformed_deg_to_sim_rad(j, lo)
        b = transformed_deg_to_sim_rad(j, hi)
        lowers.append(min(a, b))
        uppers.append(max(a, b))
    return lowers, uppers


def deg_speed_to_ascii_speed(v_deg_per_s_list: Sequence[float]) -> float:
    demand = max(abs(v_deg_per_s_list[j]) * JOINT_REDUCTION[j] for j in range(6)) / 36.0
    min_ascii = MIN_SPEED * DOMINANT_REDUCTION / 36.0
    max_ascii = MAX_SPEED * DOMINANT_REDUCTION / 36.0
    return clamp(demand, min_ascii, max_ascii)


def max_ascii_speed() -> float:
    """Peak ``&`` speed field for streaming moves (deg/s at the protocol layer)."""
    return deg_speed_to_ascii_speed([MAX_SPEED] * 6)


def home_stream_ascii_speed() -> float:
    """Use when driving the robot to WORK via streamed waypoints (``T`` latch)."""
    return max_ascii_speed()

