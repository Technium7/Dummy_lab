"""Shared pose conventions: manual_control GUI (robot frame A/B/C) ↔ planner (world RPY).

The End Link GUI (``manual_control/end_link_window.py``) shows:
  - position in **robot base frame** (m)
  - orientation **A/B/C (deg)** semantic: HOME joints ``[0,0,90,0,0,0]`` → ``A=0, B=0, C=180``

The trajectory planner (``trajectory_planner/scene.py``) uses PyBullet **world** frame
with the robot base at the origin rotated 180° about Z.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as SciRotation

from manual_control.ik_core import (
    abc_relative_to_semantic_home,
    measure_home_abc_reference,
    quat_to_rotmat_colmajor,
    rotmat_colmajor_to_firmware_abc_deg,
)

# Same as trajectory_planner.scene._BASE_ORIENTATION_Z_180
_BASE_ORIENTATION_Z_180 = p.getQuaternionFromEuler([0.0, 0.0, math.pi])
_IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


def semantic_abc_to_firmware_abc(
    semantic_abc_deg: Sequence[float],
    home_abc_ref: Sequence[float],
) -> Tuple[float, float, float]:
    """Invert ``abc_relative_to_semantic_home``."""
    a_s, b_s, c_s = (float(semantic_abc_deg[i]) for i in range(3))
    ah, bh, ch = (float(home_abc_ref[i]) for i in range(3))
    return (a_s + ah, b_s + bh, c_s + ch - 180.0)


def firmware_abc_deg_to_quat(a_deg: float, b_deg: float, c_deg: float) -> list[float]:
    """Inverse of ``rotmat_colmajor_to_firmware_abc_deg`` (numerical, firmware ``#GETLPOS``)."""
    from scipy.optimize import minimize

    target = np.array([float(a_deg), float(b_deg), float(c_deg)], dtype=np.float64)

    def cost(quat: np.ndarray) -> float:
        qn = quat / max(float(np.linalg.norm(quat)), 1e-12)
        abc = np.array(
            rotmat_colmajor_to_firmware_abc_deg(quat_to_rotmat_colmajor(qn.tolist())),
            dtype=np.float64,
        )
        delta = (abc - target + 180.0) % 360.0 - 180.0
        return float(np.dot(delta, delta))

    best_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    best_val = cost(best_q)
    rng = np.random.default_rng(0)
    for _ in range(48):
        q0 = rng.standard_normal(4)
        q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
        res = minimize(cost, q0, method="Powell", options={"maxiter": 250})
        qn = res.x / max(float(np.linalg.norm(res.x)), 1e-12)
        val = cost(qn)
        if val < best_val:
            best_val = val
            best_q = qn
    if best_val > 0.25:
        raise ValueError(
            f"Could not invert firmware A/B/C ({a_deg:.2f}, {b_deg:.2f}, {c_deg:.2f}); "
            f"residual^2={best_val:.3f}"
        )
    return best_q.tolist()


def firmware_abc_deg_to_rotmat_colmajor(a_deg: float, b_deg: float, c_deg: float) -> list[float]:
    return quat_to_rotmat_colmajor(firmware_abc_deg_to_quat(a_deg, b_deg, c_deg))


def rotmat_colmajor_to_quat(rot_colmajor: Sequence[float]) -> list[float]:
    matrix = np.array(
        [
            [rot_colmajor[0], rot_colmajor[3], rot_colmajor[6]],
            [rot_colmajor[1], rot_colmajor[4], rot_colmajor[7]],
            [rot_colmajor[2], rot_colmajor[5], rot_colmajor[8]],
        ],
        dtype=np.float64,
    )
    quat = SciRotation.from_matrix(matrix).as_quat()
    return [float(v) for v in quat]


def robot_frame_pos_to_world(pos_robot_m: Sequence[float]) -> Tuple[float, float, float]:
    world_pos, _ = p.multiplyTransforms(
        [0.0, 0.0, 0.0],
        _BASE_ORIENTATION_Z_180,
        list(pos_robot_m)[:3],
        _IDENTITY_QUAT,
    )
    return tuple(float(v) for v in world_pos)


def world_pos_to_robot_frame(pos_world_m: Sequence[float]) -> Tuple[float, float, float]:
    inv_pos, inv_orn = p.invertTransform([0.0, 0.0, 0.0], _BASE_ORIENTATION_Z_180)
    rel_pos, _ = p.multiplyTransforms(inv_pos, inv_orn, list(pos_world_m)[:3], _IDENTITY_QUAT)
    return tuple(float(v) for v in rel_pos)


def grasp_c_deg_from_robot_xy(x_robot_m: float, y_robot_m: float) -> float:
    """Heading C (deg) from sensor/grasp XY in robot frame (matches taught down-grasp sweep)."""
    return float(math.degrees(math.atan2(y_robot_m, x_robot_m)))


def semantic_abc_deg_to_world_rpy_rad(
    semantic_abc_deg: Sequence[float],
    home_abc_ref: Sequence[float],
) -> Tuple[float, float, float]:
    """Convert GUI semantic A/B/C to PyBullet world roll-pitch-yaw (rad) for IK/planner."""
    fw_a, fw_b, fw_c = semantic_abc_to_firmware_abc(semantic_abc_deg, home_abc_ref)
    rot = firmware_abc_deg_to_rotmat_colmajor(fw_a, fw_b, fw_c)
    quat_robot = rotmat_colmajor_to_quat(rot)
    _, world_quat = p.multiplyTransforms([0.0, 0.0, 0.0], _BASE_ORIENTATION_Z_180, [0.0, 0.0, 0.0], quat_robot)
    rpy = p.getEulerFromQuaternion(world_quat)
    return tuple(float(v) for v in rpy)


def world_rpy_rad_to_semantic_abc_deg(
    world_rpy_rad: Sequence[float],
    home_abc_ref: Sequence[float],
) -> Tuple[float, float, float]:
    """Convert planner world RPY back to GUI semantic A/B/C (deg)."""
    world_quat = p.getQuaternionFromEuler(list(world_rpy_rad))
    inv_pos, inv_orn = p.invertTransform([0.0, 0.0, 0.0], _BASE_ORIENTATION_Z_180)
    _, rel_quat = p.multiplyTransforms(inv_pos, inv_orn, [0.0, 0.0, 0.0], world_quat)
    abc = rotmat_colmajor_to_firmware_abc_deg(quat_to_rotmat_colmajor(rel_quat))
    return abc_relative_to_semantic_home(abc, home_abc_ref)


def resolve_target_end_link(
    target: dict,
    *,
    home_abc_ref: Sequence[float],
    position_robot_m: Sequence[float] | None = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Resolve ``target_end_link`` block to planner world (pos_m, rpy_rad).

    Supports:
      - ``position_robot_m`` + ``orientation_abc_deg`` (manual_control GUI convention)
      - legacy ``position_m`` + ``orientation_rpy_rad`` (world frame)
    """
    if position_robot_m is not None:
        pos_robot = position_robot_m
    elif "position_robot_m" in target:
        pos_robot = target["position_robot_m"]
    elif "position_m" in target:
        pos_world = target["position_m"]
        pos_robot = world_pos_to_robot_frame(pos_world)
    else:
        raise KeyError("target_end_link requires position_robot_m or position_m")

    pos_robot = tuple(float(v) for v in pos_robot[:3])
    pos_world = robot_frame_pos_to_world(pos_robot)

    abc_raw = target.get("orientation_abc_deg")
    if abc_raw is not None:
        abc = [float(v) for v in abc_raw[:3]]
        if bool(target.get("orientation_c_from_xy", False)):
            abc[2] = grasp_c_deg_from_robot_xy(pos_robot[0], pos_robot[1]) + abc[2]
        rpy = semantic_abc_deg_to_world_rpy_rad(abc, home_abc_ref)
        return pos_world, rpy

    if "orientation_rpy_rad" in target:
        rpy = tuple(float(v) for v in target["orientation_rpy_rad"][:3])
        return pos_world, rpy

    raise KeyError("target_end_link requires orientation_abc_deg or orientation_rpy_rad")


def measure_home_abc_ref_for_scene(
    robot_id: int,
    end_link_idx: int,
    joint_indices: Sequence[int],
    home_joints_deg: Sequence[float],
) -> Tuple[float, float, float]:
    return measure_home_abc_reference(robot_id, end_link_idx, joint_indices, home_joints_deg)
