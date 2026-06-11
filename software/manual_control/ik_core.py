from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import pybullet as p


def joint_limits_or_defaults(joint_type: int, lower: float, upper: float) -> Tuple[float, float]:
    if lower <= upper and not (lower == 0.0 and upper == -1.0):
        return lower, upper
    if joint_type == p.JOINT_REVOLUTE:
        return -math.pi, math.pi
    if joint_type == p.JOINT_PRISMATIC:
        return -0.2, 0.2
    return -1.0, 1.0


def get_controllable_joints(robot_id: int) -> Tuple[List[int], List[float], List[float], List[float], List[str]]:
    indices: List[int] = []
    lowers: List[float] = []
    uppers: List[float] = []
    ranges: List[float] = []
    names: List[str] = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            continue
        lower, upper = joint_limits_or_defaults(info[2], float(info[8]), float(info[9]))
        indices.append(i)
        lowers.append(lower)
        uppers.append(upper)
        ranges.append(max(1e-6, upper - lower))
        names.append(info[1].decode("utf-8", errors="replace"))
    return indices, lowers, uppers, ranges, names


def resolve_end_link(robot_id: int, hint: str) -> int:
    num_joints = p.getNumJoints(robot_id)
    if num_joints <= 0:
        return -1
    if hint:
        if hint.isdigit() or (hint.startswith("-") and hint[1:].isdigit()):
            idx = int(hint)
            if -1 <= idx < num_joints:
                return idx
            raise SystemExit(f"--end-link index out of range: {idx}")
        lookup = hint.lower()
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            if lookup in {
                info[12].decode("utf-8", errors="replace").lower(),
                info[1].decode("utf-8", errors="replace").lower(),
            }:
                return i
        raise SystemExit(f"--end-link={hint!r} did not match any link/joint name")
    return num_joints - 1


def get_end_link_pose(
    robot_id: int, end_link_idx: int
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """World-frame end-link position (m) and orientation as roll-pitch-yaw (rad)."""
    state = p.getLinkState(robot_id, end_link_idx, computeForwardKinematics=True)
    pos = tuple(float(v) for v in state[4])
    rpy = tuple(float(v) for v in p.getEulerFromQuaternion(state[5]))
    return pos, rpy


def end_link_name(robot_id: int, end_link_idx: int) -> str:
    return p.getJointInfo(robot_id, end_link_idx)[12].decode("utf-8", errors="replace")


def is_key_down(events: Dict[int, int], char: str) -> bool:
    lower = ord(char.lower())
    upper = ord(char.upper())
    return ((events.get(lower, 0) & p.KEY_IS_DOWN) != 0) or ((events.get(upper, 0) & p.KEY_IS_DOWN) != 0)


def solve_linear_system(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(a)
    if n == 0 or len(b) != n:
        return None
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def compute_dls_joint_velocity(
    robot_id: int,
    end_link_idx: int,
    joint_indices: Sequence[int],
    task_twist: Sequence[float],
    damping: float,
    rotation_weight: float,
) -> Optional[List[float]]:
    q_all = [float(p.getJointState(robot_id, j)[0]) for j in range(p.getNumJoints(robot_id))]
    zeros = [0.0] * len(q_all)
    jacobian = p.calculateJacobian(robot_id, end_link_idx, [0.0, 0.0, 0.0], q_all, zeros, zeros)
    linear_jac = jacobian[0]
    angular_jac = jacobian[1]
    rows: List[List[float]] = []
    weights = (1.0, 1.0, 1.0, float(rotation_weight), float(rotation_weight))
    for src, w in zip((linear_jac[0], linear_jac[1], linear_jac[2], angular_jac[0], angular_jac[1]), weights):
        rows.append([float(src[j]) * w for j in joint_indices])
    n = len(joint_indices)
    if n == 0:
        return None
    n_task = 5
    jt = [[rows[r][c] for r in range(n_task)] for c in range(n)]
    h = [[0.0 for _ in range(n)] for _ in range(n)]
    g = [0.0 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            h[i][j] = sum(jt[i][k] * rows[k][j] for k in range(n_task))
        h[i][i] += damping * damping
        g[i] = sum(jt[i][k] * float(task_twist[k]) * weights[k] for k in range(n_task))
    return solve_linear_system(h, g)

