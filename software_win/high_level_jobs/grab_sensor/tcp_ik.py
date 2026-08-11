"""TCP FK/IK for grasp planning (URDF ``grasp_tcp_link``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from trajectory_planner.collision import is_state_valid
from trajectory_planner.goals import solve_goal_from_pose
from trajectory_planner.scene import (
    PlanningScene,
    pose_error_rad,
    refine_joint_config_to_pose,
    set_joint_positions,
)

from .models import GraspTargets

_IK_REST_SEED = 42
_IK_DIRECT_ATTEMPTS = 8
_IK_FALLBACK_ATTEMPTS = 12
_TCP_POS_TOL_M = 1e-5


def _full_rest_poses(scene: PlanningScene, controllable_q: Sequence[float]) -> List[float]:
    num_joints = p.getNumJoints(scene.robot_id, physicsClientId=scene.client_id)
    rest = [0.0] * num_joints
    for j_idx, value in zip(scene.joint_indices, controllable_q):
        rest[j_idx] = float(value)
    return rest


def solve_tcp_ik(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_orn_quat: Sequence[float],
    rest_pose: Sequence[float],
) -> List[float]:
    """PyBullet IK on ``grasp_tcp_link`` (position + orientation coupled in the chain)."""
    if scene.tcp_link_idx is None:
        raise ValueError("scene has no tcp_link_idx")
    n = len(scene.joint_indices)
    lowers = scene.lower_limits
    uppers = scene.upper_limits
    ranges = [max(1e-6, u - lo) for lo, u in zip(lowers, uppers)]
    sol = p.calculateInverseKinematics(
        bodyUniqueId=scene.robot_id,
        endEffectorLinkIndex=scene.tcp_link_idx,
        targetPosition=list(target_pos),
        targetOrientation=list(target_orn_quat),
        lowerLimits=lowers,
        upperLimits=uppers,
        jointRanges=ranges,
        restPoses=_full_rest_poses(scene, rest_pose),
        maxNumIterations=500,
        residualThreshold=1e-8,
        physicsClientId=scene.client_id,
    )
    return [float(sol[j]) for j in scene.joint_indices[:n]]


def measure_tcp_target_pose_error(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    q: Sequence[float],
    tcp: TcpOffset,
) -> Tuple[float, float]:
    """TCP position + ``grasp_tcp_link`` orientation error vs target (rad)."""
    set_joint_positions(scene, q)
    tip_pos = tcp.tip_world(scene)
    tip_rpy = tcp_world_rpy_from_q(scene, q, tcp)
    return pose_error_rad(tip_pos, tip_rpy, target_pos, target_rpy)


def refine_joints_to_balanced_tcp_pose(
    scene: PlanningScene,
    tip_target_pos: Sequence[float],
    target_rpy: Sequence[float],
    q_init: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_mm: float = 20.0,
    ori_tol_deg: float = 50.0,
    ori_vs_pos: float = 1.0,
) -> List[float]:
    """Joint refine minimizing TCP and orientation together.

    ``cost = (pos_mm / pos_tol_mm)^2 + ori_vs_pos * (ori_deg / ori_tol_deg)^2``

    ``ori_vs_pos=1`` treats being at the position tol and orientation tol as equal;
    ``>1`` favors orientation, ``<1`` favors TCP XYZ.
    """
    if not tcp.has_tcp(scene):
        return list(q_init)

    tip_target = np.asarray(tip_target_pos, dtype=np.float64)[:3]
    target_rpy_arr = np.asarray(target_rpy, dtype=np.float64)
    low = np.asarray(scene.lower_limits, dtype=np.float64)
    high = np.asarray(scene.upper_limits, dtype=np.float64)
    q = np.clip(np.asarray(q_init, dtype=np.float64), low, high)
    pos_scale = max(float(pos_tol_mm), 1e-3)
    ori_scale = max(float(ori_tol_deg), 1e-3)
    w_ori = max(float(ori_vs_pos), 0.0)

    def metric(q_try: np.ndarray) -> float:
        set_joint_positions(scene, q_try)
        tip_pos = tcp.tip_world(scene)
        tip_rpy = tcp_world_rpy_from_q(scene, q_try, tcp)
        pos_err_m, ori_err_rad = pose_error_rad(tip_pos, tip_rpy, tip_target, target_rpy_arr)
        pos_n = (pos_err_m * 1000.0) / pos_scale
        ori_n = math.degrees(ori_err_rad) / ori_scale
        return pos_n * pos_n + w_ori * ori_n * ori_n

    steps = [0.02, 0.005, 0.001, 0.0002, 0.00005, 0.00001]
    for step in steps:
        improved = True
        while improved:
            improved = False
            base = metric(q)
            for i in range(len(q)):
                for delta in (step, -step):
                    q_try = q.copy()
                    q_try[i] += delta
                    q_try = np.clip(q_try, low, high)
                    m = metric(q_try)
                    if m + 1e-12 < base:
                        q = q_try
                        base = m
                        improved = True

    set_joint_positions(scene, q)
    return q.tolist()


def tcp_ik_goal_from_home(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    home_q: Sequence[float],
    tcp: TcpOffset | None = None,
    *,
    prefer_position: bool = True,
    ori_tol_rad: float = math.radians(25.0),
    pos_hold_m: float = 0.003,
    pos_tol_mm: float | None = None,
    ori_tol_deg: float | None = None,
    ori_vs_pos: float = 1.0,
    balanced: bool = True,
) -> List[float] | None:
    """HOME warm-start TCP IK for an arbitrary pose (resets arm to HOME first).

    Default ``balanced=True`` refines with simultaneous TCP+ori cost
    (see ``refine_joints_to_balanced_tcp_pose`` / config ``ik_ori_vs_pos``).

    Legacy ``prefer_position`` path: TCP XYZ first, then soft ori hold.
    """
    tool = tcp if tcp is not None else TcpOffset()
    if not tool.has_tcp(scene):
        return None
    warm_q = [float(v) for v in home_q]
    set_joint_positions(scene, warm_q)
    tip = np.asarray(target_pos, dtype=np.float64)[:3]
    target_orn = p.getQuaternionFromEuler([float(v) for v in target_rpy[:3]])
    try:
        q_goal = solve_tcp_ik(
            scene,
            tip,
            target_orn,
            warm_q,
        )
    except (ValueError, TypeError):
        return None
    if not is_state_valid(scene, q_goal):
        return None

    ori_deg = float(ori_tol_deg) if ori_tol_deg is not None else math.degrees(float(ori_tol_rad))
    pos_mm = float(pos_tol_mm) if pos_tol_mm is not None else 20.0

    if balanced:
        q_goal = refine_joints_to_balanced_tcp_pose(
            scene,
            tip,
            target_rpy,
            q_goal,
            tool,
            pos_tol_mm=pos_mm,
            ori_tol_deg=ori_deg,
            ori_vs_pos=float(ori_vs_pos),
        )
        if not is_state_valid(scene, q_goal):
            return None
        return q_goal

    if prefer_position:
        q_goal = refine_joints_to_tcp_position(
            scene, tip, q_goal, tool, pos_tol_m=_TCP_POS_TOL_M
        )
        if not is_state_valid(scene, q_goal):
            return None
        q_goal = refine_joints_to_wrist_orientation_hold_tcp(
            scene,
            target_rpy,
            q_goal,
            tool,
            pos_tol_m=pos_hold_m,
            ori_tol_rad=math.radians(ori_deg),
        )
        if not is_state_valid(scene, q_goal):
            return None
    return q_goal


def grasp_ik_goal_from_home(
    scene: PlanningScene,
    targets: GraspTargets,
    home_q: Sequence[float],
    tcp: TcpOffset | None = None,
    *,
    prefer_position: bool = True,
    ori_tol_rad: float = math.radians(25.0),
    pos_tol_mm: float | None = None,
    ori_tol_deg: float | None = None,
    ori_vs_pos: float = 1.0,
    balanced: bool = True,
) -> List[float] | None:
    """HOME warm-start TCP IK for ``targets`` grasp pose (resets arm to HOME first)."""
    tool = tcp if tcp is not None else TcpOffset()
    return tcp_ik_goal_from_home(
        scene,
        targets.grasp_pos_m,
        targets.grasp_rpy_rad,
        home_q,
        tool,
        prefer_position=prefer_position,
        ori_tol_rad=ori_tol_rad,
        pos_tol_mm=pos_tol_mm,
        ori_tol_deg=ori_tol_deg,
        ori_vs_pos=ori_vs_pos,
        balanced=balanced,
    )


@dataclass(frozen=True)
class TcpOffset:
    """Tool center point from URDF ``grasp_tcp_link`` FK (see ``paths.tcp_link`` in config)."""

    @staticmethod
    def has_tcp(scene: PlanningScene) -> bool:
        return scene.tcp_link_idx is not None

    def _link_state(
        self, scene: PlanningScene, link_idx: int
    ) -> tuple[np.ndarray, list[float]]:
        state = p.getLinkState(
            scene.robot_id,
            link_idx,
            computeForwardKinematics=True,
            physicsClientId=scene.client_id,
        )
        return np.asarray(state[4], dtype=np.float64), [float(v) for v in state[5]]

    def urdf_offset_in_link6_m(self, scene: PlanningScene) -> tuple[float, float, float]:
        """Fixed ``grasp_tcp_fixed`` origin in link6 frame (from URDF)."""
        wrist_pos, wrist_orn = self._link_state(scene, scene.end_link_idx)
        tcp_pos, _ = self._link_state(scene, scene.tcp_link_idx)
        inv_pos, inv_orn = p.invertTransform(list(wrist_pos), wrist_orn)
        local, _ = p.multiplyTransforms(inv_pos, inv_orn, list(tcp_pos), [0.0, 0.0, 0.0, 1.0])
        return (float(local[0]), float(local[1]), float(local[2]))

    def wrist_world(self, scene: PlanningScene) -> np.ndarray:
        return self._link_state(scene, scene.end_link_idx)[0]

    def tip_world(self, scene: PlanningScene) -> np.ndarray:
        if not self.has_tcp(scene):
            return self.wrist_world(scene)
        return self._link_state(scene, scene.tcp_link_idx)[0]

    def wrist_attach_kwargs(self, scene: PlanningScene) -> dict:
        return {
            "parentObjectUniqueId": scene.robot_id,
            "parentLinkIndex": scene.end_link_idx,
            "physicsClientId": scene.client_id,
        }

    def tcp_attach_kwargs(self, scene: PlanningScene) -> dict:
        link_idx = scene.tcp_link_idx if self.has_tcp(scene) else scene.end_link_idx
        return {
            "parentObjectUniqueId": scene.robot_id,
            "parentLinkIndex": link_idx,
            "physicsClientId": scene.client_id,
        }

    def wrist_pos_from_tip(
        self,
        scene: PlanningScene,
        tip_pos_m: Sequence[float],
        grasp_rpy_rad: Sequence[float],
    ) -> np.ndarray:
        """Wrist position for a tip target and taught link6 orientation (URDF offset)."""
        if not self.has_tcp(scene):
            return np.asarray(tip_pos_m, dtype=np.float64)[:3].copy()
        link_orn = p.getQuaternionFromEuler(list(grasp_rpy_rad))
        world_off, _ = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            link_orn,
            list(self.urdf_offset_in_link6_m(scene)),
            [0.0, 0.0, 0.0, 1.0],
        )
        return np.asarray(tip_pos_m, dtype=np.float64)[:3] - np.asarray(world_off, dtype=np.float64)


def link_pose_from_q(
    scene: PlanningScene,
    q: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Wrist link pose at ``q`` (sim radians after mapping)."""
    set_joint_positions(scene, q)
    state = p.getLinkState(
        scene.robot_id,
        scene.end_link_idx,
        computeForwardKinematics=True,
        physicsClientId=scene.client_id,
    )
    link_pos = np.asarray(state[4], dtype=np.float64)
    link_orn = [float(v) for v in state[5]]
    link_rpy = np.asarray(p.getEulerFromQuaternion(link_orn), dtype=np.float64)
    return link_pos, link_rpy, link_orn


def target_pose_from_q(
    scene: PlanningScene,
    q: Sequence[float],
    tcp: TcpOffset,
) -> tuple[np.ndarray, np.ndarray]:
    set_joint_positions(scene, q)
    _, link_rpy, _ = link_pose_from_q(scene, q)
    return tcp.tip_world(scene), link_rpy


def tcp_world_rpy_from_q(
    scene: PlanningScene,
    q: Sequence[float],
    tcp: TcpOffset,
) -> np.ndarray:
    """``grasp_tcp_link`` orientation in planner world (rad); equals link6 if fixed joint has zero rpy."""
    set_joint_positions(scene, q)
    if tcp.has_tcp(scene):
        _, link_orn = tcp._link_state(scene, scene.tcp_link_idx)
    else:
        _, link_orn = tcp._link_state(scene, scene.end_link_idx)
    return np.asarray(p.getEulerFromQuaternion(link_orn), dtype=np.float64)


def wrist_tcp_distance_mm(
    scene: PlanningScene,
    q: Sequence[float],
    tcp: TcpOffset,
) -> float:
    """Wrist-to-TCP distance from URDF FK (mm)."""
    if not tcp.has_tcp(scene):
        return 0.0
    set_joint_positions(scene, q)
    return float(np.linalg.norm(tcp.tip_world(scene) - tcp.wrist_world(scene))) * 1000.0


def tcp_error_mm(
    scene: PlanningScene,
    target_pos: Sequence[float],
    q: Sequence[float],
    tcp: TcpOffset,
) -> float:
    set_joint_positions(scene, q)
    return float(np.linalg.norm(tcp.tip_world(scene) - np.asarray(target_pos, dtype=np.float64))) * 1000.0


def print_grasp_ik_debug(
    scene: PlanningScene,
    targets: GraspTargets,
    goal_q: Sequence[float],
    *,
    label: str,
    warm_start_q: Sequence[float] | None = None,
    tcp: TcpOffset | None = None,
) -> float:
    """Print joint solution, TCP FK pose, and target pose. Returns TCP position error (mm)."""
    tool = tcp if tcp is not None else TcpOffset()
    set_joint_positions(scene, goal_q)
    target_pos = np.asarray(targets.grasp_pos_m, dtype=np.float64)[:3]
    target_rpy_deg = [round(math.degrees(v), 4) for v in targets.grasp_rpy_rad]
    tip_pos = tool.tip_world(scene)
    tip_rpy_deg = [round(math.degrees(v), 4) for v in tcp_world_rpy_from_q(scene, goal_q, tool)]
    _, link_rpy, _ = link_pose_from_q(scene, goal_q)
    link_rpy_deg = [round(math.degrees(v), 4) for v in link_rpy]
    pos_err_mm = tcp_error_mm(scene, target_pos, goal_q, tool)
    _, ori_err = measure_tcp_target_pose_error(scene, target_pos, targets.grasp_rpy_rad, goal_q, tool)

    # Geometric approach: wrist→TCP vs world -Z (what you see in the GUI)
    wrist = tool.wrist_world(scene)
    approach = tip_pos - wrist
    n = float(np.linalg.norm(approach))
    tilt_deg = float("nan")
    if n > 1e-9:
        approach_u = approach / n
        tilt_deg = math.degrees(math.acos(float(np.clip(-approach_u[2], -1.0, 1.0))))

    print(f"[{label}] grasp TCP target pos m: {target_pos.round(6).tolist()}")
    print(f"[{label}] grasp TCP target rpy deg: {target_rpy_deg}")
    if warm_start_q is not None:
        print(f"[{label}] IK warm-start joints sim rad: {[round(float(v), 6) for v in warm_start_q]}")
    print(f"[{label}] IK goal joints sim rad: {[round(float(v), 6) for v in goal_q]}")
    print(f"[{label}] FK TCP pos m: {tip_pos.round(6).tolist()}")
    print(f"[{label}] FK TCP rpy deg: {tip_rpy_deg}")
    print(f"[{label}] FK link6 rpy deg: {link_rpy_deg}")
    print(
        f"[{label}] TCP err: {pos_err_mm:.3f} mm  ori err (vs target RPY): {math.degrees(ori_err):.3f} deg  "
        f"approach tilt from vertical: {tilt_deg:.2f} deg"
    )
    return pos_err_mm


def measure_target_pose_error(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    q: Sequence[float],
    tcp: TcpOffset,
) -> Tuple[float, float]:
    pos, rpy = target_pose_from_q(scene, q, tcp)
    return pose_error_rad(pos, rpy, target_pos, target_rpy)


def refine_joints_to_tcp_position(
    scene: PlanningScene,
    tip_target_pos: Sequence[float],
    q_init: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_m: float = _TCP_POS_TOL_M,
) -> List[float]:
    if not tcp.has_tcp(scene):
        return list(q_init)

    target = np.asarray(tip_target_pos, dtype=np.float64)
    low = np.asarray(scene.lower_limits, dtype=np.float64)
    high = np.asarray(scene.upper_limits, dtype=np.float64)
    q = np.clip(np.asarray(q_init, dtype=np.float64), low, high)

    def metric(q_try: np.ndarray) -> float:
        set_joint_positions(scene, q_try)
        return float(np.linalg.norm(tcp.tip_world(scene) - target))

    steps = [0.02, 0.005, 0.001, 0.0002, 0.00005, 0.00001, 0.000002]
    for step in steps:
        improved = True
        while improved:
            improved = False
            base = metric(q)
            if base <= pos_tol_m:
                break
            for i in range(len(q)):
                for delta in (step, -step):
                    q_try = q.copy()
                    q_try[i] += delta
                    q_try = np.clip(q_try, low, high)
                    m = metric(q_try)
                    if m + 1e-12 < base:
                        q = q_try
                        base = m
                        improved = True
        if metric(q) <= pos_tol_m:
            break

    set_joint_positions(scene, q)
    return q.tolist()


def refine_joints_to_tcp_pose(
    scene: PlanningScene,
    tip_target_pos: Sequence[float],
    target_rpy: Sequence[float],
    q_init: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_m: float = _TCP_POS_TOL_M,
    ori_tol_rad: float = 0.02,
) -> List[float]:
    """Joint coordinate descent on TCP position + wrist orientation."""
    if not tcp.has_tcp(scene):
        return refine_joint_config_to_pose(
            scene,
            tip_target_pos,
            target_rpy,
            q_init,
            pos_tol_m=pos_tol_m,
            ori_tol_rad=ori_tol_rad,
        )

    tip_target = np.asarray(tip_target_pos, dtype=np.float64)
    target_rpy_arr = np.asarray(target_rpy, dtype=np.float64)
    low = np.asarray(scene.lower_limits, dtype=np.float64)
    high = np.asarray(scene.upper_limits, dtype=np.float64)
    q = np.clip(np.asarray(q_init, dtype=np.float64), low, high)

    def metric(q_try: np.ndarray) -> float:
        set_joint_positions(scene, q_try)
        tip_pos = tcp.tip_world(scene)
        _, link_rpy, _ = link_pose_from_q(scene, q_try)
        pos_err_m, ori_err_rad = pose_error_rad(tip_pos, link_rpy, tip_target, target_rpy_arr)
        return pos_err_m * 10000.0 + ori_err_rad

    steps = [0.02, 0.005, 0.001, 0.0002, 0.00005, 0.00001]
    for step in steps:
        improved = True
        while improved:
            improved = False
            base = metric(q)
            for i in range(len(q)):
                for delta in (step, -step):
                    q_try = q.copy()
                    q_try[i] += delta
                    q_try = np.clip(q_try, low, high)
                    m = metric(q_try)
                    if m + 1e-12 < base:
                        q = q_try
                        base = m
                        improved = True
            set_joint_positions(scene, q)
            tip_pos = tcp.tip_world(scene)
            _, link_rpy, _ = link_pose_from_q(scene, q)
            pos_err_m, ori_err_rad = pose_error_rad(tip_pos, link_rpy, tip_target, target_rpy_arr)
            if pos_err_m <= pos_tol_m and ori_err_rad <= ori_tol_rad:
                break

    set_joint_positions(scene, q)
    return q.tolist()


def refine_joints_to_wrist_orientation_hold_tcp(
    scene: PlanningScene,
    target_rpy: Sequence[float],
    q_init: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_m: float = _TCP_POS_TOL_M,
    ori_tol_rad: float = 0.02,
) -> List[float]:
    """Align wrist (link6) RPY to ``target_rpy`` while keeping TCP position fixed."""
    if not tcp.has_tcp(scene):
        return list(q_init)

    target_rpy_arr = np.asarray(target_rpy, dtype=np.float64)
    low = np.asarray(scene.lower_limits, dtype=np.float64)
    high = np.asarray(scene.upper_limits, dtype=np.float64)
    q = np.clip(np.asarray(q_init, dtype=np.float64), low, high)
    set_joint_positions(scene, q)
    tip_locked = tcp.tip_world(scene).copy()

    def metric(q_try: np.ndarray) -> float:
        set_joint_positions(scene, q_try)
        tip_pos = tcp.tip_world(scene)
        _, link_rpy, _ = link_pose_from_q(scene, q_try)
        pos_err_m, ori_err_rad = pose_error_rad(tip_pos, link_rpy, tip_locked, target_rpy_arr)
        return ori_err_rad + pos_err_m * 10000.0

    steps = [0.02, 0.005, 0.001, 0.0002, 0.00005, 0.00001]
    for step in steps:
        improved = True
        while improved:
            improved = False
            base = metric(q)
            for i in range(len(q)):
                for delta in (step, -step):
                    q_try = q.copy()
                    q_try[i] += delta
                    q_try = np.clip(q_try, low, high)
                    m = metric(q_try)
                    if m + 1e-12 < base:
                        q = q_try
                        base = m
                        improved = True
            set_joint_positions(scene, q)
            tip_pos = tcp.tip_world(scene)
            _, link_rpy, _ = link_pose_from_q(scene, q)
            pos_err_m, ori_err_rad = pose_error_rad(tip_pos, link_rpy, tip_locked, target_rpy_arr)
            if pos_err_m <= pos_tol_m and ori_err_rad <= ori_tol_rad:
                break

    set_joint_positions(scene, q)
    return q.tolist()


def refine_joint_config_to_target_pose(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    q_init: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_m: float = 1e-4,
    ori_tol_rad: float = 0.02,
) -> List[float]:
    if not tcp.has_tcp(scene):
        return refine_joint_config_to_pose(
            scene, target_pos, target_rpy, q_init, pos_tol_m=pos_tol_m, ori_tol_rad=ori_tol_rad
        )
    q = refine_joints_to_tcp_position(scene, target_pos, q_init, tcp, pos_tol_m=pos_tol_m)
    return refine_joints_to_wrist_orientation_hold_tcp(
        scene,
        target_rpy,
        q,
        tcp,
        pos_tol_m=pos_tol_m,
        ori_tol_rad=ori_tol_rad,
    )


def snap_joints_to_tcp_target(
    scene: PlanningScene,
    target_pos: Sequence[float],
    q: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_m: float = _TCP_POS_TOL_M,
) -> List[float]:
    if not tcp.has_tcp(scene):
        return list(q)
    q_out = refine_joints_to_tcp_position(scene, target_pos, q, tcp, pos_tol_m=pos_tol_m)
    err_mm = tcp_error_mm(scene, target_pos, q_out, tcp)
    if err_mm > pos_tol_m * 1000.0 * 1.5:
        print(f"[grab_sensor] TCP snap residual: {err_mm:.3f} mm")
    return q_out


def refine_joint_config_to_target_position(
    scene: PlanningScene,
    target_pos: Sequence[float],
    q_init: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_m: float = 0.005,
) -> List[float]:
    return refine_joints_to_tcp_position(scene, target_pos, q_init, tcp, pos_tol_m=pos_tol_m)


def _report_tcp_goal(
    scene: PlanningScene,
    tip_pos: Sequence[float],
    tip_rpy: Sequence[float],
    q: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_err: float,
    ori_err: float,
) -> None:
    tcp_len_mm = wrist_tcp_distance_mm(scene, q, tcp)
    print(
        f"Goal TCP IK: tip_err={pos_err * 1000:.3f} mm, "
        f"ori_err={math.degrees(ori_err):.2f} deg, "
        f"wrist_tcp={tcp_len_mm:.1f} mm"
    )


def _solve_wrist_ik_for_tcp(
    scene: PlanningScene,
    tip_pos: Sequence[float],
    tip_rpy: Sequence[float],
    start_q: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_m: float,
    ik_ori_tol_rad: float,
    refine_ori_tol_rad: float,
    ik_pos_tol_m: float | None,
) -> Optional[List[float]]:
    """Direct 6-DOF IK on ``grasp_tcp_link`` + joint refinement."""
    best_pos_tol_m = pos_tol_m * 1.5 if ik_pos_tol_m is None else float(ik_pos_tol_m)
    tip_pos_arr = np.asarray(tip_pos, dtype=np.float64)
    tip_rpy_arr = np.asarray(tip_rpy, dtype=np.float64)
    target_orn = p.getQuaternionFromEuler(list(tip_rpy_arr))

    rng = np.random.default_rng(_IK_REST_SEED)
    low = np.asarray(scene.lower_limits)
    high = np.asarray(scene.upper_limits)
    best_q: Optional[List[float]] = None
    best_ori_err = float("inf")
    best_pos_err = float("inf")

    for i in range(_IK_DIRECT_ATTEMPTS):
        q_rest = list(start_q) if i == 0 else (low + rng.random(len(low)) * (high - low)).tolist()
        try:
            q_seed = solve_tcp_ik(scene, tip_pos_arr, target_orn, q_rest)
            q_goal = refine_joint_config_to_target_pose(
                scene,
                tip_pos_arr,
                tip_rpy_arr,
                q_seed,
                tcp,
                pos_tol_m=pos_tol_m,
                ori_tol_rad=refine_ori_tol_rad,
            )
        except Exception:
            continue
        if not is_state_valid(scene, q_goal):
            continue
        pos_err, ori_err = measure_target_pose_error(scene, tip_pos_arr, tip_rpy_arr, q_goal, tcp)
        if ori_err > ik_ori_tol_rad:
            continue
        # Prefer lower TCP error; break ori ties with smaller ori_err.
        better = pos_err + 1e-9 < best_pos_err or (
            abs(pos_err - best_pos_err) <= 1e-9 and ori_err < best_ori_err
        )
        if better:
            best_pos_err = pos_err
            best_ori_err = ori_err
            best_q = q_goal
        if pos_err <= pos_tol_m and ori_err <= refine_ori_tol_rad:
            _report_tcp_goal(scene, tip_pos, tip_rpy, q_goal, tcp, pos_err=pos_err, ori_err=ori_err)
            return q_goal

    if best_q is not None and best_pos_err <= best_pos_tol_m:
        _report_tcp_goal(
            scene, tip_pos, tip_rpy, best_q, tcp, pos_err=best_pos_err, ori_err=best_ori_err
        )
        return best_q
    return None


def _solve_tcp_joint_fallback(
    scene: PlanningScene,
    tip_pos: Sequence[float],
    tip_rpy: Sequence[float],
    start_q: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_m: float,
    ik_ori_tol_rad: float,
    refine_ori_tol_rad: float,
    ik_pos_tol_m: float | None,
) -> Optional[List[float]]:
    """Fallback when wrist IK misses: few restarts + joint refinement on TCP + orientation."""
    best_pos_tol_m = pos_tol_m * 1.5 if ik_pos_tol_m is None else float(ik_pos_tol_m)
    tip_pos_arr = np.asarray(tip_pos, dtype=np.float64)
    tip_rpy_arr = np.asarray(tip_rpy, dtype=np.float64)
    target_orn = p.getQuaternionFromEuler(list(tip_rpy_arr))

    rng = np.random.default_rng(_IK_REST_SEED)
    low = np.asarray(scene.lower_limits)
    high = np.asarray(scene.upper_limits)
    best_q: Optional[List[float]] = None
    best_ori_err = float("inf")
    best_pos_err = float("inf")

    for i in range(_IK_FALLBACK_ATTEMPTS):
        q_rest = list(start_q) if i == 0 else (low + rng.random(len(low)) * (high - low)).tolist()
        try:
            q_seed = solve_tcp_ik(scene, tip_pos_arr, target_orn, q_rest)
            q_goal = refine_joints_to_tcp_pose(
                scene,
                tip_pos_arr,
                tip_rpy_arr,
                q_seed,
                tcp,
                pos_tol_m=_TCP_POS_TOL_M,
                ori_tol_rad=refine_ori_tol_rad,
            )
        except Exception:
            continue
        if not is_state_valid(scene, q_goal):
            continue
        pos_err, ori_err = measure_target_pose_error(scene, tip_pos_arr, tip_rpy_arr, q_goal, tcp)
        if pos_err <= best_pos_tol_m and ori_err < best_ori_err:
            best_ori_err = ori_err
            best_pos_err = pos_err
            best_q = q_goal

    if best_q is not None and best_pos_err <= best_pos_tol_m and best_ori_err <= ik_ori_tol_rad:
        _report_tcp_goal(scene, tip_pos, tip_rpy, best_q, tcp, pos_err=best_pos_err, ori_err=best_ori_err)
        return best_q
    return None


def solve_goal_from_target_pose(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    start_q: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_m: float = 1e-4,
    ik_ori_tol_rad: float = 0.35,
    refine_ori_tol_rad: float = 0.02,
    ik_pos_tol_m: float | None = None,
) -> Optional[List[float]]:
    if not tcp.has_tcp(scene):
        return solve_goal_from_pose(
            scene,
            target_pos,
            target_rpy,
            start_q,
            pos_tol_m=pos_tol_m,
            ik_ori_tol_rad=ik_ori_tol_rad,
            refine_ori_tol_rad=refine_ori_tol_rad,
            ik_pos_tol_m=ik_pos_tol_m,
        )

    q = _solve_wrist_ik_for_tcp(
        scene,
        target_pos,
        target_rpy,
        start_q,
        tcp,
        pos_tol_m=pos_tol_m,
        ik_ori_tol_rad=ik_ori_tol_rad,
        refine_ori_tol_rad=refine_ori_tol_rad,
        ik_pos_tol_m=ik_pos_tol_m,
    )
    if q is not None:
        return q

    print("[grab_sensor] TCP link IK missed tolerance; retrying with joint refinement seeds.")
    return _solve_tcp_joint_fallback(
        scene,
        target_pos,
        target_rpy,
        start_q,
        tcp,
        pos_tol_m=pos_tol_m,
        ik_ori_tol_rad=ik_ori_tol_rad,
        refine_ori_tol_rad=refine_ori_tol_rad,
        ik_pos_tol_m=ik_pos_tol_m,
    )


def enforce_goal_orientation(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    q: Sequence[float],
    tcp: TcpOffset,
    *,
    pos_tol_m: float,
    max_pos_err_m: float,
    ori_tol_rad: float,
    refine_ori_tol_rad: float,
) -> Optional[List[float]]:
    q_refined = refine_joint_config_to_target_pose(
        scene,
        target_pos,
        target_rpy,
        q,
        tcp,
        pos_tol_m=pos_tol_m,
        ori_tol_rad=refine_ori_tol_rad,
    )
    if not is_state_valid(scene, q_refined):
        return None
    pos_err, ori_err = measure_target_pose_error(scene, target_pos, target_rpy, q_refined, tcp)
    print(
        f"Orientation enforce: pos_err={pos_err * 1000:.3f} mm, "
        f"ori_err={math.degrees(ori_err):.2f} deg "
        f"(limit {math.degrees(ori_tol_rad):.1f} deg)"
    )
    if pos_err > max_pos_err_m or ori_err > ori_tol_rad:
        return None
    return q_refined


def position_error_mm(
    scene: PlanningScene,
    target_pos: Sequence[float],
    q: Sequence[float],
    tcp: TcpOffset,
) -> float:
    return tcp_error_mm(scene, target_pos, q, tcp)


def fk_grasp_points(
    scene: PlanningScene,
    goal_q: Sequence[float],
    tcp: TcpOffset,
) -> tuple[np.ndarray, np.ndarray] | None:
    if scene is None or goal_q is None:
        return None
    set_joint_positions(scene, goal_q)
    return tcp.tip_world(scene), tcp.wrist_world(scene)
