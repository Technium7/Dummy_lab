"""Resolve planning goal joint configurations (teleop, IK, or --goal-pose)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pybullet as p

from .collision import is_state_valid
from .scene import (
    PlanningScene,
    get_end_link_pose,
    joint_array_from_transformed_deg,
    measure_pose_error,
    pose_error_rad,
    refine_joint_config_to_pose,
    set_joint_positions,
    solve_ik,
)

_IK_REST_SEED = 42
_IK_MAX_ATTEMPTS = 60


def solve_goal_from_pose(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    start_q: Sequence[float],
    pos_tol_m: float = 1e-4,
) -> Optional[List[float]]:
    """IK + refinement for ``target_end_link``; ignores teleop ``goal_joints_deg``."""
    return solve_goal_configuration(
        scene, target_pos, target_rpy, start_q, goal_joints_deg=None, pos_tol_m=pos_tol_m
    )


def solve_goal_configuration(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    start_q: Sequence[float],
    goal_joints_deg: Optional[Sequence[float]] = None,
    pos_tol_m: float = 1e-4,
) -> Optional[List[float]]:
    """Prefer teleop ``goal_joints_deg``; otherwise Cartesian IK + refinement."""
    if goal_joints_deg is not None and len(goal_joints_deg) >= len(scene.joint_indices):
        q_goal = joint_array_from_transformed_deg(scene, goal_joints_deg)
        if not is_state_valid(scene, q_goal):
            print("Warning: goal_joints_deg is in collision; falling back to Cartesian IK.")
        else:
            pos_err, ori_err = measure_pose_error(scene, target_pos, target_rpy, q_goal)
            print(
                f"Goal from joint angles (teleop): pos_err={pos_err*1000:.3f} mm, "
                f"ori_err={np.degrees(ori_err):.2f} deg"
            )
            return q_goal

    return _solve_ik_goal_configuration(scene, target_pos, target_rpy, start_q, pos_tol_m)


def _solve_ik_goal_configuration(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    start_q: Sequence[float],
    pos_tol_m: float,
) -> Optional[List[float]]:
    target_orn = p.getQuaternionFromEuler(list(target_rpy))
    rest_candidates: List[List[float]] = [list(start_q)]
    rng = np.random.default_rng(_IK_REST_SEED)
    low = np.asarray(scene.lower_limits)
    high = np.asarray(scene.upper_limits)
    best_q: Optional[List[float]] = None
    best_pos_err = float("inf")

    for i in range(_IK_MAX_ATTEMPTS):
        if i > 0:
            rest_candidates.append((low + rng.random(len(low)) * (high - low)).tolist())
        try:
            q_seed = solve_ik(scene, target_pos, target_orn, rest_candidates[-1])
            q_goal = refine_joint_config_to_pose(
                scene, target_pos, target_rpy, q_seed, pos_tol_m=pos_tol_m
            )
        except Exception:
            continue
        if not is_state_valid(scene, q_goal):
            continue
        set_joint_positions(scene, q_goal)
        pos, rpy = get_end_link_pose(scene)
        pos_err, ori_err = pose_error_rad(pos, rpy, target_pos, target_rpy)
        if pos_err < best_pos_err:
            best_pos_err = pos_err
            best_q = q_goal
        if pos_err <= pos_tol_m and ori_err < 0.35:
            print(f"Goal IK: pos_err={pos_err*1000:.3f} mm, ori_err={np.degrees(ori_err):.2f} deg")
            return q_goal

    if best_q is not None:
        set_joint_positions(scene, best_q)
        pos, rpy = get_end_link_pose(scene)
        pos_err, ori_err = pose_error_rad(pos, rpy, target_pos, target_rpy)
        print(
            f"Goal IK (best effort): pos_err={pos_err*1000:.3f} mm, "
            f"ori_err={np.degrees(ori_err):.2f} deg"
        )
        if pos_err <= pos_tol_m * 1.5:
            return best_q
    return None
