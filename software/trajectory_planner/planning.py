"""Run discrete iLQR / SCP and verify terminal FK against targets."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .collision import is_state_valid, signed_obstacle_distance
from .config import (
    DEFAULT_CLEARANCE_MARGIN,
    DEFAULT_PLANNER_COLLISION_STRIDE,
    DEFAULT_SCP_CONSTRAINT_WEIGHT,
)
from .lqr_planner import plan_lqr_trajectory
from .scene import (
    PlanningScene,
    get_end_link_pose,
    pose_error_rad,
    set_joint_positions,
)
from .trajectory_utils import pin_trajectory_endpoints


def plan_joint_trajectory(
    scene: PlanningScene,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    enable_collision_penalty: bool = False,
    clearance_margin: float = DEFAULT_CLEARANCE_MARGIN,
    exact_clearance: bool = False,
    collision_stride: int = DEFAULT_PLANNER_COLLISION_STRIDE,
    scp_constraint_weight: float = DEFAULT_SCP_CONSTRAINT_WEIGHT,
) -> Optional[np.ndarray]:
    start_arr = np.asarray(start_q, dtype=np.float64)
    goal_arr = np.asarray(goal_q, dtype=np.float64)

    if enable_collision_penalty:
        d_start = signed_obstacle_distance(scene, start_arr)
        d_goal = signed_obstacle_distance(scene, goal_arr)
        print(
            f"Signed obstacle distance (m): start={d_start:+.4f}, goal={d_goal:+.4f} "
            "(+ clearance, - penetration)"
        )
        print(
            "Planning with SCP: nominal iLQR (no collision), then sequential convex "
            "programming on linearized constraints (see scp_planner.py)"
        )
    else:
        print("Planning with JAX iLQR (terminal goal: joint angles; collision OFF)...")

    dense = plan_lqr_trajectory(
        scene,
        start_arr,
        goal_arr,
        horizon=80,
        n_ilqr_iterations=35,
        enable_collision_penalty=enable_collision_penalty,
        clearance_margin=clearance_margin,
        exact_clearance=exact_clearance,
        collision_stride=collision_stride,
        scp_constraint_weight=scp_constraint_weight,
        verbose=True,
    )
    if dense is None:
        print("LQR planning failed (invalid start/goal or optimizer could not proceed).")
        return None

    dense = pin_trajectory_endpoints(dense, start_arr, goal_arr)
    if enable_collision_penalty:
        collisions = sum(1 for i in range(len(dense)) if not is_state_valid(scene, dense[i]))
        if collisions > 0:
            print(f"Warning: resampled trajectory has {collisions} colliding samples.")

    set_joint_positions(scene, dense[-1])
    pos, rpy = get_end_link_pose(scene)
    pos_err, ori_err = pose_error_rad(pos, rpy, target_pos, target_rpy)
    print(
        f"Planned goal FK: pos_err={pos_err*1000:.3f} mm, "
        f"ori_err={np.degrees(ori_err):.2f} deg"
    )
    print(f"Final trajectory: {len(dense)} samples")
    return dense


def finalize_trajectory_for_cache(
    trajectory: np.ndarray,
    start_q: Sequence[float],
    goal_q: Sequence[float],
) -> np.ndarray:
    """Ensure cached trajectory endpoints match the planned goal joints exactly."""
    return pin_trajectory_endpoints(trajectory, start_q, goal_q)
