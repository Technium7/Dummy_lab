"""Run discrete iLQR / SCP and verify terminal FK against targets."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

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

_BASE_JOINT_IDX = 0
_DEFAULT_HORIZON = 80


def _mid_q_j1_first(start: np.ndarray, goal: np.ndarray, base_idx: int = _BASE_JOINT_IDX) -> np.ndarray:
    """Waypoint after finishing J1 only (other joints still at start)."""
    mid = np.asarray(start, dtype=np.float64).copy()
    mid[base_idx] = float(goal[base_idx])
    return mid


def _mid_q_j1_last(start: np.ndarray, goal: np.ndarray, base_idx: int = _BASE_JOINT_IDX) -> np.ndarray:
    """Waypoint after finishing non-J1 joints (J1 still at start)."""
    mid = np.asarray(goal, dtype=np.float64).copy()
    mid[base_idx] = float(start[base_idx])
    return mid


def _phase_horizons(start: np.ndarray, goal: np.ndarray, total: int, base_idx: int) -> tuple[int, int]:
    """Split horizon by |ΔJ1| vs other-joint motion (min 8 samples each)."""
    total = max(16, int(total))
    d_j1 = abs(float(goal[base_idx] - start[base_idx]))
    rest = np.delete(goal - start, base_idx)
    d_rest = float(np.linalg.norm(rest))
    s = d_j1 + d_rest
    if s < 1e-9:
        h1 = total // 2
    else:
        h1 = int(round(total * (d_j1 / s)))
    h1 = int(np.clip(h1, 8, total - 8))
    return h1, total - h1


def _run_lqr(
    scene: PlanningScene,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int,
    enable_collision_penalty: bool,
    clearance_margin: float,
    exact_clearance: bool,
    collision_stride: int,
    scp_constraint_weight: float,
    joint_guess_mode: str | None,
) -> Optional[np.ndarray]:
    return plan_lqr_trajectory(
        scene,
        start,
        goal,
        horizon=horizon,
        n_ilqr_iterations=35,
        enable_collision_penalty=enable_collision_penalty,
        clearance_margin=clearance_margin,
        exact_clearance=exact_clearance,
        collision_stride=collision_stride,
        scp_constraint_weight=scp_constraint_weight,
        verbose=True,
        joint_guess_mode=joint_guess_mode,
    )


def _report_terminal_fk(
    scene: PlanningScene,
    dense: np.ndarray,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    tcp_position_error_m: Callable[[PlanningScene, Sequence[float], Sequence[float]], float]
    | None,
) -> None:
    set_joint_positions(scene, dense[-1])
    if tcp_position_error_m is not None:
        pos_err = float(tcp_position_error_m(scene, dense[-1], target_pos))
        _, rpy = get_end_link_pose(scene)
        _, ori_err = pose_error_rad([0.0, 0.0, 0.0], rpy, [0.0, 0.0, 0.0], target_rpy)
        print(
            f"Planned goal TCP FK: tip_err={pos_err * 1000:.3f} mm, "
            f"ori_err={np.degrees(ori_err):.2f} deg"
        )
    else:
        pos, rpy = get_end_link_pose(scene)
        pos_err, ori_err = pose_error_rad(pos, rpy, target_pos, target_rpy)
        print(
            f"Planned goal FK: pos_err={pos_err * 1000:.3f} mm, "
            f"ori_err={np.degrees(ori_err):.2f} deg"
        )


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
    tcp_position_error_m: Callable[[PlanningScene, Sequence[float], Sequence[float]], float]
    | None = None,
    joint_guess_mode: str | None = None,
) -> Optional[np.ndarray]:
    start_arr = np.asarray(start_q, dtype=np.float64)
    goal_arr = np.asarray(goal_q, dtype=np.float64)
    mode = (joint_guess_mode or "linear").strip().lower()
    if mode in ("", "none"):
        mode = "linear"

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

    # Strict sequencing: finish J1 alone first (early) or alone last (late).
    # Two concatenated plans beat a single phased guess (iLQR otherwise blends joints).
    if mode in ("early", "late"):
        bi = _BASE_JOINT_IDX
        d_j1 = abs(float(goal_arr[bi] - start_arr[bi]))
        if d_j1 < 1e-4:
            print(f"Joint guess mode: {mode} (ΔJ1≈0 → single plan)")
            mode = "linear"
        else:
            mid = (
                _mid_q_j1_first(start_arr, goal_arr, bi)
                if mode == "early"
                else _mid_q_j1_last(start_arr, goal_arr, bi)
            )
            if enable_collision_penalty and not is_state_valid(scene, mid):
                print(
                    f"Joint guess mode: {mode} (J1 waypoint in collision → "
                    "falling back to single phased plan)"
                )
            else:
                h_j1, h_rest = _phase_horizons(start_arr, goal_arr, _DEFAULT_HORIZON, bi)
                if mode == "early":
                    print(
                        f"Joint sequencing: early — J1 first ({h_j1} samples), "
                        f"then other joints ({h_rest} samples)"
                    )
                    leg_a = (start_arr, mid, h_j1)
                    leg_b = (mid, goal_arr, h_rest)
                else:
                    print(
                        f"Joint sequencing: late — other joints first ({h_rest} samples), "
                        f"then J1 ({h_j1} samples)"
                    )
                    leg_a = (start_arr, mid, h_rest)
                    leg_b = (mid, goal_arr, h_j1)

                traj_a = _run_lqr(
                    scene,
                    leg_a[0],
                    leg_a[1],
                    horizon=leg_a[2],
                    enable_collision_penalty=enable_collision_penalty,
                    clearance_margin=clearance_margin,
                    exact_clearance=exact_clearance,
                    collision_stride=collision_stride,
                    scp_constraint_weight=scp_constraint_weight,
                    joint_guess_mode="linear",
                )
                if traj_a is None:
                    print("LQR planning failed on J1 sequencing leg A.")
                    return None
                traj_b = _run_lqr(
                    scene,
                    leg_b[0],
                    leg_b[1],
                    horizon=leg_b[2],
                    enable_collision_penalty=enable_collision_penalty,
                    clearance_margin=clearance_margin,
                    exact_clearance=exact_clearance,
                    collision_stride=collision_stride,
                    scp_constraint_weight=scp_constraint_weight,
                    joint_guess_mode="linear",
                )
                if traj_b is None:
                    print("LQR planning failed on J1 sequencing leg B.")
                    return None

                traj_a = pin_trajectory_endpoints(traj_a, leg_a[0], leg_a[1])
                traj_b = pin_trajectory_endpoints(traj_b, leg_b[0], leg_b[1])
                dense = np.vstack([traj_a[:-1], traj_b])
                dense = pin_trajectory_endpoints(dense, start_arr, goal_arr)

                if enable_collision_penalty:
                    collisions = sum(
                        1 for i in range(len(dense)) if not is_state_valid(scene, dense[i])
                    )
                    if collisions > 0:
                        print(f"Warning: resampled trajectory has {collisions} colliding samples.")

                _report_terminal_fk(
                    scene, dense, target_pos, target_rpy, tcp_position_error_m
                )
                print(f"Final trajectory: {len(dense)} samples (J1 sequenced)")
                return dense

    if joint_guess_mode and mode == "linear":
        print("Joint guess mode: linear")
    elif joint_guess_mode and mode not in ("early", "late"):
        print(f"Joint guess mode: {mode}")

    dense = _run_lqr(
        scene,
        start_arr,
        goal_arr,
        horizon=_DEFAULT_HORIZON,
        enable_collision_penalty=enable_collision_penalty,
        clearance_margin=clearance_margin,
        exact_clearance=exact_clearance,
        collision_stride=collision_stride,
        scp_constraint_weight=scp_constraint_weight,
        joint_guess_mode=None if mode == "linear" else mode,
    )
    if dense is None:
        print("LQR planning failed (invalid start/goal or optimizer could not proceed).")
        return None

    dense = pin_trajectory_endpoints(dense, start_arr, goal_arr)
    if enable_collision_penalty:
        collisions = sum(1 for i in range(len(dense)) if not is_state_valid(scene, dense[i]))
        if collisions > 0:
            print(f"Warning: resampled trajectory has {collisions} colliding samples.")

    _report_terminal_fk(scene, dense, target_pos, target_rpy, tcp_position_error_m)
    print(f"Final trajectory: {len(dense)} samples")
    return dense


def finalize_trajectory_for_cache(
    trajectory: np.ndarray,
    start_q: Sequence[float],
    goal_q: Sequence[float],
) -> np.ndarray:
    """Ensure cached trajectory endpoints match the planned goal joints exactly."""
    return pin_trajectory_endpoints(trajectory, start_q, goal_q)
