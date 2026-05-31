"""PyBullet collision queries for trajectory planning and analysis.

Environment-only for validity and SCP constraints. Self-collision is reported
separately and does not affect ``valid`` or optimization.

Signed distance (m): ``+clearance`` when separated, ``-penetration`` when in contact.
Penetration uses ``getContactPoints``; clearance uses ``getClosestPoints`` on arm
links only (base link ``-1`` excluded — fixed mount).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pybullet as p

from .scene import PlanningScene, set_joint_positions
from .trajectory_utils import representative_waypoint_map

# --- Defaults -----------------------------------------------------------------

DEFAULT_CLEARANCE_MARGIN = 0.003  # 3 mm; goals often sit near ~4 mm env clearance
DEFAULT_PENETRATION_WEIGHT = 5.0e4
DEFAULT_CLEARANCE_WEIGHT = 80.0
DEFAULT_VALIDITY_THRESHOLD = 1e-4
DEFAULT_PLANNER_EXACT_CLEARANCE = False
DEFAULT_GRADIENT_EPS_RAD = 2e-3

_SKIP_ENV_BASE_LINK = True


# --- Data ---------------------------------------------------------------------


@dataclass(frozen=True)
class CollisionMetrics:
    """Per-configuration collision terms (environment only)."""

    penetration_m: float
    clearance_m: float
    signed_distance_m: float
    valid: bool
    penalty: float
    self_penetration_m: float = 0.0


# --- PyBullet helpers (joints must be set for *_at_pose) ----------------------


def _arm_link_indices(robot_id: int) -> range:
    n = p.getNumJoints(robot_id)
    return range(0, n) if _SKIP_ENV_BASE_LINK else range(-1, n)


def _max_penetration_env_at_pose(scene: PlanningScene) -> float:
    worst = 0.0
    for contact in p.getContactPoints(bodyA=scene.robot_id, bodyB=scene.env_id):
        if _SKIP_ENV_BASE_LINK and int(contact[3]) < 0:
            continue
        depth = float(contact[8])
        if depth < worst:
            worst = depth
    return -worst


def _max_penetration_self_at_pose(scene: PlanningScene) -> float:
    worst = 0.0
    for contact in p.getContactPoints(bodyA=scene.robot_id, bodyB=scene.robot_id):
        link_a = int(contact[3])
        link_b = int(contact[4])
        if link_a < 0 or link_b < 0 or abs(link_a - link_b) <= 1:
            continue
        depth = float(contact[8])
        if depth < worst:
            worst = depth
    return -worst


def _env_clearance_at_pose(
    scene: PlanningScene,
    *,
    exact: bool,
    probe_distance: float = 1.0,
) -> float:
    """Minimum arm-link vs environment separation at the current pose."""
    del exact  # reserved; fast and exact both use per-arm-link env queries
    min_d = float("inf")
    for link_a in _arm_link_indices(scene.robot_id):
        for pt in p.getClosestPoints(
            bodyA=scene.robot_id,
            bodyB=scene.env_id,
            distance=probe_distance,
            linkIndexA=link_a,
            linkIndexB=-1,
        ):
            d = float(pt[8])
            if d < min_d:
                min_d = d
    return probe_distance if min_d == float("inf") else min_d


def _env_signed_distance_at_pose(scene: PlanningScene, *, exact_clearance: bool) -> float:
    pen_env = _max_penetration_env_at_pose(scene)
    if pen_env > 1e-6:
        return -pen_env
    clearance = _env_clearance_at_pose(scene, exact=exact_clearance)
    return max(0.0, clearance)


def _subsample_indices(n: int, stride: int) -> list[int]:
    step = max(1, int(stride))
    indices = list(range(0, n, step))
    if not indices or indices[-1] != n - 1:
        indices.append(n - 1)
    return indices


def _forward_fill(values: np.ndarray, sample_indices: list[int]) -> None:
    sample_set = set(sample_indices)
    last = 0.0
    for i in range(len(values)):
        if i in sample_set:
            last = values[i]
        else:
            values[i] = last


# --- Public query API ---------------------------------------------------------


def max_penetration_depth(
    scene: PlanningScene,
    q: Sequence[float],
    *,
    include_self: bool = False,
) -> float:
    set_joint_positions(scene, q)
    pen_env = _max_penetration_env_at_pose(scene)
    if not include_self:
        return pen_env
    return max(pen_env, _max_penetration_self_at_pose(scene))


def min_clearance_distance(
    scene: PlanningScene,
    q: Sequence[float],
    probe_distance: float = 1.0,
) -> float:
    set_joint_positions(scene, q)
    return _env_clearance_at_pose(scene, exact=True, probe_distance=probe_distance)


def signed_obstacle_distance(
    scene: PlanningScene,
    q: Sequence[float],
    *,
    exact_clearance: bool = True,
) -> float:
    """Signed env distance (m): positive clearance, negative penetration depth."""
    set_joint_positions(scene, q)
    return _env_signed_distance_at_pose(scene, exact_clearance=exact_clearance)


def is_state_valid(
    scene: PlanningScene,
    q: Sequence[float],
    penetration_threshold: float = DEFAULT_VALIDITY_THRESHOLD,
) -> bool:
    return max_penetration_depth(scene, q, include_self=False) <= penetration_threshold


def env_clearance_constraint(signed_distance_m: float, clearance_margin: float) -> float:
    """Inequality constraint; feasible when ``c <= 0``."""
    return float(clearance_margin) - float(signed_distance_m)


def penalty_from_signed_distance(
    signed_distance_m: float,
    clearance_margin: float = DEFAULT_CLEARANCE_MARGIN,
    penetration_weight: float = DEFAULT_PENETRATION_WEIGHT,
    clearance_weight: float = DEFAULT_CLEARANCE_WEIGHT,
) -> float:
    """Soft cost for ``--show-collision`` reporting (not used by SCP planner)."""
    d = signed_distance_m
    if d < 0.0:
        return penetration_weight * d * d
    gap = clearance_margin - d
    if gap > 0.0:
        return clearance_weight * gap * gap
    return 0.0


def evaluate_collision_metrics(
    scene: PlanningScene,
    q: Sequence[float],
    clearance_margin: float = DEFAULT_CLEARANCE_MARGIN,
    penetration_weight: float = DEFAULT_PENETRATION_WEIGHT,
    clearance_weight: float = DEFAULT_CLEARANCE_WEIGHT,
    validity_threshold: float = DEFAULT_VALIDITY_THRESHOLD,
    *,
    exact_clearance: bool = True,
) -> CollisionMetrics:
    set_joint_positions(scene, q)
    pen_env = _max_penetration_env_at_pose(scene)
    pen_self = _max_penetration_self_at_pose(scene)
    if pen_env > 1e-6:
        clearance = 0.0
        signed_d = -pen_env
    else:
        clearance = _env_clearance_at_pose(scene, exact=exact_clearance)
        signed_d = max(0.0, clearance)
    return CollisionMetrics(
        penetration_m=pen_env,
        clearance_m=clearance,
        signed_distance_m=signed_d,
        valid=pen_env <= validity_threshold,
        penalty=penalty_from_signed_distance(
            signed_d, clearance_margin, penetration_weight, clearance_weight
        ),
        self_penetration_m=pen_self,
    )


def evaluate_trajectory_collision(
    scene: PlanningScene,
    trajectory: np.ndarray,
    stride: int = 1,
    exact_clearance: bool = True,
    progress: bool = False,
    on_sample: Optional[Callable[[int, int, CollisionMetrics], None]] = None,
    **penalty_kwargs,
) -> list[CollisionMetrics]:
    """Evaluate collision metrics along a trajectory (deduped + subsampled)."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    n = len(trajectory)
    if n == 0:
        return []
    representatives, rep_for = representative_waypoint_map(trajectory)
    n_unique = len(representatives)
    if n - n_unique > 0 and progress:
        print(
            f"  {n - n_unique} duplicate waypoint(s) skipped (same joint angles as a prior sample)",
            flush=True,
        )

    step = max(1, int(stride))
    eval_indices = representatives[::step]
    if eval_indices[-1] != representatives[-1]:
        eval_indices.append(representatives[-1])

    sparse: dict[int, CollisionMetrics] = {}
    t0 = time.perf_counter()
    for k, i in enumerate(eval_indices):
        metrics = evaluate_collision_metrics(
            scene, trajectory[i], exact_clearance=exact_clearance, **penalty_kwargs
        )
        sparse[i] = metrics
        if on_sample is not None:
            on_sample(i, n, metrics)
        if progress:
            elapsed = time.perf_counter() - t0
            print(
                f"  collision eval {k + 1}/{len(eval_indices)} "
                f"(unique waypoint {i + 1}/{n}, {n_unique} unique) [{elapsed:.1f}s]",
                flush=True,
            )

    return [_lookup_sparse_metrics(sparse, rep_for[i]) for i in range(n)]


def _lookup_sparse_metrics(sparse: dict[int, CollisionMetrics], index: int) -> CollisionMetrics:
    if index in sparse:
        return sparse[index]
    nearest = min(sparse, key=lambda k: abs(k - index))
    return sparse[nearest]


# --- SCP constraint batches ---------------------------------------------------


def _constraint_gradient_at_q(
    scene: PlanningScene,
    q: Sequence[float],
    clearance_margin: float,
    eps: float = DEFAULT_GRADIENT_EPS_RAD,
    exact_clearance: bool = DEFAULT_PLANNER_EXACT_CLEARANCE,
) -> tuple[float, np.ndarray]:
    q_arr = np.asarray(q, dtype=np.float64)
    set_joint_positions(scene, q_arr)
    d0 = _env_signed_distance_at_pose(scene, exact_clearance=exact_clearance)
    c0 = env_clearance_constraint(d0, clearance_margin)
    grad = np.zeros_like(q_arr)
    h = float(eps)
    for i in range(len(q_arr)):
        up = q_arr.copy()
        dn = q_arr.copy()
        up[i] += h
        dn[i] -= h
        set_joint_positions(scene, up)
        d_up = _env_signed_distance_at_pose(scene, exact_clearance=exact_clearance)
        set_joint_positions(scene, dn)
        d_dn = _env_signed_distance_at_pose(scene, exact_clearance=exact_clearance)
        grad[i] = (
            env_clearance_constraint(d_up, clearance_margin)
            - env_clearance_constraint(d_dn, clearance_margin)
        ) / (2.0 * h)
    return c0, grad


def batch_constraint_values(
    scene: PlanningScene,
    states: np.ndarray,
    clearance_margin: float | np.ndarray = DEFAULT_CLEARANCE_MARGIN,
    *,
    exact_clearance: bool = DEFAULT_PLANNER_EXACT_CLEARANCE,
    stride: int = 1,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    n = states.shape[0]
    margins = (
        np.full(n, float(clearance_margin), dtype=np.float64)
        if np.ndim(clearance_margin) == 0
        else np.asarray(clearance_margin, dtype=np.float64)
    )
    indices = _subsample_indices(n, stride)
    values = np.zeros(n, dtype=np.float64)
    for i in indices:
        d = signed_obstacle_distance(scene, states[i], exact_clearance=exact_clearance)
        values[i] = env_clearance_constraint(d, margins[i])
    _forward_fill(values, indices)
    return values


def batch_constraint_linearization(
    scene: PlanningScene,
    states_ref: np.ndarray,
    clearance_margin: float | np.ndarray = DEFAULT_CLEARANCE_MARGIN,
    eps: float = DEFAULT_GRADIENT_EPS_RAD,
    stride: int = 1,
    active_only: bool = True,
    exact_clearance: bool = DEFAULT_PLANNER_EXACT_CLEARANCE,
    skip_tail: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize ``c(q) <= 0`` about ``states_ref`` for SCP."""
    states_ref = np.asarray(states_ref, dtype=np.float64)
    n, dof = states_ref.shape
    margins = (
        np.full(n, float(clearance_margin), dtype=np.float64)
        if np.ndim(clearance_margin) == 0
        else np.asarray(clearance_margin, dtype=np.float64)
    )
    c_ref = np.zeros(n, dtype=np.float64)
    grad_c = np.zeros((n, dof), dtype=np.float64)
    skip_tail = max(0, int(skip_tail))
    free_from = n - skip_tail if skip_tail > 0 else n
    indices = _subsample_indices(n, stride)
    sample_set = set(indices)

    for i in indices:
        if i >= free_from:
            continue
        c_i, g_i = _constraint_gradient_at_q(
            scene, states_ref[i], margins[i], eps, exact_clearance=exact_clearance
        )
        if active_only and c_i <= 0.0:
            continue
        c_ref[i] = c_i
        grad_c[i] = g_i

    last_c = 0.0
    last_g = np.zeros(dof, dtype=np.float64)
    for i in range(n):
        if i >= free_from:
            c_ref[i] = 0.0
            grad_c[i] = 0.0
        elif i in sample_set:
            last_c = c_ref[i]
            last_g = grad_c[i]
        else:
            c_ref[i] = last_c
            grad_c[i] = last_g
    return c_ref, grad_c


def batch_scp_linearized_penalties(
    states: np.ndarray,
    states_ref: np.ndarray,
    c_ref: np.ndarray,
    grad_c: np.ndarray,
    scp_weight: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convex SCP subproblem cost from fixed linearized constraints."""
    states = np.asarray(states, dtype=np.float64)
    states_ref = np.asarray(states_ref, dtype=np.float64)
    n, dof = states.shape
    weights = (
        np.full(n, float(scp_weight), dtype=np.float64)
        if np.ndim(scp_weight) == 0
        else np.asarray(scp_weight, dtype=np.float64)
    )
    costs = np.zeros(n, dtype=np.float64)
    grads = np.zeros((n, dof), dtype=np.float64)
    huber_m = 0.001
    for i in range(n):
        w = float(weights[i])
        if w <= 0.0:
            continue
        c_lin = float(c_ref[i]) + float(grad_c[i] @ (states[i] - states_ref[i]))
        if c_lin <= 0.0:
            continue
        if c_lin < huber_m:
            costs[i] = w * c_lin
            grads[i] = w * grad_c[i]
        else:
            costs[i] = w * c_lin * c_lin
            grads[i] = 2.0 * w * c_lin * grad_c[i]
    return costs, grads
