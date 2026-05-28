"""PyBullet collision queries for trajectory optimization.

Validity (``is_state_valid``) is binary: penetrating contacts exceed a small threshold.

Optimization penalties use a **continuous signed distance** (meters):
  - separated: minimum clearance from ``getClosestPoints`` (positive)
  - penetrating: worst interpenetration depth from ``getContactPoints`` (reported as negative)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pybullet as p

from .scene import PlanningScene, set_joint_positions


def max_penetration_depth(scene: PlanningScene, q: Sequence[float]) -> float:
    """Worst interpenetration depth (m); 0 means no penetrating contacts."""
    set_joint_positions(scene, q)
    worst = 0.0
    for contact in p.getContactPoints(bodyA=scene.robot_id, bodyB=scene.env_id):
        depth = float(contact[8])
        if depth < worst:
            worst = depth
    for contact in p.getContactPoints(bodyA=scene.robot_id, bodyB=scene.robot_id):
        link_a = int(contact[3])
        link_b = int(contact[4])
        if link_a < 0 or link_b < 0:
            continue
        if abs(link_a - link_b) <= 1:
            continue
        depth = float(contact[8])
        if depth < worst:
            worst = depth
    return -worst


def min_clearance_distance(
    scene: PlanningScene,
    q: Sequence[float],
    probe_distance: float = 1.0,
    include_self: bool = True,
) -> float:
    """Minimum separation distance (m) when not penetrating; large value if no points returned."""
    set_joint_positions(scene, q)
    min_d = float("inf")
    n_links = p.getNumJoints(scene.robot_id)
    link_range = range(-1, n_links)

    def _update(body_b: int, link_b: int = -1) -> None:
        nonlocal min_d
        for link_a in link_range:
            pts = p.getClosestPoints(
                bodyA=scene.robot_id,
                bodyB=body_b,
                distance=probe_distance,
                linkIndexA=link_a,
                linkIndexB=link_b,
            )
            for pt in pts:
                d = float(pt[8])
                if d < min_d:
                    min_d = d

    _update(scene.env_id)
    if include_self:
        for link_b in range(n_links):
            _update(scene.robot_id, link_b)

    if min_d == float("inf"):
        return probe_distance
    return min_d


def signed_obstacle_distance(scene: PlanningScene, q: Sequence[float]) -> float:
    """
    Signed distance to obstacles (m): positive clearance, negative penetration depth.
    """
    pen = max_penetration_depth(scene, q)
    if pen > 1e-6:
        return -pen
    return min_clearance_distance(scene, q)


def is_state_valid(
    scene: PlanningScene,
    q: Sequence[float],
    penetration_threshold: float = 1e-4,
) -> bool:
    """Binary collision check (0 = free, 1 = in collision)."""
    return max_penetration_depth(scene, q) <= penetration_threshold


def collision_penalty(
    scene: PlanningScene,
    q: Sequence[float],
    clearance_margin: float = 0.02,
    penetration_weight: float = 5.0e4,
    clearance_weight: float = 80.0,
) -> float:
    """
    Smooth penalty from signed distance *d* (m):

    - *d < 0* (penetration): ``penetration_weight * d^2``
    - *0 <= d < margin*: ``clearance_weight * (margin - d)^2``
    - *d >= margin*: 0
    """
    d = signed_obstacle_distance(scene, q)
    if d < 0.0:
        return penetration_weight * d * d
    gap = clearance_margin - d
    if gap > 0.0:
        return clearance_weight * gap * gap
    return 0.0


def collision_penalty_gradient(
    scene: PlanningScene,
    q: Sequence[float],
    eps: float = 1e-3,
    **penalty_kwargs,
) -> np.ndarray:
    """Central finite-difference gradient of ``collision_penalty`` w.r.t. joints."""
    q_arr = np.asarray(q, dtype=np.float64)
    grad = np.zeros_like(q_arr)
    for i in range(len(q_arr)):
        up = q_arr.copy()
        dn = q_arr.copy()
        up[i] += eps
        dn[i] -= eps
        c_up = collision_penalty(scene, up, **penalty_kwargs)
        c_dn = collision_penalty(scene, dn, **penalty_kwargs)
        grad[i] = (c_up - c_dn) / (2.0 * eps)
    return grad


def batch_collision_penalty_and_gradient(
    scene: PlanningScene,
    states: np.ndarray,
    eps: float = 1e-3,
    **penalty_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Collision penalty and joint gradient for each row of ``states`` (N, dof)."""
    states = np.asarray(states, dtype=np.float64)
    n, dof = states.shape
    penalties = np.zeros(n, dtype=np.float64)
    grads = np.zeros((n, dof), dtype=np.float64)
    for i in range(n):
        q = states[i]
        penalties[i] = collision_penalty(scene, q, **penalty_kwargs)
        grads[i] = collision_penalty_gradient(scene, q, eps=eps, **penalty_kwargs)
    return penalties, grads
