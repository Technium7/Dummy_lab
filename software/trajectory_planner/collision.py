"""PyBullet collision validity for joint-space planning."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pybullet as p

from .scene import PlanningScene, set_joint_positions


def max_penetration_depth(scene: PlanningScene, q: Sequence[float]) -> float:
    """Return worst interpenetration depth (m); 0 means free of penetrating contacts."""
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


def is_state_valid(
    scene: PlanningScene,
    q: Sequence[float],
    penetration_threshold: float = 1e-4,
) -> bool:
    return max_penetration_depth(scene, q) <= penetration_threshold


def collision_cost(scene: PlanningScene, q: Sequence[float], margin: float = 0.01) -> float:
    """CHOMP obstacle cost: hinge on penetration beyond margin."""
    pen = max_penetration_depth(scene, q)
    if pen <= margin:
        return 0.0
    excess = pen - margin
    return excess * excess
