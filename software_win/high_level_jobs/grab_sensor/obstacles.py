"""Post cylinders as PyBullet obstacles for SCP collision avoidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .models import GrabSensorConfig, PostObstacleConfig
from .table_position import table_point_to_planner_world_m

_INCH_TO_M = 0.0254


@dataclass
class PostCylinder:
    center_robot_m: np.ndarray  # XYZ base center on table (robot GUI)
    radius_m: float
    height_m: float
    body_id: int = -1


def estimate_post_height_m(
    depth_m: np.ndarray | None,
    uv: tuple[float, float],
    *,
    table_depth_m: float | None,
    cfg: PostObstacleConfig,
) -> float:
    """Height above table from depth at keypoint vs estimated table depth."""
    min_h = float(cfg.min_height_mm) * 1e-3
    max_h = float(cfg.max_height_mm) * 1e-3
    if depth_m is None or table_depth_m is None:
        return min_h
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = depth_m.shape[:2]
    r = int(cfg.depth_patch_radius_px)
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[(patch > 0.05) & np.isfinite(patch)]
    if valid.size == 0:
        return min_h
    top_depth = float(np.median(valid))
    # Camera looking down: smaller depth = taller object above table
    height = float(table_depth_m) - top_depth
    return float(np.clip(height, min_h, max_h))


def estimate_table_depth_m(
    depth_m: np.ndarray | None,
    sample_uvs: Sequence[tuple[float, float]],
) -> float | None:
    if depth_m is None or not sample_uvs:
        return None
    vals: list[float] = []
    h, w = depth_m.shape[:2]
    for u, v in sample_uvs:
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            d = float(depth_m[vi, ui])
            if d > 0.05 and np.isfinite(d):
                vals.append(d)
    if not vals:
        return None
    return float(np.median(vals))


def build_post_cylinders(
    cfg: GrabSensorConfig,
    posts_debug: Sequence[dict[str, Any]],
    *,
    depth_m: np.ndarray | None = None,
    table_depth_m: float | None = None,
) -> list[PostCylinder]:
    radius = float(cfg.post_obstacle.diameter_in) * _INCH_TO_M * 0.5
    out: list[PostCylinder] = []
    for post in posts_debug:
        table_xyz = post.get("table_xyz_m")
        uv = post.get("uv")
        if not table_xyz or uv is None:
            continue
        height = estimate_post_height_m(
            depth_m,
            (float(uv[0]), float(uv[1])),
            table_depth_m=table_depth_m,
            cfg=cfg.post_obstacle,
        )
        # Cylinder sits on table; base at z≈0 table → robot GUI
        base_table = (float(table_xyz[0]), float(table_xyz[1]), 0.0)
        base_robot = table_point_to_planner_world_m(base_table, cfg.robot_in_table_frame)
        # Center of cylinder volume at half height
        center = base_robot.copy()
        center[2] += height * 0.5
        out.append(PostCylinder(center_robot_m=center, radius_m=radius, height_m=height))
    return out


def add_post_cylinders_to_scene(scene, cylinders: Sequence[PostCylinder]) -> list[int]:
    """Create static cylinder collision bodies in an existing planning scene."""
    import pybullet as p

    ids: list[int] = []
    for cyl in cylinders:
        col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=float(cyl.radius_m),
            height=float(cyl.height_m),
            physicsClientId=scene.client_id,
        )
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=float(cyl.radius_m),
            length=float(cyl.height_m),
            rgbaColor=[0.2, 0.7, 0.3, 0.55],
            physicsClientId=scene.client_id,
        )
        body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[float(v) for v in cyl.center_robot_m],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=scene.client_id,
        )
        cyl.body_id = int(body)
        ids.append(int(body))
    return ids


def inject_posts_into_scene(
    scene,
    cfg: GrabSensorConfig,
    targets_debug: dict[str, Any],
    *,
    depth_m: np.ndarray | None = None,
) -> list[PostCylinder]:
    posts = targets_debug.get("posts") or []
    if not posts:
        return []
    # Approximate table depth from sensor UV if available
    table_depth = None
    sensor_uv = targets_debug.get("sensor_uv")
    if sensor_uv is not None and depth_m is not None:
        table_depth = estimate_table_depth_m(depth_m, [(float(sensor_uv[0]), float(sensor_uv[1]))])
        # Prefer a nearby table reading: use max depth near sensor as table (object is closer)
        if table_depth is not None:
            # object top is closer; table slightly farther — bump table depth estimate
            table_depth = table_depth + float(cfg.grasp_motion.grasp_height_above_table_mm) * 1e-3 * 0.3

    cylinders = build_post_cylinders(cfg, posts, depth_m=depth_m, table_depth_m=table_depth)
    add_post_cylinders_to_scene(scene, cylinders)
    print(f"[grab_sensor] added {len(cylinders)} post cylinder obstacle(s) for SCP")
    return cylinders
