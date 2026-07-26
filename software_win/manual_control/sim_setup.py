"""PyBullet scene setup shared by manual_control and calibration teleop."""

from __future__ import annotations

import math
from pathlib import Path

import pybullet as p
import pybullet_data

# 180° about world Z — same as trajectory_planner preview (robot faces +X in GUI).
ROBOT_BASE_ORIENTATION = p.getQuaternionFromEuler([0.0, 0.0, math.pi])


def configure_pybullet_gui(*, low_latency: bool = True) -> None:
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    if low_latency:
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)


def refresh_pybullet_display(steps: int = 1) -> None:
    """Force the GUI to draw loaded bodies (needed before the teleop loop starts)."""
    for _ in range(max(1, steps)):
        p.stepSimulation()


def load_ground_plane() -> None:
    try:
        p.loadURDF(str(Path(pybullet_data.getDataPath()) / "plane.urdf"))
    except Exception:
        pass


def load_robot_urdf(urdf_path: Path, *, fixed_base: bool) -> int:
    return p.loadURDF(
        str(urdf_path),
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=ROBOT_BASE_ORIENTATION,
        useFixedBase=fixed_base,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )
