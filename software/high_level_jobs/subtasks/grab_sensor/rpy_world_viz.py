"""Interactive PyBullet explorer for world roll-pitch-yaw (planner / grasp convention).

A sphere at the origin plus RGB axes show how ``p.getQuaternionFromEuler([roll, pitch, yaw])``
orients a body — same as ``targets.grasp_rpy_rad`` in grasp planning.

Run from ``software/``::

    python -m high_level_jobs.subtasks.grab_sensor.rpy_world_viz
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Sequence

_AXIS_LEN_M = 0.12
_WORLD_AXIS_LEN_M = 0.15
_SPHERE_RADIUS_M = 0.025


def draw_world_frame_axes(
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    half_m: float = _WORLD_AXIS_LEN_M,
    width: float = 1.5,
) -> None:
    """Fixed world X/Y/Z through ``origin`` (dim RGB)."""
    import pybullet as p

    ox, oy, oz = (float(origin[i]) for i in range(3))
    colors = ([0.45, 0.25, 0.25], [0.25, 0.45, 0.25], [0.25, 0.25, 0.45])
    for axis_i, c in enumerate(colors):
        start = [ox, oy, oz]
        end = [ox, oy, oz]
        start[axis_i] -= half_m
        end[axis_i] += half_m
        p.addUserDebugLine(start, end, list(c), width, 0)


def draw_world_rpy_axes(
    origin: Sequence[float],
    rpy_rad: Sequence[float],
    *,
    axis_len_m: float = _AXIS_LEN_M,
    width: float = 3.0,
) -> None:
    """Body X/Y/Z after PyBullet intrinsic XYZ Euler (bright RGB)."""
    import pybullet as p

    quat = p.getQuaternionFromEuler([float(v) for v in rpy_rad[:3]])
    o = [float(origin[0]), float(origin[1]), float(origin[2])]
    colors = ([1.0, 0.15, 0.15], [0.15, 1.0, 0.15], [0.2, 0.45, 1.0])
    identity = [0.0, 0.0, 0.0, 1.0]
    for axis_i, color in enumerate(colors):
        local_tip = [0.0, 0.0, 0.0]
        local_tip[axis_i] = axis_len_m
        tip, _ = p.multiplyTransforms(o, quat, local_tip, identity)
        p.addUserDebugLine(o, tip, color, width, 0)


def _rpy_sliders(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[int, int, int]:
    import pybullet as p

    p.removeAllUserParameters()
    roll_id = p.addUserDebugParameter("world roll (deg)", -180.0, 180.0, float(roll_deg))
    pitch_id = p.addUserDebugParameter("world pitch (deg)", -180.0, 180.0, float(pitch_deg))
    yaw_id = p.addUserDebugParameter("world yaw (deg)", -180.0, 180.0, float(yaw_deg))
    return roll_id, pitch_id, yaw_id


def _create_origin_sphere(client_id: int) -> int:
    import pybullet as p

    col = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=_SPHERE_RADIUS_M,
        physicsClientId=client_id,
    )
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=_SPHERE_RADIUS_M,
        rgbaColor=[0.85, 0.85, 0.9, 1.0],
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        physicsClientId=client_id,
    )


def show_world_rpy_explorer(
    *,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> None:
    """PyBullet GUI: sphere at origin + sliders for world RPY (deg)."""
    import pybullet as p

    client = p.connect(p.GUI)
    p.setGravity(0.0, 0.0, 0.0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.45,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=[0.0, 0.0, 0.0],
        physicsClientId=client,
    )

    sphere_id = _create_origin_sphere(client)
    origin = [0.0, 0.0, 0.0]
    roll_id, pitch_id, yaw_id = _rpy_sliders(roll_deg, pitch_deg, yaw_deg)
    last_print = ""

    print(
        "[rpy_world_viz] Sphere at origin. Dim lines = world X/Y/Z. "
        "Bright RGB = body axes from getQuaternionFromEuler(roll,pitch,yaw)."
    )
    print("[rpy_world_viz] Close the PyBullet window to exit.")

    dt = 1.0 / 30.0
    try:
        while p.isConnected(physicsClientId=client):
            roll = float(p.readUserDebugParameter(roll_id))
            pitch = float(p.readUserDebugParameter(pitch_id))
            yaw = float(p.readUserDebugParameter(yaw_id))
            rpy_rad = [math.radians(roll), math.radians(pitch), math.radians(yaw)]
            quat = p.getQuaternionFromEuler(rpy_rad)

            p.resetBasePositionAndOrientation(
                sphere_id, origin, quat, physicsClientId=client
            )

            p.removeAllUserDebugItems(physicsClientId=client)
            draw_world_frame_axes(origin)
            draw_world_rpy_axes(origin, rpy_rad)

            p.addUserDebugText(
                f"RPY deg: {roll:.1f}, {pitch:.1f}, {yaw:.1f}",
                [0.0, 0.0, 0.08],
                textColorRGB=[1.0, 1.0, 0.85],
                textSize=1.2,
                lifeTime=0,
                physicsClientId=client,
            )

            summary = (
                f"roll={roll:.2f} pitch={pitch:.2f} yaw={yaw:.2f}  "
                f"quat={[round(v, 4) for v in quat]}"
            )
            if summary != last_print:
                print(f"[rpy_world_viz] {summary}")
                last_print = summary

            time.sleep(dt)
    finally:
        if p.isConnected(physicsClientId=client):
            p.disconnect(physicsClientId=client)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drag world roll/pitch/yaw sliders; sphere + RGB axes at origin.",
    )
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    show_world_rpy_explorer(
        roll_deg=args.roll_deg,
        pitch_deg=args.pitch_deg,
        yaw_deg=args.yaw_deg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
