"""Interactive PyBullet explorer for world roll-pitch-yaw on link6 (grasp planner convention).

Loads the grasp URDF, shows the vision **grasp target** (yellow cross + RGB axes), and
IK-moves the arm toward that pose as you drag world roll/pitch/yaw sliders.

Run from ``software/``::

    python -m high_level_jobs.subtasks.grab_sensor.rpy_world_viz
    python -m high_level_jobs.subtasks.grab_sensor.rpy_world_viz --offline
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from manual_control.mapping import HOME_JOINTS_DEG  # noqa: E402
from trajectory_planner.scene import disconnect_planning_scene, set_joint_positions  # noqa: E402

from .config_loader import default_config_path, load_grab_sensor_config  # noqa: E402
from .grasp import grasp_abc_to_world_rpy_deg, load_grasp_targets_from_vision  # noqa: E402
from .models import GrabSensorConfig  # noqa: E402
from .plan_motion import build_scene, start_q_from_robot_joints_deg  # noqa: E402
from .tcp_ik import TcpOffset, grasp_ik_goal_from_home, link_pose_from_q, print_grasp_ik_debug, tcp_error_mm  # noqa: E402

_AXIS_LEN_M = 0.08
_WORLD_AXIS_LEN_M = 0.06
_CROSS_HALF_M = 0.018


def draw_world_frame_axes(
    origin: Sequence[float],
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
    colors: Sequence[Sequence[float]] | None = None,
) -> None:
    """Body X/Y/Z after PyBullet intrinsic XYZ Euler (bright RGB by default)."""
    import pybullet as p

    quat = p.getQuaternionFromEuler([float(v) for v in rpy_rad[:3]])
    o = [float(origin[0]), float(origin[1]), float(origin[2])]
    axis_colors = colors or ([1.0, 0.15, 0.15], [0.15, 1.0, 0.15], [0.2, 0.45, 1.0])
    identity = [0.0, 0.0, 0.0, 1.0]
    for axis_i, color in enumerate(axis_colors):
        local_tip = [0.0, 0.0, 0.0]
        local_tip[axis_i] = axis_len_m
        tip, _ = p.multiplyTransforms(o, quat, local_tip, identity)
        p.addUserDebugLine(o, tip, [float(v) for v in color[:3]], width, 0)


def draw_world_cross(
    pos: Sequence[float],
    *,
    label: str,
    color: Sequence[float],
    half_m: float = _CROSS_HALF_M,
) -> None:
    import pybullet as p

    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    c = [float(v) for v in color[:3]]
    p.addUserDebugLine([x - half_m, y, z], [x + half_m, y, z], c, 3.0, 0)
    p.addUserDebugLine([x, y - half_m, z], [x, y + half_m, z], c, 3.0, 0)
    p.addUserDebugLine([x, y, z - half_m], [x, y, z + half_m], c, 2.0, 0)
    p.addUserDebugText(label, [x, y, z + 0.015], textColorRGB=c, textSize=1.1, lifeTime=0)


def _abc_sliders(a_deg: float, b_deg: float, c_deg: float) -> tuple[int, int, int]:
    import pybullet as p

    p.removeAllUserParameters()
    a_id = p.addUserDebugParameter("input a (deg)", -180.0, 180.0, float(a_deg))
    b_id = p.addUserDebugParameter("input b (deg)", -180.0, 180.0, float(b_deg))
    c_id = p.addUserDebugParameter("input c (deg)", -180.0, 180.0, float(c_deg))
    return a_id, b_id, c_id


def _frame_on_grasp(scene, grasp_pos: Sequence[float]) -> None:
    import pybullet as p

    gx, gy, gz = (float(grasp_pos[i]) for i in range(3))
    p.resetDebugVisualizerCamera(
        cameraDistance=0.85,
        cameraYaw=55,
        cameraPitch=-28,
        cameraTargetPosition=[gx * 0.45, gy * 0.45, gz * 0.6 + 0.05],
        physicsClientId=scene.client_id,
    )


def show_world_rpy_explorer(
    cfg: GrabSensorConfig,
    *,
    abc_deg: Sequence[float] | None = None,
    start_joints_deg: Sequence[float] | None = None,
    offline: bool = True,
    grasp_pos_m: Sequence[float] | None = None,
) -> None:
    """PyBullet GUI: grasp target + link6/TCP IK from [a,b,c] input sliders."""
    import pybullet as p

    targets = load_grasp_targets_from_vision(cfg, offline=offline, grasp_pos_m=grasp_pos_m)
    grasp_pos = np.asarray(targets.grasp_pos_m, dtype=np.float64)[:3]

    abc0 = list(cfg.fixed_grasp_abc_input_deg)
    if abc_deg is not None:
        abc0 = [float(v) for v in abc_deg[:3]]

    scene, temp_file = build_scene(cfg, use_gui=True)
    tcp = TcpOffset()
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=scene.client_id)
        _frame_on_grasp(scene, grasp_pos)

        home_q = start_q_from_robot_joints_deg(scene, start_joints_deg or HOME_JOINTS_DEG)
        set_joint_positions(scene, home_q)

        a_id, b_id, c_id = _abc_sliders(abc0[0], abc0[1], abc0[2])
        last_debug_key = None

        print(
            "[rpy_world_viz] Yellow cross = grasp TCP target; bright RGB = world RPY from [a,b,c]. "
            "Close the PyBullet window to exit."
        )
        print(f"[rpy_world_viz] Grasp target m: {grasp_pos.round(4).tolist()}")
        if heading := targets.debug.get("grasp_heading_deg"):
            print(f"[rpy_world_viz] atan2(x,y) deg: {heading:.2f}")
        print(
            f"[rpy_world_viz] Config abc deg: {abc0}  "
            f"planned world RPY deg: {targets.debug.get('grasp_rpy_world_deg')}"
        )

        dt = 1.0 / 30.0
        while p.isConnected(physicsClientId=scene.client_id):
            a = float(p.readUserDebugParameter(a_id))
            b = float(p.readUserDebugParameter(b_id))
            c = float(p.readUserDebugParameter(c_id))
            roll, pitch, yaw = grasp_abc_to_world_rpy_deg(cfg, [a, b, c], grasp_pos)
            targets.grasp_rpy_rad = np.radians(np.asarray([roll, pitch, yaw], dtype=np.float64))
            rpy_rad = [float(v) for v in targets.grasp_rpy_rad]

            q_goal = grasp_ik_goal_from_home(scene, targets, home_q, tcp) or list(home_q)
            set_joint_positions(scene, q_goal)

            debug_key = (
                round(a, 3),
                round(b, 3),
                round(c, 3),
                tuple(round(float(v), 6) for v in q_goal),
            )
            if debug_key != last_debug_key:
                print_grasp_ik_debug(
                    scene,
                    targets,
                    q_goal,
                    label="rpy_world_viz",
                    warm_start_q=home_q,
                    tcp=tcp,
                )
                last_debug_key = debug_key

            wrist_pos, fk_rpy, _ = link_pose_from_q(scene, q_goal)
            tip_pos = tcp.tip_world(scene)
            tip_err_mm = tcp_error_mm(scene, targets.grasp_pos_m, q_goal, tcp) if tcp.has_tcp(scene) else float("nan")
            anchor = [float(wrist_pos[0]), float(wrist_pos[1]), float(wrist_pos[2])]
            grasp = [float(grasp_pos[0]), float(grasp_pos[1]), float(grasp_pos[2])]

            p.removeAllUserDebugItems(physicsClientId=scene.client_id)

            draw_world_cross(grasp, label="grasp target", color=[1.0, 1.0, 0.0])
            draw_world_frame_axes(grasp, half_m=_WORLD_AXIS_LEN_M * 0.9)
            draw_world_rpy_axes(grasp, rpy_rad)

            draw_world_frame_axes(anchor)
            draw_world_rpy_axes(
                anchor,
                fk_rpy,
                axis_len_m=_AXIS_LEN_M * 0.85,
                width=2.0,
                colors=([0.9, 0.9, 0.9], [0.75, 0.75, 0.75], [0.6, 0.6, 0.6]),
            )

            if tcp.has_tcp(scene):
                draw_world_cross(tip_pos, label="TCP FK", color=[1.0, 0.35, 0.35], half_m=_CROSS_HALF_M * 0.7)
                p.addUserDebugLine(anchor, grasp, [0.6, 0.6, 0.6], 1.5, 0)

            p.addUserDebugText(
                f"abc [{a:.1f}, {b:.1f}, {c:.1f}] -> world RPY deg: {roll:.1f}, {pitch:.1f}, {yaw:.1f}",
                [grasp[0], grasp[1], grasp[2] + 0.06],
                textColorRGB=[1.0, 1.0, 0.85],
                textSize=1.05,
                lifeTime=0,
                physicsClientId=scene.client_id,
            )
            fk_deg = [math.degrees(v) for v in fk_rpy]
            p.addUserDebugText(
                f"link6 FK deg: {fk_deg[0]:.1f}, {fk_deg[1]:.1f}, {fk_deg[2]:.1f}  "
                f"TCP err {tip_err_mm:.1f} mm",
                [grasp[0], grasp[1], grasp[2] + 0.04],
                textColorRGB=[0.85, 0.85, 1.0],
                textSize=0.95,
                lifeTime=0,
                physicsClientId=scene.client_id,
            )

            time.sleep(dt)
    finally:
        disconnect_planning_scene(scene)
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grasp target + link6/TCP; drag [a,b,c] input sliders.",
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use saved debug capture for grasp target (default when neither --live nor --offline).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live camera for grasp target.",
    )
    parser.add_argument(
        "--grasp-pos-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override grasp TCP position (planner world m).",
    )
    parser.add_argument(
        "--abc-deg",
        type=float,
        nargs=3,
        metavar=("A", "B", "C"),
        default=None,
        help="Initial taught input [a, b, c] (deg).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.live and args.offline:
        parser.error("Use only one of --offline or --live.")

    cfg = load_grab_sensor_config(args.config)
    show_world_rpy_explorer(
        cfg,
        abc_deg=args.abc_deg,
        offline=not args.live,
        grasp_pos_m=args.grasp_pos_m,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
