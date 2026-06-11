#!/usr/bin/env python3
"""
Hand-eye camera calibration session.

Reuses manual_control teleop (PyBullet + physical robot over USB) and adds:
  - Live RealSense preview with checkerboard overlay
  - C key capture while moving the arm
  - Eye-to-hand solve for camera_to_robot extrinsics
  - Live convergence plot (distance, orientation, spread)
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_CALIBRATION_DIR = Path(__file__).resolve().parent
_SOFTWARE_DIR = _CALIBRATION_DIR.parents[1]
_MANUAL_CONTROL_DIR = _SOFTWARE_DIR / "manual_control"
for _path in (_SOFTWARE_DIR, _MANUAL_CONTROL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from camera.calibration.calibration_runtime import launch_calibration_session
from camera.calibration.card_config import CalibrationCardConfig
from manual_control.urdf_tools import build_package_map, choose_description, discover_descriptions, materialize_urdf
from robot_config import DEFAULT_ROBOT, default_robot_port, repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate fixed camera pose relative to the robot using a checkerboard on the gripper.",
    )
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT)
    parser.add_argument("--fixed-base", action="store_true", default=True)
    parser.add_argument("--floating-base", action="store_true")
    parser.add_argument("--end-link", type=str, default="")
    parser.add_argument("--linear-speed", type=float, default=0.018)
    parser.add_argument("--angular-speed-deg", type=float, default=12.0)
    parser.add_argument("--joint-vel-limit", type=float, default=0.0)
    parser.add_argument(
        "--sim-hz",
        type=float,
        default=480.0,
        help="PyBullet IK loop rate (default 480; more sim steps per robot waypoint).",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="Sim time vs wall clock (default 1 keeps preview in sync with robot).",
    )
    parser.add_argument("--rotation-weight", type=float, default=8.0)
    parser.add_argument("--joint-force", type=float, default=2000.0)
    parser.add_argument("--ik-damping", type=float, default=0.05)
    parser.add_argument("--port", type=str, default=default_robot_port())
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.4)
    parser.add_argument(
        "--control-dt-ms",
        type=float,
        default=20.0,
        help="Robot & waypoint period in ms (default 20 on Linux; try 15 if serial keeps up).",
    )
    parser.add_argument("--no-home-firmware", action="store_true")
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help="PyBullet preview only; skip robot serial (capture uses sim joint targets).",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("software/camera/calibration/runs"),
        help="Directory for captured samples and camera_to_robot.json",
    )
    parser.add_argument("--camera-serial", type=str, default="")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)

    parser.add_argument("--squares-x", type=int, default=10, help="Checkerboard squares along X (default 10)")
    parser.add_argument("--squares-y", type=int, default=10, help="Checkerboard squares along Y (default 10)")
    parser.add_argument("--square-size-mm", type=float, default=10.128, help="Square width in mm (default 10.128)")
    parser.add_argument(
        "--board-mount",
        type=str,
        choices=("gripper_xy", "gripper_xz"),
        default="gripper_xz",
        help="How the board plane aligns with the end link (default: gripper_xz = pattern axes on gripper X/Z)",
    )
    parser.add_argument(
        "--grasp-point",
        type=str,
        choices=("corner_tip", "edge_center", "far_corner_tip"),
        default="corner_tip",
        help="Where the gripper pinches the board (default: corner_tip = outer corner near pattern origin)",
    )
    parser.add_argument(
        "--gripper-to-board-xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override gripper->checkerboard origin (m). Default comes from --grasp-point and --board-mount.",
    )
    parser.add_argument(
        "--gripper-to-board-rpy-deg",
        type=float,
        nargs=3,
        metavar=("R", "P", "Y"),
        default=None,
        help="Override gripper->board rotation (deg). Default comes from --board-mount.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    fixed_base = args.fixed_base and not args.floating_base
    candidates = discover_descriptions(root)
    selected = choose_description(candidates, args.robot, root)
    package_map = build_package_map(root)
    urdf_path, temp_file = materialize_urdf(selected, package_map)
    print(f"Selected robot: {selected}")
    print(f"Loading URDF: {urdf_path}")
    if selected.suffix.lower() == ".xacro":
        urdf_text = urdf_path.read_text(encoding="utf-8")
        if len(re.findall(r"<joint\b", urdf_text)) == 0:
            print("Warning: resolved URDF has 0 joints. IK control will not work.")

    square_size_m = args.square_size_mm * 1e-3
    card_cfg = CalibrationCardConfig.from_squares(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_size_m=square_size_m,
        board_mount=args.board_mount,
        grasp_point=args.grasp_point,
        gripper_to_board_xyz_m=args.gripper_to_board_xyz,
        gripper_to_board_rpy_deg=args.gripper_to_board_rpy_deg,
    )
    print(
        f"Checkerboard: {args.squares_x}x{args.squares_y} squares, "
        f"{args.square_size_mm} mm, inner corners {card_cfg.pattern_size}"
    )
    print(f"Board mount: {card_cfg.board_mount}")
    print(f"Grasp point: {card_cfg.grasp_point}")
    print(f"Gripper->board rpy (deg): {card_cfg.gripper_to_board_rpy_deg}")
    print(f"Gripper->board xyz (m): {card_cfg.gripper_to_board_xyz_m}")

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    run_dir = out_dir / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    search_paths = {root, selected.parent, urdf_path.parent, *package_map.values()}
    try:
        launch_calibration_session(
            urdf_path=urdf_path,
            fixed_base=fixed_base,
            extra_search_paths=search_paths,
            end_link_hint=args.end_link,
            linear_speed=args.linear_speed,
            angular_speed_deg=args.angular_speed_deg,
            joint_vel_limit=args.joint_vel_limit,
            sim_hz=args.sim_hz,
            speed_factor=args.speed_factor,
            rotation_weight=args.rotation_weight,
            joint_force=args.joint_force,
            ik_damping=args.ik_damping,
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
            control_dt_ms=args.control_dt_ms,
            home_use_firmware=not args.no_home_firmware,
            sim_only=args.sim_only,
            card_cfg=card_cfg,
            out_dir=run_dir,
            camera_serial=args.camera_serial.strip(),
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
