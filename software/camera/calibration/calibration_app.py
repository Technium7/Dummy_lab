#!/usr/bin/env python3
"""
Hand-eye camera calibration session (reference / fallback).

Primary table-grid workflow: ``camera.calibration_table.table_calib_app``.

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
from camera.calibration.hand_eye import CameraMountConstraint
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
    parser.add_argument("--linear-speed", type=float, default=0.015)
    parser.add_argument("--angular-speed-deg", type=float, default=10.0)
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
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--control-dt-ms",
        type=float,
        default=70.0,
        help="Robot & waypoint period in ms (default 70; try 20–15 if serial keeps up).",
    )
    parser.add_argument(
        "--home-firmware",
        action="store_true",
        help="T also sends firmware !HOME (blocks serial ~3s; default is stream-only home).",
    )
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help="PyBullet preview only; skip robot serial (capture uses sim joint targets).",
    )
    parser.add_argument(
        "--debug-serial",
        action="store_true",
        help="Log every serial command sent to and response drained from the robot.",
    )
    parser.add_argument(
        "--joint-current-limit-a",
        type=float,
        default=0.0,
        help="Set motor current limit (A) for J1-J6 at connect via #I_LIMIT_J (default 0=skip).",
    )
    parser.add_argument(
        "--teleop-only",
        action="store_true",
        help="Run teleop without RealSense (PyBullet + robot only; no capture/solve).",
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
        choices=("gripper_xy", "gripper_xz", "gripper_yz"),
        default="gripper_yz",
        help=(
            "How the board plane aligns with the end link "
            "(default: gripper_yz = pattern on gripper Y/Z, PyBullet green+blue)"
        ),
    )
    parser.add_argument(
        "--grasp-point",
        type=str,
        choices=("corner_tip", "origin_corner_tip", "link_side_edge", "edge_center", "far_corner_tip"),
        default="corner_tip",
        help=(
            "Where the gripper pinches the board "
            "(default: corner_tip = outer corner on the wrist-facing side)"
        ),
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
    parser.add_argument(
        "--end-link-tcp-xyz-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Custom wrist→TCP vector in the end-link frame (m).",
    )
    parser.add_argument(
        "--end-link-tcp-distance-m",
        type=float,
        default=None,
        help="Pinch distance (m) along link6 −Y (default 0.065).",
    )

    parser.add_argument(
        "--camera-constrained-mount",
        action="store_true",
        help=(
            "Fix camera roll/pitch and fit only translation + base-Z yaw "
            "(legacy; default is full 6-DOF rotation fit)."
        ),
    )
    parser.add_argument(
        "--camera-fixed-roll-deg",
        type=float,
        default=-45.0,
        help="Fixed roll about base X when using --camera-constrained-mount (deg).",
    )
    parser.add_argument(
        "--camera-fixed-pitch-deg",
        type=float,
        default=0.0,
        help="Fixed pitch about base Y when using --camera-constrained-mount (deg).",
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
    card_cfg = CalibrationCardConfig.from_cli_or_json(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_size_m=square_size_m,
        board_mount=args.board_mount,
        grasp_point=args.grasp_point,
        end_link_to_tcp_xyz_m=args.end_link_tcp_xyz_m,
        end_link_to_tcp_distance_m=args.end_link_tcp_distance_m,
        gripper_to_board_xyz_m=args.gripper_to_board_xyz,
        gripper_to_board_rpy_deg=args.gripper_to_board_rpy_deg,
    )
    print(
        f"Checkerboard: {args.squares_x}x{args.squares_y} squares, "
        f"{args.square_size_mm} mm, inner corners {card_cfg.pattern_size}"
    )
    print(f"Board mount: {card_cfg.board_mount}")
    print(f"Grasp point: {card_cfg.grasp_point}")
    print(f"End-link->TCP xyz (m): {card_cfg.end_link_to_tcp_xyz_m}")
    print(f"TCP->board rpy (deg): {card_cfg.gripper_to_board_rpy_deg}")
    print(f"TCP->board xyz (m): {card_cfg.gripper_to_board_xyz_m}")

    camera_mount = CameraMountConstraint(
        fixed_roll_deg=args.camera_fixed_roll_deg,
        fixed_pitch_deg=args.camera_fixed_pitch_deg,
        fit_full_rotation=not args.camera_constrained_mount,
    )
    if camera_mount.fit_full_rotation:
        print("Camera solve: full 6-DOF rotation + translation (no roll/pitch constraint).")
    else:
        print(
            f"Camera solve: roll={camera_mount.fixed_roll_deg:g}°, "
            f"pitch={camera_mount.fixed_pitch_deg:g}° fixed; "
            "fit translation + base-Z yaw only."
        )

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
            home_use_firmware=args.home_firmware,
            sim_only=args.sim_only,
            debug_serial=args.debug_serial,
            joint_current_limit_a=args.joint_current_limit_a,
            teleop_only=args.teleop_only,
            card_cfg=card_cfg,
            out_dir=run_dir,
            camera_serial=args.camera_serial.strip(),
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            camera_mount=camera_mount,
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
