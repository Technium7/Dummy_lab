from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SOFTWARE_DIR = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from high_level_jobs.subtasks.grab_sensor.config_loader import (
    apply_runtime_calibrations,
    default_config_path,
    load_grab_sensor_config,
)
from high_level_jobs.subtasks.grab_sensor.pipeline import run_grab_sensor_pipeline

_OFFLINE_COLOR = (
    Path(__file__).resolve().parent / "debug" / "grab_sensor.color.png"
)
_OFFLINE_DEPTH = (
    Path(__file__).resolve().parent / "debug" / "grab_sensor.depth.npy"
)
from robot_ascii_interface import RobotAsciiInterface
from robot_config import default_robot_port


def main() -> int:
    p = argparse.ArgumentParser(description="Detect sensor with D415, plan grasp, optionally execute.")
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument(
        "--execute",
        action="store_true",
        help="Stream plan to robot (open at pre-grasp, close at grasp, lift/HOME)",
    )
    p.add_argument("--port", default=default_robot_port())
    p.add_argument("--baudrate", type=int, default=int(os.environ.get("ROBOT_BAUDRATE", "115200")))
    p.add_argument("--color-file", type=Path, default=None, help="Offline test from saved capture")
    p.add_argument("--depth-file", type=Path, default=None)
    p.add_argument(
        "--offline",
        action="store_true",
        help=f"Use saved debug capture ({_OFFLINE_COLOR.name}) instead of live D415",
    )
    p.add_argument("--detect-only", action="store_true", help="Vision + print targets; no planning")
    p.add_argument(
        "--no-fallback-offline",
        action="store_true",
        help="Do not use saved debug/grab_sensor.color.png if live camera fails",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Save grasp_overlay.png (camera image with grasp/base markers; not PyBullet)",
    )
    p.add_argument("--visualize-out", type=Path, default=None, help="Output path for overlay PNG")
    p.add_argument(
        "--plan-gui",
        "--gui",
        action="store_true",
        dest="plan_gui",
        help="After planning, open PyBullet GUI and play back the arm path",
    )
    p.add_argument(
        "--ik-gui",
        action="store_true",
        help="Step 1: show IK goal only — arm at final joints, two yellow crosses, print positions",
    )
    p.add_argument(
        "--rpy-viz",
        action="store_true",
        help="Interactive world RPY sliders (PyBullet); see rpy_world_viz module",
    )
    args = p.parse_args()

    if args.rpy_viz:
        from high_level_jobs.subtasks.grab_sensor.rpy_world_viz import main as rpy_viz_main

        rpy_argv: list[str] = []
        if args.offline:
            rpy_argv.append("--offline")
        return rpy_viz_main(rpy_argv if rpy_argv else None)

    color_file = args.color_file
    depth_file = args.depth_file
    if args.offline:
        if color_file is None:
            color_file = _OFFLINE_COLOR
        if depth_file is None:
            depth_file = _OFFLINE_DEPTH
        if not color_file.is_file():
            print(f"Offline color capture not found: {color_file}", file=sys.stderr)
            return 1
        try:
            from high_level_jobs.subtasks.grab_sensor.detection.capture import validate_color_image_file

            validate_color_image_file(color_file)
        except OSError as exc:
            print(f"Offline color capture is unreadable: {exc}", file=sys.stderr)
            print(
                "Restore from git (git checkout -- <path>) or recapture with a live run.",
                file=sys.stderr,
            )
            return 1

    cfg = apply_runtime_calibrations(load_grab_sensor_config(args.config), {})
    params = {
        "execute": args.execute,
        "color_file": str(color_file) if color_file else None,
        "depth_file": str(depth_file) if depth_file else None,
        "offline": bool(args.offline),
        "visualize": args.visualize or args.detect_only,
        "visualize_out": str(args.visualize_out) if args.visualize_out else None,
        "plan_gui": args.plan_gui,
        "ik_gui": args.ik_gui,
        "allow_offline_fallback": not args.no_fallback_offline,
    }

    robot = None
    if args.execute:
        robot = RobotAsciiInterface(port=args.port, baudrate=args.baudrate)
        if not robot.connect():
            print(f"Failed to connect to {args.port}", file=sys.stderr)
            return 1

    if args.detect_only:
        from high_level_jobs.subtasks.grab_sensor.pipeline import run_detection

        _, targets, _, _ = run_detection(cfg, params)
        print("Grasp target (robot m):", targets.grasp_pos_m.round(4).tolist())
        print("Grasp rpy (robot deg):", targets.debug.get("grasp_rpy_robot_deg"))
        return 0

    run_grab_sensor_pipeline(cfg, params, robot=robot)
    if robot is not None:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
