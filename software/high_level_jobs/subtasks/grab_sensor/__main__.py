from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SOFTWARE_DIR = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from high_level_jobs.subtasks.grab_sensor.config_loader import default_config_path, load_grab_sensor_config
from high_level_jobs.subtasks.grab_sensor.pipeline import run_grab_sensor_pipeline
from robot_ascii_interface import RobotAsciiInterface


def main() -> int:
    p = argparse.ArgumentParser(description="Detect sensor with D415, plan grasp, optionally execute.")
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--execute", action="store_true", help="Stream plan to robot and close gripper")
    p.add_argument("--port", default=os.environ.get("ROBOT_PORT", "COM3"))
    p.add_argument("--baudrate", type=int, default=int(os.environ.get("ROBOT_BAUDRATE", "115200")))
    p.add_argument("--color-file", type=Path, default=None, help="Offline test from saved capture")
    p.add_argument("--depth-file", type=Path, default=None)
    p.add_argument("--detect-only", action="store_true", help="Vision + print targets; no planning")
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Save grasp_overlay.png with grasp target highlighted on the image",
    )
    p.add_argument("--visualize-out", type=Path, default=None, help="Output path for overlay PNG")
    args = p.parse_args()

    cfg = load_grab_sensor_config(args.config)
    params = {
        "execute": args.execute,
        "color_file": str(args.color_file) if args.color_file else None,
        "depth_file": str(args.depth_file) if args.depth_file else None,
        "visualize": args.visualize or args.detect_only,
        "visualize_out": str(args.visualize_out) if args.visualize_out else None,
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
        print("Grasp position (robot m):", targets.grasp_pos_m.round(4).tolist())
        print("Grasp rpy (robot deg):", targets.debug.get("grasp_rpy_robot_deg"))
        return 0

    run_grab_sensor_pipeline(cfg, params, robot=robot)
    if robot is not None:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
