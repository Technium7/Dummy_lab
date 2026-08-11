from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[2]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.grab_sensor.config_loader import (  # noqa: E402
    apply_runtime_calibrations,
    default_config_path,
    load_grab_sensor_config,
)
from high_level_jobs.grab_sensor.pipeline import run_grab_sensor_pipeline  # noqa: E402
from robot_config import default_robot_port  # noqa: E402

_OFFLINE_COLOR = Path(__file__).resolve().parent / "debug" / "grab_sensor.color.png"
_OFFLINE_DEPTH = Path(__file__).resolve().parent / "debug" / "grab_sensor.depth.npy"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Windows flat grab_sensor: D415 detect, perpendicular grasp, optional SCP posts."
    )
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--execute", action="store_true", help="Stream plan to robot")
    p.add_argument("--port", default=default_robot_port())
    p.add_argument("--baudrate", type=int, default=int(os.environ.get("ROBOT_BAUDRATE", "115200")))
    p.add_argument("--color-file", type=Path, default=None)
    p.add_argument("--depth-file", type=Path, default=None)
    p.add_argument(
        "--offline",
        action="store_true",
        help=f"Use saved debug capture ({_OFFLINE_COLOR.name})",
    )
    p.add_argument("--detect-only", action="store_true")
    p.add_argument("--no-fallback-offline", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--visualize-out", type=Path, default=None)
    p.add_argument("--plan-gui", "--gui", action="store_true", dest="plan_gui")
    p.add_argument("--ik-gui", action="store_true")
    p.add_argument("--joint-current-limit-a", type=float, default=None)
    p.add_argument("--camera-serial", type=str, default=None)
    args = p.parse_args()

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
    if args.joint_current_limit_a is not None:
        params["joint_current_limit_a"] = float(args.joint_current_limit_a)
    elif args.execute:
        params.setdefault("joint_current_limit_a", cfg.execute.joint_current_limit_a)
    if args.camera_serial is not None:
        params["camera_serial"] = str(args.camera_serial).strip()

    if args.detect_only:
        from high_level_jobs.grab_sensor.pipeline import run_detection

        _, targets, _ = run_detection(cfg, params)
        print("Grasp target (robot m):", targets.grasp_pos_m.round(4).tolist())
        print("Pre-grasp (robot m):", targets.pre_grasp_pos_m.round(4).tolist())
        print("Grasp RPY (world deg):", targets.debug.get("grasp_rpy_world_deg"))
        return 0

    if args.execute:
        params["port"] = args.port
        params["baudrate"] = args.baudrate

    try:
        run_grab_sensor_pipeline(cfg, params)
    except KeyboardInterrupt:
        print("\n[grab_sensor] Interrupted — robot shutdown runs in finally.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
