from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from controller_runtime import launch_controller
from robot_config import DEFAULT_ROBOT, default_robot_port, repo_root
from urdf_tools import build_package_map, choose_description, discover_descriptions, materialize_urdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual control runtime (modular).")
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT)
    parser.add_argument("--fixed-base", action="store_true", default=True)
    parser.add_argument("--floating-base", action="store_true")
    parser.add_argument("--end-link", type=str, default="")
    parser.add_argument("--linear-speed", type=float, default=0.018)
    parser.add_argument("--angular-speed-deg", type=float, default=12.0)
    parser.add_argument("--joint-vel-limit", type=float, default=0.0)
    parser.add_argument("--sim-hz", type=float, default=480.0)
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="Sim time multiplier (>1 runs sim faster than real time; 1 keeps sim in sync with wall clock).",
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
        help="Period between robot & waypoints in ms (lower = less lag; min 20).",
    )
    parser.add_argument(
        "--no-home-firmware",
        action="store_true",
        help="T key only streams HOME joints; default also sends firmware !HOME in the background.",
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
    search_paths = {root, selected.parent, urdf_path.parent, *package_map.values()}
    try:
        launch_controller(
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
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

