from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_MANUAL_CONTROL_DIR = Path(__file__).resolve().parent
_SOFTWARE_DIR = _MANUAL_CONTROL_DIR.parent
for _path in (_SOFTWARE_DIR, _MANUAL_CONTROL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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
    parser.add_argument(
        "--creep-step-mm",
        type=float,
        default=0.12,
        help="WASDQE click step size in mm (default 0.12, matches epoxy z_creep).",
    )
    parser.add_argument(
        "--creep-speed-mm-s",
        type=float,
        default=1.65,
        help="Peak creep linear speed mm/s (default 1.65, matches epoxy z_creep).",
    )
    parser.add_argument(
        "--creep-accel-s",
        type=float,
        default=0.20,
        help="Creep smoothstep ramp-up time in seconds (default 0.20).",
    )
    parser.add_argument(
        "--no-creep",
        action="store_true",
        help="Disable click-creep; restore hold-to-move WASDQE velocity jog.",
    )
    parser.add_argument("--port", type=str, default=default_robot_port())
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--control-dt-ms",
        type=float,
        default=70.0,
        help="Robot & waypoint period in ms (default 70).",
    )
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help="PyBullet preview only; skip robot serial connection.",
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
        help="Set motor current limit (A) for J1-J6 after !START via #I_LIMIT_J "
        "(default 0=skip; motors already store limit in EEPROM).",
    )
    parser.add_argument(
        "--record-demos",
        action="store_true",
        help="Arm epoxy_app demo recording (live Zelux+Infinity; R/C keys).",
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
    teleop_hook = None
    on_shutdown = None
    if args.record_demos:
        from high_level_jobs.epoxy_app.record import create_manual_control_record_hook

        teleop_hook, recorder = create_manual_control_record_hook()
        on_shutdown = recorder.shutdown
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
            sim_only=args.sim_only,
            debug_serial=args.debug_serial,
            joint_current_limit_a=args.joint_current_limit_a,
            creep_enabled=not bool(args.no_creep),
            creep_step_m=max(0.0, float(args.creep_step_mm) * 1e-3),
            creep_linear_speed=max(0.0, float(args.creep_speed_mm_s) * 1e-3),
            creep_accel_time=max(1e-3, float(args.creep_accel_s)),
            teleop_hook=teleop_hook,
            on_shutdown=on_shutdown,
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

