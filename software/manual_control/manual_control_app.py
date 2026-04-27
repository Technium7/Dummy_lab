#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ascii_comm import AsciiRobotClient
from control_solver import TeleopIKGenerator
from motion_plan import ContinuousTracker, deg_speed_to_ascii_speed
from trajectory_schema import TrajectoryData, TrajectoryPoint, load_trajectory, save_trajectory


IGNORED_DIR_NAMES = {
    ".git",
    ".conda",
    ".venv",
    "build",
    "cmake-build-debug",
    "install",
    "log",
    "node_modules",
}


def read_package_name(package_xml: Path) -> Optional[str]:
    try:
        root = ET.parse(package_xml).getroot()
    except ET.ParseError:
        return None
    node = root.find("name")
    if node is None or node.text is None:
        return None
    return node.text.strip() or None


def build_package_map(search_root: Path) -> Dict[str, Path]:
    package_map: Dict[str, Path] = {}
    for pkg_xml in search_root.rglob("package.xml"):
        if any(part in IGNORED_DIR_NAMES for part in pkg_xml.parts):
            continue
        pkg_name = read_package_name(pkg_xml)
        if pkg_name:
            package_map[pkg_name] = pkg_xml.parent
    return package_map


def rewrite_ros_paths(xml_text: str, package_map: Dict[str, Path]) -> str:
    find_pattern = re.compile(r"\$\(find\s+([^)\s]+)\)")

    def replace_find(match: re.Match[str]) -> str:
        pkg = match.group(1)
        root = package_map.get(pkg)
        return str(root) if root else match.group(0)

    xml_text = find_pattern.sub(replace_find, xml_text)

    pkg_uri_pattern = re.compile(r"package://([^/]+)/")

    def replace_pkg_uri(match: re.Match[str]) -> str:
        pkg = match.group(1)
        root = package_map.get(pkg)
        if not root:
            return match.group(0)
        return f"{root.as_posix()}/"

    return pkg_uri_pattern.sub(replace_pkg_uri, xml_text)


def xacro_to_urdf_text(xacro_path: Path) -> str:
    try:
        import xacro  # type: ignore

        return xacro.process_file(str(xacro_path)).toxml()
    except ImportError:
        cmd = ["xacro", str(xacro_path)]
        try:
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
            return proc.stdout
        except FileNotFoundError as exc:
            raise SystemExit(
                "Xacro selected but xacro is unavailable. Install python package 'xacro' or ROS xacro CLI."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"xacro failed for {xacro_path}:\n{exc.stderr}") from exc


def materialize_urdf(description_path: Path, package_map: Dict[str, Path]) -> Tuple[Path, Optional[Path]]:
    if description_path.suffix.lower() == ".urdf":
        return description_path, None

    xml_text = xacro_to_urdf_text(description_path)
    xml_text = rewrite_ros_paths(xml_text, package_map)

    temp_file = tempfile.NamedTemporaryFile(prefix="manual_ctrl_", suffix=".urdf", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.write(xml_text.encode("utf-8"))
    temp_file.flush()
    temp_file.close()
    return temp_path, temp_path


def record_trajectory(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    description_path = Path(args.robot).resolve() if Path(args.robot).is_absolute() else (root / args.robot).resolve()
    if not description_path.exists():
        raise SystemExit(f"Robot description not found: {description_path}")

    package_map = build_package_map(root)
    urdf_path, temp_file = materialize_urdf(description_path, package_map)

    search_paths = {root, description_path.parent, urdf_path.parent, *package_map.values()}

    generator = TeleopIKGenerator(
        urdf_path=urdf_path,
        fixed_base=(not args.floating_base),
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
    )

    points: List[TrajectoryPoint] = []
    try:
        generator.setup()
        print("[Record] PyBullet running. Control with WASD/QE + HK/UJ and close the window to stop.")

        start = time.perf_counter()
        while True:
            sample = generator.step()
            if sample is None:
                break
            if sample.t > args.duration:
                break
            points.append(
                TrajectoryPoint(
                    t=sample.t,
                    joints_deg=sample.joints_deg,
                    joint_vel_deg_s=sample.joint_vel_deg_s,
                    active_keys=" ".join(sample.active_keys),
                )
            )
    finally:
        generator.close()
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)

    if not points:
        raise SystemExit("No points recorded")

    data = TrajectoryData(
        version=1,
        dt=1.0 / max(1.0, float(args.sim_hz)),
        joint_names=[f"J{i+1}" for i in range(6)],
        points=points,
    )
    save_trajectory(Path(args.output).resolve(), data)

    elapsed = time.perf_counter() - start
    print(f"[Record] Saved {len(points)} points in {elapsed:.2f}s to {args.output}")


def execute_trajectory(args: argparse.Namespace) -> None:
    data = load_trajectory(Path(args.input).resolve())
    dt_sec = float(args.control_dt_ms) / 1000.0
    tracker = ContinuousTracker(dt_sec=dt_sec)

    robot = AsciiRobotClient(port=args.port, baudrate=args.baudrate, timeout=args.timeout)
    if not robot.connect():
        raise SystemExit(f"Failed to connect serial port: {args.port}")

    try:
        print(f"[Exec] START -> {robot.start()}")
        print(f"[Exec] CMDMODE 2 -> {robot.set_command_mode(2)}")

        if args.home_first:
            home = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]
            robot.move_joint(home, speed=10.0, wait_ms=300)
            time.sleep(1.2)

        feedback = robot.get_joint_pos()
        if feedback is None:
            raise SystemExit("Failed to read initial joint position")

        next_index = 0
        t0 = time.perf_counter()
        next_tick = t0
        last_sent_velocity = [0.0] * 6

        while next_index < len(data.points) or tracker.has_plan():
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)

            elapsed = time.perf_counter() - t0
            while next_index < len(data.points) and data.points[next_index].t <= elapsed:
                target = data.points[next_index].joints_deg
                tracker.replan(start_position=feedback, start_velocity=last_sent_velocity, target_position=target)
                next_index += 1

            fresh = robot.get_joint_pos()
            if fresh is not None:
                feedback = fresh

            if tracker.has_plan():
                cmd, v_traj, _done, mode = tracker.step(feedback)
                speed = deg_speed_to_ascii_speed(v_traj)
                ok = robot.send_trajectory_point_async(cmd, speed=speed)
                if not ok:
                    raise SystemExit("Failed to stream trajectory point")
                last_sent_velocity = v_traj[:]
                if mode == "DONE":
                    last_sent_velocity = [0.0] * 6
            else:
                # Keep hold command while waiting for next point.
                robot.send_trajectory_point_async(feedback, speed=deg_speed_to_ascii_speed([0.0] * 6))

            next_tick += dt_sec

        print("[Exec] Trajectory execution completed")
    finally:
        robot.stop()
        robot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual control Phase 1 pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Record teleop IK trajectory from PyBullet")
    rec.add_argument("--root", type=Path, default=Path.cwd())
    rec.add_argument("--robot", type=str, required=True, help="Path to URDF/Xacro")
    rec.add_argument("--output", type=str, default="software/manual_control/recorded_trajectory.json")
    rec.add_argument("--duration", type=float, default=30.0)
    rec.add_argument("--floating-base", action="store_true")
    rec.add_argument("--end-link", type=str, default="")
    rec.add_argument("--linear-speed", type=float, default=0.15)
    rec.add_argument("--angular-speed-deg", type=float, default=120.0)
    rec.add_argument("--joint-vel-limit", type=float, default=0.0)
    rec.add_argument("--sim-hz", type=float, default=240.0)
    rec.add_argument("--speed-factor", type=float, default=1.0)
    rec.add_argument("--rotation-weight", type=float, default=8.0)
    rec.add_argument("--joint-force", type=float, default=2000.0)
    rec.add_argument("--ik-damping", type=float, default=0.05)

    exe = sub.add_parser("execute", help="Execute recorded trajectory through ASCII protocol")
    exe.add_argument("--input", type=str, required=True)
    exe.add_argument("--port", type=str, default="COM5")
    exe.add_argument("--baudrate", type=int, default=115200)
    exe.add_argument("--timeout", type=float, default=0.4)
    exe.add_argument("--control-dt-ms", type=float, default=150.0)
    exe.add_argument("--home-first", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "record":
        record_trajectory(args)
    elif args.cmd == "execute":
        execute_trajectory(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
