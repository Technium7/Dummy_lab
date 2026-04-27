#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
import types
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DUMMY_XACRO = "software/manual_control/Dummy_ros2_description/urdf/Dummy_ros2.xacro"
REST_JOINTS_DEG = [0.0, -73.0, 180.0, 0.0, 0.0, 0.0]
HOME_JOINTS_DEG = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]
CONTROL_DT_MS_DEFAULT = 150.0
LIVE_STREAM_HZ_DEFAULT = 1000.0 / CONTROL_DT_MS_DEFAULT


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


def xacro_to_urdf_text(xacro_path: Path, package_map: Dict[str, Path]) -> str:
    injected_ament_shim = False
    prev_ament_module = None
    prev_ament_packages_module = None

    try:
        import ament_index_python.packages  # type: ignore # noqa: F401
    except ImportError:
        # Provide minimal ament lookup API so python xacro can resolve $(find pkg)
        # from local workspace packages when ROS is not installed.
        prev_ament_module = sys.modules.get("ament_index_python")
        prev_ament_packages_module = sys.modules.get("ament_index_python.packages")

        ament_root = types.ModuleType("ament_index_python")
        ament_packages = types.ModuleType("ament_index_python.packages")

        def get_package_share_directory(package_name: str) -> str:
            root = package_map.get(package_name)
            if root is None:
                raise ModuleNotFoundError(f"package not found: {package_name}")
            return str(root)

        # Some xacro/ROS helpers may request path-returning variant.
        def get_package_share_path(package_name: str) -> Path:
            root = package_map.get(package_name)
            if root is None:
                raise ModuleNotFoundError(f"package not found: {package_name}")
            return root

        ament_packages.get_package_share_directory = get_package_share_directory  # type: ignore[attr-defined]
        ament_packages.get_package_share_path = get_package_share_path  # type: ignore[attr-defined]
        ament_root.packages = ament_packages  # type: ignore[attr-defined]

        sys.modules["ament_index_python"] = ament_root
        sys.modules["ament_index_python.packages"] = ament_packages
        injected_ament_shim = True

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
    finally:
        if injected_ament_shim:
            if prev_ament_module is None:
                sys.modules.pop("ament_index_python", None)
            else:
                sys.modules["ament_index_python"] = prev_ament_module

            if prev_ament_packages_module is None:
                sys.modules.pop("ament_index_python.packages", None)
            else:
                sys.modules["ament_index_python.packages"] = prev_ament_packages_module


def materialize_urdf(description_path: Path, package_map: Dict[str, Path]) -> Tuple[Path, Optional[Path]]:
    if description_path.suffix.lower() == ".urdf":
        return description_path, None

    xml_text = xacro_to_urdf_text(description_path, package_map)
    xml_text = rewrite_ros_paths(xml_text, package_map)

    # Debug: verify calibration tags are present
    if "pybullet_initial_angle" in xml_text:
        print("[materialize_urdf] ✓ Calibration tags (pybullet_initial_angle) found in materialized URDF")
    else:
        print("[materialize_urdf] ✗ WARNING: No calibration tags found in materialized URDF!")

    temp_file = tempfile.NamedTemporaryFile(prefix="manual_ctrl_", suffix=".urdf", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.write(xml_text.encode("utf-8"))
    temp_file.flush()
    temp_file.close()
    return temp_path, temp_path


def move_home_with_feedback(
    robot: AsciiRobotClient,
    home_deg: Sequence[float],
    speed: float = 10.0,
    timeout_s: float = 12.0,
    tolerance_deg: float = 2.0,
    resend_period_s: float = 0.8,
    use_reset: bool = False,
    label: str = "Home",
    on_feedback=None,
) -> Optional[List[float]]:
    print(f"[{label}] Target pose: {[float(v) for v in home_deg]}")
    if use_reset:
        # Firmware has dedicated RESET-to-rest behavior.
        print(f"[{label}] RESET -> {robot.reset()}")
        time.sleep(0.2)
    robot.move_joint(home_deg, speed=speed, wait_ms=250)

    t0 = time.perf_counter()
    next_resend = t0 + resend_period_s
    last_feedback: Optional[List[float]] = None

    while True:
        now = time.perf_counter()
        if now - t0 > timeout_s:
            break

        feedback = robot.get_joint_pos()
        if feedback is not None:
            last_feedback = feedback
            if on_feedback is not None:
                on_feedback(feedback)
            err = max(abs(float(feedback[j]) - float(home_deg[j])) for j in range(6))
            if err <= tolerance_deg:
                print(f"[{label}] Reached target (max error={err:.2f} deg)")
                return feedback

        if now >= next_resend:
            robot.move_joint(home_deg, speed=speed, wait_ms=80)
            next_resend = now + resend_period_s

        time.sleep(0.08)

    if last_feedback is None:
        print(f"[{label}] Timed out waiting for feedback")
    else:
        err = max(abs(float(last_feedback[j]) - float(home_deg[j])) for j in range(6))
        print(f"[{label}] Timed out before convergence (max error={err:.2f} deg)")
    return last_feedback


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
        enable_twist_sliders=False,
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
        print(f"[Exec] STOP -> {robot.stop()}")
        time.sleep(0.2)
        print(f"[Exec] START -> {robot.start()}")
        time.sleep(0.2)
        print(f"[Exec] CMDMODE 2 -> {robot.set_command_mode(2)}")
        time.sleep(0.2)

        if args.home_first:
            move_home_with_feedback(robot, REST_JOINTS_DEG, speed=10.0, use_reset=True, label="Rest")
            move_home_with_feedback(robot, HOME_JOINTS_DEG, speed=10.0, label="Home")

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


def live_control(args: argparse.Namespace) -> None:
    def _fmt6(values: Sequence[float]) -> str:
        return ", ".join(f"{float(v):7.2f}" for v in values[:6])

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
        enable_twist_sliders=args.twist_sliders,
    )

    robot = AsciiRobotClient(port=args.port, baudrate=args.baudrate, timeout=args.timeout)
    if not robot.connect():
        raise SystemExit(f"Failed to connect serial port: {args.port}")

    stream_period = 1.0 / max(1.0, float(args.stream_hz))
    tracker = ContinuousTracker(dt_sec=stream_period)
    feedback_period = 1.0 / max(0.5, float(args.feedback_hz))
    start = time.perf_counter()
    next_stream_at = start
    next_feedback_at = start
    last_debug_at = start
    debug_interval = max(0.0, float(args.debug_log_interval))
    feedback: Optional[List[float]] = None
    latest_target: Optional[List[float]] = None
    last_sent_velocity = [0.0] * 6

    try:
        print(f"[Live] STOP -> {robot.stop()}")
        time.sleep(0.2)
        print(f"[Live] START -> {robot.start()}")
        time.sleep(0.2)
        print(f"[Live] CMDMODE 2 -> {robot.set_command_mode(2)}")
        time.sleep(0.2)

        # Get actual robot position BEFORE opening GUI
        actual_robot_pos = robot.get_joint_pos()
        if actual_robot_pos is not None:
            print(f"[Live] Actual robot position: {_fmt6(actual_robot_pos)}")
        else:
            print("[Live] Warning: Could not read actual robot position; using calibrated REST pose")
            actual_robot_pos = REST_JOINTS_DEG[:]

        generator.setup()
        print("[Live] Teleop running. Control with WASD/QE + HK/UJ and close the window to stop.")
        print("[Live] Click inside PyBullet window to capture keyboard focus.")
        if args.twist_sliders:
            print("[Live] Use PyBullet sliders (vx/vy/vz/wx/wy) if keyboard input is not captured.")

        # Override PyBullet GUI with actual robot position so they start synchronized
        generator.set_joint_positions_deg(actual_robot_pos)
        feedback = actual_robot_pos[:]
        print(f"[Live] GUI synced to actual robot position: {_fmt6(actual_robot_pos)}")

        if args.home_first:
            move_home_with_feedback(
                robot,
                REST_JOINTS_DEG,
                speed=10.0,
                use_reset=True,
                label="Rest",
                on_feedback=generator.set_joint_positions_deg,
            )
            home_feedback = move_home_with_feedback(
                robot,
                HOME_JOINTS_DEG,
                speed=10.0,
                label="Home",
                on_feedback=generator.set_joint_positions_deg,
            )

            # Force a clean state after startup homing so GUI and planner align.
            if home_feedback is None:
                home_feedback = robot.get_joint_pos()
            if home_feedback is not None:
                feedback = home_feedback[:]
                generator.set_joint_positions_deg(home_feedback)
                print("[Live] Post-home GUI sync from robot feedback applied.")
            tracker.clear()
            latest_target = None
            last_sent_velocity = [0.0] * 6
            now_sync = time.perf_counter()
            next_stream_at = now_sync + stream_period
            next_feedback_at = now_sync

        while True:
            sample = generator.step()
            if sample is None:
                break
            if sample.t > args.duration:
                break
            latest_target = sample.joints_deg[:]

            now = time.perf_counter()

            if now >= next_feedback_at:
                fresh = robot.get_joint_pos()
                if fresh is not None:
                    feedback = fresh[:]
                    generator.set_joint_positions_deg(fresh)
                next_feedback_at = now + feedback_period

            if now < next_stream_at:
                continue

            if feedback is None:
                # Hold until first valid feedback to match smooth_ct feedback-based control.
                next_stream_at = now + stream_period
                continue

            if latest_target is not None:
                tracker.replan(start_position=feedback, start_velocity=last_sent_velocity, target_position=latest_target)

            if tracker.has_plan():
                cmd, v_traj, _done, mode = tracker.step(feedback)
                speed = deg_speed_to_ascii_speed(v_traj)
                ok = robot.send_trajectory_point_async(cmd, speed=speed)
                if ok:
                    last_sent_velocity = v_traj[:]
                    if mode == "DONE":
                        last_sent_velocity = [0.0] * 6
            else:
                cmd = feedback[:]
                v_traj = [0.0] * 6
                mode = "HOLD"
                speed = deg_speed_to_ascii_speed(v_traj)
                ok = robot.send_trajectory_point_async(feedback, speed=deg_speed_to_ascii_speed([0.0] * 6))

            if not ok:
                raise SystemExit("Failed to stream live teleop command")

            if debug_interval == 0.0 or (now - last_debug_at) >= debug_interval:
                key_text = " ".join(sample.active_keys) if sample.active_keys else "-"
                print(
                    f"[LiveDbg] mode={mode:>4} keys={key_text:<12} "
                    f"fb=[{_fmt6(feedback)}] ik=[{_fmt6(latest_target or feedback)}] "
                    f"cmd=[{_fmt6(cmd)}] spd={speed:6.2f}"
                )
                last_debug_at = now

            next_stream_at = now + stream_period

        print("[Live] Teleop streaming stopped")
    finally:
        generator.close()
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
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
    exe.add_argument("--control-dt-ms", type=float, default=CONTROL_DT_MS_DEFAULT)
    exe.add_argument("--home-first", action="store_true")

    live = sub.add_parser("live", help="Live teleop: keyboard control in PyBullet streamed directly to robot")
    live.add_argument("--root", type=Path, default=PROJECT_ROOT)
    live.add_argument(
        "--robot",
        type=str,
        default=DEFAULT_DUMMY_XACRO,
        help="Path to URDF/Xacro",
    )
    live.add_argument("--duration", type=float, default=300.0)
    live.add_argument("--floating-base", action="store_true")
    live.add_argument("--end-link", type=str, default="")
    live.add_argument("--linear-speed", type=float, default=0.15)
    live.add_argument("--angular-speed-deg", type=float, default=120.0)
    live.add_argument("--joint-vel-limit", type=float, default=0.0)
    live.add_argument("--sim-hz", type=float, default=240.0)
    live.add_argument("--speed-factor", type=float, default=1.0)
    live.add_argument("--rotation-weight", type=float, default=8.0)
    live.add_argument("--joint-force", type=float, default=2000.0)
    live.add_argument("--ik-damping", type=float, default=0.05)
    live.add_argument("--port", type=str, default="COM5")
    live.add_argument("--baudrate", type=int, default=115200)
    live.add_argument("--timeout", type=float, default=0.4)
    live.add_argument("--stream-hz", type=float, default=LIVE_STREAM_HZ_DEFAULT)
    live.add_argument("--feedback-hz", type=float, default=8.0)
    live.add_argument("--debug-log-interval", type=float, default=0.4)
    slider_group = live.add_mutually_exclusive_group()
    slider_group.add_argument("--twist-sliders", dest="twist_sliders", action="store_true")
    slider_group.add_argument("--no-twist-sliders", dest="twist_sliders", action="store_false")
    live.set_defaults(twist_sliders=True)
    live_home = live.add_mutually_exclusive_group()
    live_home.add_argument("--home-first", dest="home_first", action="store_true")
    live_home.add_argument("--no-home-first", dest="home_first", action="store_false")
    live.set_defaults(home_first=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "record":
        record_trajectory(args)
    elif args.cmd == "execute":
        execute_trajectory(args)
    elif args.cmd == "live":
        live_control(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
