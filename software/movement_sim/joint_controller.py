"""Interactive URDF/Xacro joint controller with a separate end-link status window.

Usage:
  python software/movement_sim/joint_controller.py
  python software/movement_sim/joint_controller.py --robot Dummy_ros2
  python software/movement_sim/joint_controller.py --robot software/movement_sim/Dummy_ros2_description/urdf/Dummy_ros2.xacro
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    raise SystemExit("Missing dependency: pybullet. Install with: pip install pybullet") from exc


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize URDF/Xacro and control joints with GUI sliders.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Workspace root to search recursively for .urdf/.xacro files.")
    parser.add_argument("--robot", type=str, default="", help="URDF/Xacro path, or a case-insensitive name filter.")
    parser.add_argument("--fixed-base", action="store_true", default=True, help="Load robot as fixed base (default: true).")
    parser.add_argument("--floating-base", action="store_true", help="Override and allow robot base to move under gravity.")
    parser.add_argument("--collision-check", choices=("off", "warn", "stop"), default="warn", help="Self-collision behavior when links interpenetrate: off, warn, or stop.")
    parser.add_argument("--penetration-threshold", type=float, default=1e-4, help="Minimum penetration depth in meters to report (default: 1e-4).")
    parser.add_argument("--status-window", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def discover_descriptions(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in {".urdf", ".xacro"} and path.name.lower() != "materials.xacro":
            candidates.append(path)
    return sorted(candidates)


def choose_description(candidates: Sequence[Path], robot_hint: str, root: Path) -> Path:
    if not candidates:
        raise SystemExit(f"No .urdf or .xacro files found under {root}")

    if robot_hint:
        hint_path = (root / robot_hint).resolve() if not Path(robot_hint).is_absolute() else Path(robot_hint)
        if hint_path.exists() and hint_path.suffix.lower() in {".urdf", ".xacro"}:
            return hint_path
        filtered = [path for path in candidates if robot_hint.lower() in str(path).lower()]
        if len(filtered) == 1:
            return filtered[0]
        if len(filtered) > 1:
            candidates = filtered
        else:
            raise SystemExit(f"No robot description matched --robot={robot_hint!r}")

    if len(candidates) == 1:
        return candidates[0]

    print("Found multiple robot descriptions:")
    for idx, item in enumerate(candidates, start=1):
        print(f"  {idx}. {item}")
    while True:
        selection = input("Choose one by number: ").strip()
        if not selection.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(selection)
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]
        print(f"Number out of range (1..{len(candidates)}).")


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


def prepare_xacro_without_ament(xacro_path: Path, package_map: Dict[str, Path]) -> Path:
    matching_roots = [root for root in package_map.values() if root in xacro_path.parents]
    package_root = sorted(matching_roots, key=lambda p: len(str(p)), reverse=True)[0] if matching_roots else xacro_path.parent

    temp_root = Path(tempfile.mkdtemp(prefix="xacro_pre_"))
    temp_pkg_root = temp_root / package_root.name
    shutil.copytree(package_root, temp_pkg_root)

    for path in temp_pkg_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".xacro", ".urdf", ".gazebo", ".ros2control", ".xml"}:
            continue
        text = path.read_text(encoding="utf-8")
        path.write_text(rewrite_ros_paths(text, package_map), encoding="utf-8")

    return temp_pkg_root / xacro_path.relative_to(package_root)


def xacro_to_urdf_text(xacro_path: Path, package_map: Dict[str, Path]) -> str:
    try:
        import xacro  # type: ignore

        try:
            return xacro.process_file(str(xacro_path)).toxml()
        except Exception as exc:
            if "ament_index_python" not in str(exc):
                raise
            patched_xacro = prepare_xacro_without_ament(xacro_path, package_map)
            try:
                return xacro.process_file(str(patched_xacro)).toxml()
            finally:
                shutil.rmtree(patched_xacro.parent.parent, ignore_errors=True)
    except ImportError:
        cmd = ["xacro", str(xacro_path)]
        try:
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
            return proc.stdout
        except FileNotFoundError as exc:
            raise SystemExit("Xacro file selected but xacro is not available. Install python package 'xacro' or ROS xacro CLI.") from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"xacro failed for {xacro_path}:\n{exc.stderr}") from exc


def materialize_urdf(description_path: Path, package_map: Dict[str, Path]) -> Tuple[Path, Optional[Path]]:
    if description_path.suffix.lower() == ".urdf":
        return description_path, None

    xml_text = rewrite_ros_paths(xacro_to_urdf_text(description_path, package_map), package_map)
    temp_file = tempfile.NamedTemporaryFile(prefix="urdf_view_", suffix=".urdf", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.write(xml_text.encode("utf-8"))
    temp_file.flush()
    temp_file.close()
    return temp_path, temp_path


def joint_limits_or_defaults(joint_type: int, lower: float, upper: float) -> Tuple[float, float]:
    if lower <= upper and not (lower == 0.0 and upper == -1.0):
        return lower, upper
    if joint_type == p.JOINT_REVOLUTE:
        return -math.pi, math.pi
    if joint_type == p.JOINT_PRISMATIC:
        return -0.1, 0.1
    return -1.0, 1.0


def start_status_window_process(script_path: Path) -> Optional[subprocess.Popen[str]]:
    cmd = [sys.executable, str(script_path), "--status-window"]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except Exception:
        return None


def send_status_update(
    proc: Optional[subprocess.Popen[str]],
    end_link_pose: Optional[Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]],
    note: str = "",
) -> None:
    if proc is None or proc.stdin is None:
        return
    payload: Dict[str, object] = {"note": note, "available": end_link_pose is not None}
    if end_link_pose is not None:
        index, pos, rpy = end_link_pose
        payload.update({"index": index, "pos": list(pos), "rpy": list(rpy)})
    try:
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
    except Exception:
        pass


def stop_status_window_process(proc: Optional[subprocess.Popen[str]]) -> None:
    if proc is None:
        return
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass


def run_status_window() -> None:
    try:
        import tkinter as tk
    except Exception as exc:
        raise SystemExit(f"Tkinter is not available in this Python environment: {exc}") from exc

    try:
        import select
    except Exception as exc:
        raise SystemExit(f"Select is not available: {exc}") from exc

    root = tk.Tk()
    root.title("End Link Status")
    root.geometry("420x160")
    root.attributes("-topmost", True)

    title_var = tk.StringVar(value="End Link: unavailable")
    pos_var = tk.StringVar(value="pos: -")
    rpy_var = tk.StringVar(value="rpy: -")
    note_var = tk.StringVar(value="")

    tk.Label(root, textvariable=title_var, font=("Helvetica", 16, "bold")).pack(pady=(10, 2))
    tk.Label(root, textvariable=pos_var, font=("Helvetica", 13)).pack()
    tk.Label(root, textvariable=rpy_var, font=("Helvetica", 13)).pack()
    tk.Label(root, textvariable=note_var, font=("Helvetica", 11), foreground="#aa0000").pack(pady=(6, 0))

    def apply_payload(payload: Dict[str, object]) -> None:
        note_var.set(str(payload.get("note", "")))
        if not bool(payload.get("available", False)):
            title_var.set("End Link: unavailable")
            pos_var.set("pos: -")
            rpy_var.set("rpy: -")
            return
        index = int(payload.get("index", -1))
        pos = payload.get("pos", [0.0, 0.0, 0.0])
        rpy = payload.get("rpy", [0.0, 0.0, 0.0])
        title_var.set(f"End Link {index}")
        pos_var.set(f"pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        rpy_var.set(f"rpy: ({rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f})")

    def poll_stdin() -> None:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                line = sys.stdin.readline()
                if line == "":
                    root.destroy()
                    return
                payload = json.loads(line)
                if isinstance(payload, dict):
                    apply_payload(payload)
        except Exception:
            pass
        root.after(50, poll_stdin)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.after(50, poll_stdin)
    root.mainloop()


def launch_viewer(
    urdf_path: Path,
    fixed_base: bool,
    extra_search_paths: Iterable[Path],
    collision_check: str,
    penetration_threshold: float,
    status_proc: Optional[subprocess.Popen[str]],
) -> None:
    cid = p.connect(p.GUI)
    if cid < 0:
        raise SystemExit("Failed to open PyBullet GUI")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    for path in extra_search_paths:
        if path.exists() and path.is_dir():
            p.setAdditionalSearchPath(str(path))

    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    plane_path = Path(pybullet_data.getDataPath()) / "plane.urdf"
    if plane_path.exists():
        try:
            p.loadURDF(str(plane_path))
        except Exception:
            pass

    load_flags = p.URDF_USE_INERTIA_FROM_FILE
    if collision_check != "off":
        load_flags |= p.URDF_USE_SELF_COLLISION
        load_flags |= p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    robot_id = p.loadURDF(str(urdf_path), useFixedBase=fixed_base, flags=load_flags)

    parent_of_link: Dict[int, int] = {}
    for joint_idx in range(p.getNumJoints(robot_id)):
        parent_of_link[joint_idx] = int(p.getJointInfo(robot_id, joint_idx)[16])

    def should_ignore_pair(link_a: int, link_b: int) -> bool:
        if link_a < 0 or link_b < 0 or link_a == link_b:
            return True
        if parent_of_link.get(link_a, -999) == link_b or parent_of_link.get(link_b, -999) == link_a:
            return True
        return False

    def collect_penetrations() -> Dict[Tuple[int, int], float]:
        pairs: Dict[Tuple[int, int], float] = {}
        for contact in p.getContactPoints(bodyA=robot_id, bodyB=robot_id):
            link_a = int(contact[3])
            link_b = int(contact[4])
            if should_ignore_pair(link_a, link_b):
                continue
            penetration = max(0.0, -float(contact[8]))
            if penetration >= penetration_threshold:
                pair = tuple(sorted((link_a, link_b)))
                pairs[pair] = max(pairs.get(pair, 0.0), penetration)
        for point in p.getClosestPoints(bodyA=robot_id, bodyB=robot_id, distance=max(1e-3, penetration_threshold * 10.0)):
            link_a = int(point[3])
            link_b = int(point[4])
            if should_ignore_pair(link_a, link_b):
                continue
            distance = float(point[8])
            if distance < 0.0:
                pair = tuple(sorted((link_a, link_b)))
                pairs[pair] = max(pairs.get(pair, 0.0), -distance)
        return pairs

    def link_has_blocking_contact(link_idx: int) -> bool:
        for c in p.getContactPoints(bodyA=robot_id, linkIndexA=link_idx):
            if int(c[2]) != robot_id or not should_ignore_pair(link_idx, int(c[4])):
                return True
        for c in p.getContactPoints(bodyB=robot_id, linkIndexB=link_idx):
            if int(c[1]) != robot_id or not should_ignore_pair(int(c[3]), link_idx):
                return True
        return False

    def get_end_link_pose() -> Optional[Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]]:
        num_joints = p.getNumJoints(robot_id)
        if num_joints <= 0:
            return None
        end_link_idx = num_joints - 1
        state = p.getLinkState(robot_id, end_link_idx, computeForwardKinematics=True)
        position = tuple(float(v) for v in state[4])
        rotation = tuple(float(v) for v in p.getEulerFromQuaternion(state[5]))
        return end_link_idx, position, rotation

    def create_slider_for_joint(joint_idx: int) -> Optional[Tuple[int, int, str]]:
        info = p.getJointInfo(robot_id, joint_idx)
        joint_type = info[2]
        if joint_type not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            return None
        name = info[1].decode("utf-8", errors="replace")
        lower, upper = joint_limits_or_defaults(joint_type, float(info[8]), float(info[9]))
        current = float(p.getJointState(robot_id, joint_idx)[0])
        slider_id = p.addUserDebugParameter(name, lower, upper, max(lower, min(current, upper)))
        return joint_idx, slider_id, name

    def rebuild_all_sliders() -> Dict[int, Tuple[int, str]]:
        try:
            p.removeAllUserParameters()
        except Exception:
            pass
        sliders: Dict[int, Tuple[int, str]] = {}
        for joint_idx in range(p.getNumJoints(robot_id)):
            slider = create_slider_for_joint(joint_idx)
            if slider is not None:
                sliders[joint_idx] = (slider[1], slider[2])
        return sliders

    aabb_min, aabb_max = p.getAABB(robot_id, -1)
    cx = (aabb_min[0] + aabb_max[0]) * 0.5
    cy = (aabb_min[1] + aabb_max[1]) * 0.5
    cz = (aabb_min[2] + aabb_max[2]) * 0.5
    span = max(abs(aabb_max[0] - aabb_min[0]), abs(aabb_max[1] - aabb_min[1]), abs(aabb_max[2] - aabb_min[2]))
    p.resetDebugVisualizerCamera(cameraDistance=max(0.2, span) * 2.5, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=[cx, cy, cz])

    joint_slider_map = rebuild_all_sliders()
    baseline_pairs = collect_penetrations() if collision_check != "off" else {}
    last_reported_pairs: Dict[Tuple[int, int], float] = {}
    stall_frames: Dict[int, int] = {}
    last_stall_warn_frame: Dict[int, int] = {}
    frame_idx = 0
    last_status_frame = 0

    status_note = ""
    if not joint_slider_map:
        status_note = "No controllable joints found"
    send_status_update(status_proc, get_end_link_pose(), note=status_note)

    print("PyBullet is running. Close the GUI window to exit.")
    if collision_check != "off":
        print("Collision reporting ignores base and parent-child pairs, and compares against startup baseline.")

    while p.isConnected():
        frame_idx += 1
        rebuild_needed = False
        for joint_idx, (slider_id, joint_name) in list(joint_slider_map.items()):
            try:
                target = p.readUserDebugParameter(slider_id)
            except Exception:
                rebuild_needed = True
                break

            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=500.0,
            )

            position, velocity, _, applied_torque = p.getJointState(robot_id, joint_idx)
            error = abs(target - float(position))
            if error > 0.02 and abs(float(velocity)) < 1e-3:
                stall_frames[joint_idx] = stall_frames.get(joint_idx, 0) + 1
            else:
                stall_frames[joint_idx] = 0

            if stall_frames[joint_idx] >= 20:
                last_warn = last_stall_warn_frame.get(joint_idx, -240)
                if frame_idx - last_warn >= 240:
                    info = p.getJointInfo(robot_id, joint_idx)
                    joint_type = int(info[2])
                    lower, upper = joint_limits_or_defaults(joint_type, float(info[8]), float(info[9]))
                    limit_margin = max(5e-3, 0.02 * max(1.0, abs(upper - lower)))
                    if float(position) <= lower + limit_margin and target <= lower + limit_margin:
                        reason = "at lower joint limit"
                    elif float(position) >= upper - limit_margin and target >= upper - limit_margin:
                        reason = "at upper joint limit"
                    elif link_has_blocking_contact(joint_idx):
                        reason = "blocked by collision/contact"
                    elif abs(float(applied_torque)) > 450.0:
                        reason = "motor near force limit"
                    else:
                        reason = "constrained by kinematics or dynamics"
                    send_status_update(status_proc, get_end_link_pose(), note=f"Slider '{joint_name}' not responding: {reason}")
                    last_stall_warn_frame[joint_idx] = frame_idx

        if rebuild_needed:
            joint_slider_map = rebuild_all_sliders()
            stall_frames.clear()
            last_stall_warn_frame.clear()
            send_status_update(status_proc, get_end_link_pose(), note="Slider UI rebuilt")
            continue

        p.stepSimulation()

        if frame_idx - last_status_frame >= 30:
            send_status_update(status_proc, get_end_link_pose())
            last_status_frame = frame_idx

        if collision_check != "off":
            current_pairs = collect_penetrations()
            worst_message = ""
            for pair, penetration in current_pairs.items():
                baseline = baseline_pairs.get(pair, 0.0)
                excess = penetration - baseline
                if excess <= max(penetration_threshold, 1e-3):
                    continue
                if penetration <= last_reported_pairs.get(pair, -1.0) + 1e-5:
                    continue
                last_reported_pairs[pair] = penetration
                worst_message = (
                    "Self-collision detected beyond startup baseline: "
                    f"links {pair[0]} and {pair[1]}, penetration={penetration:.6f} m, excess={excess:.6f} m"
                )
                if collision_check == "stop":
                    send_status_update(status_proc, get_end_link_pose(), note=worst_message)
                    raise SystemExit(worst_message)
            if worst_message:
                send_status_update(status_proc, get_end_link_pose(), note=worst_message)

        time.sleep(1.0 / 240.0)


def main() -> None:
    args = parse_args()
    if args.status_window:
        run_status_window()
        return

    root = args.root.resolve()
    fixed_base = args.fixed_base and not args.floating_base
    candidates = discover_descriptions(root)
    selected = choose_description(candidates, args.robot, root)
    package_map = build_package_map(root)
    urdf_path, temp_file = materialize_urdf(selected, package_map)
    status_proc = start_status_window_process(Path(__file__).resolve())

    try:
        launch_viewer(
            urdf_path=urdf_path,
            fixed_base=fixed_base,
            extra_search_paths={root, selected.parent, urdf_path.parent, *package_map.values()},
            collision_check=args.collision_check,
            penetration_threshold=max(0.0, args.penetration_threshold),
            status_proc=status_proc,
        )
    finally:
        stop_status_window_process(status_proc)
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
