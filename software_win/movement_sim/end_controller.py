"""Interactive end-effector controller using IK.

Usage:
  python software/movement_sim/end_controller.py
  python software/movement_sim/end_controller.py --robot Dummy_ros2_gripper
  python software/movement_sim/end_controller.py --robot my_robot.urdf --end-link tool0
"""

from __future__ import annotations

import argparse
import sys
import math
import multiprocessing as mp
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import tkinter as tk
except Exception:
    tk = None  # type: ignore[assignment]

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    raise SystemExit("Missing dependency: pybullet. Install with: pip install pybullet") from exc

_SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from robot_config import DEFAULT_ROBOT, repo_root  # noqa: E402

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
    parser = argparse.ArgumentParser(
        description="Control end-effector pose with sliders and IK."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root(),
        help="Workspace root to search recursively for .urdf/.xacro files.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=DEFAULT_ROBOT,
        help="URDF/Xacro path, or a case-insensitive name filter.",
    )
    parser.add_argument(
        "--fixed-base",
        action="store_true",
        default=True,
        help="Load robot as fixed base (default: true).",
    )
    parser.add_argument(
        "--floating-base",
        action="store_true",
        help="Override and allow robot base to move (not fixed to the world).",
    )
    parser.add_argument(
        "--end-link",
        type=str,
        default="",
        help="End-effector link name or link index. Default: last joint link.",
    )
    parser.add_argument(
        "--linear-speed",
        type=float,
        default=0.15,
        help="End-effector linear speed (m/s) for WASD/QE control.",
    )
    parser.add_argument(
        "--angular-speed-deg",
        type=float,
        default=120.0,
        help="End-effector angular speed (deg/s) for HK/UJ control.",
    )
    parser.add_argument(
        "--joint-vel-limit",
        type=float,
        default=0.0,
        help="Optional absolute per-joint velocity limit; 0 disables clamping.",
    )
    parser.add_argument(
        "--sim-hz",
        type=float,
        default=480.0,
        help="Physics/control loop rate in Hz.",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=2.0,
        help="Wall-clock speed multiplier relative to real time.",
    )
    parser.add_argument(
        "--rotation-weight",
        type=float,
        default=8.0,
        help="Relative weight applied to angular task rows in the Jacobian solve.",
    )
    parser.add_argument(
        "--joint-force",
        type=float,
        default=2000.0,
        help="Maximum actuator force/torque used by joint velocity control.",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=0.05,
        help="DLS regularization factor for Jacobian inverse velocity solve.",
    )
    return parser.parse_args()


def discover_descriptions(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in {".urdf", ".xacro"}:
            if path.name.lower() in {"materials.xacro"}:
                continue
            candidates.append(path)
    return sorted(candidates)


def choose_description(candidates: Sequence[Path], robot_hint: str, root: Path) -> Path:
    if not candidates:
        raise SystemExit(f"No .urdf or .xacro files found under {root}")

    if robot_hint:
        hint_path = (root / robot_hint).resolve() if not Path(robot_hint).is_absolute() else Path(robot_hint)
        if hint_path.exists() and hint_path.suffix.lower() in {".urdf", ".xacro"}:
            return hint_path

        filtered = [p for p in candidates if robot_hint.lower() in str(p).lower()]
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
        # POSIX paths so URDF mesh URIs work on Windows.
        return root.as_posix() if root else match.group(0)

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
        text = rewrite_ros_paths(text, package_map)
        path.write_text(text, encoding="utf-8")

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
            raise SystemExit(
                "Xacro file selected but xacro is not available. "
                "Install python package 'xacro' or ROS xacro CLI."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"xacro failed for {xacro_path}:\n{exc.stderr}") from exc


def materialize_urdf(description_path: Path, package_map: Dict[str, Path]) -> Tuple[Path, Optional[Path]]:
    if description_path.suffix.lower() == ".urdf":
        return description_path, None

    xml_text = xacro_to_urdf_text(description_path, package_map)
    xml_text = rewrite_ros_paths(xml_text, package_map)

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
        return -0.2, 0.2
    return -1.0, 1.0


def get_controllable_joints(robot_id: int) -> Tuple[List[int], List[float], List[float], List[float], List[str]]:
    indices: List[int] = []
    lowers: List[float] = []
    uppers: List[float] = []
    ranges: List[float] = []
    names: List[str] = []

    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_type = info[2]
        if joint_type not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            continue
        lower, upper = joint_limits_or_defaults(joint_type, float(info[8]), float(info[9]))
        indices.append(i)
        lowers.append(lower)
        uppers.append(upper)
        ranges.append(max(1e-6, upper - lower))
        names.append(info[1].decode("utf-8", errors="replace"))

    return indices, lowers, uppers, ranges, names


def resolve_end_link(robot_id: int, hint: str) -> int:
    num_joints = p.getNumJoints(robot_id)
    if num_joints <= 0:
        return -1

    if hint:
        if hint.isdigit() or (hint.startswith("-") and hint[1:].isdigit()):
            idx = int(hint)
            if -1 <= idx < num_joints:
                return idx
            raise SystemExit(f"--end-link index out of range: {idx}")

        lookup = hint.lower()
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            child_name = info[12].decode("utf-8", errors="replace").lower()
            joint_name = info[1].decode("utf-8", errors="replace").lower()
            if lookup == child_name or lookup == joint_name:
                return i
        raise SystemExit(f"--end-link={hint!r} did not match any link/joint name")

    return num_joints - 1


def is_key_down(events: Dict[int, int], char: str) -> bool:
    lower = ord(char.lower())
    upper = ord(char.upper())
    mask = p.KEY_IS_DOWN
    return ((events.get(lower, 0) & mask) != 0) or ((events.get(upper, 0) & mask) != 0)


def status_gui_process_main(queue_obj) -> None:
    if tk is None:
        return

    root = tk.Tk()
    root.title("End Controller Inputs")
    root.geometry("760x520")

    mapping = (
        "Linear: W/S -> +X/-X, A/D -> +Y/-Y, Q/E -> +Z/-Z\n"
        "Angular (2-DOF): H/K -> +Wx/-Wx, U/J -> +Wy/-Wy"
    )

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Keyboard Mapping", font=("Helvetica", 13, "bold"), anchor="w").pack(fill="x")
    tk.Label(frame, text=mapping, justify="left", anchor="w", font=("Helvetica", 11)).pack(fill="x", pady=(4, 10))

    keys_var = tk.StringVar(value="Active keys: (none)")
    twist_var = tk.StringVar(value="vx=0.000 vy=0.000 vz=0.000 m/s | wx=0.000 wy=0.000 rad/s")
    joints_var = tk.StringVar(value="Joint positions:\n  (waiting for simulator data)")

    tk.Label(frame, textvariable=keys_var, anchor="w", font=("Helvetica", 12)).pack(fill="x", pady=(2, 6))
    tk.Label(frame, textvariable=twist_var, anchor="w", font=("Helvetica", 12)).pack(fill="x")
    tk.Label(frame, text="Current Joint Locations", font=("Helvetica", 12, "bold"), anchor="w").pack(
        fill="x", pady=(10, 4)
    )
    tk.Label(
        frame,
        textvariable=joints_var,
        justify="left",
        anchor="nw",
        font=("Consolas", 10),
    ).pack(fill="both", expand=True)

    def poll() -> None:
        while True:
            try:
                msg = queue_obj.get_nowait()
            except Exception:
                break

            if not isinstance(msg, dict):
                continue
            if msg.get("cmd") == "quit":
                try:
                    root.destroy()
                except Exception:
                    pass
                return

            keys_var.set(f"Active keys: {msg.get('active_keys', '(none)')}")
            twist_var.set(str(msg.get("twist", twist_var.get())))
            joints_var.set(str(msg.get("joint_locations", joints_var.get())))

        root.after(50, poll)

    poll()
    root.mainloop()


class ControlStatusWindow:
    def __init__(self) -> None:
        self._proc: Optional[mp.Process] = None
        self._queue = None

    def start(self) -> bool:
        if tk is None:
            return False
        try:
            self._queue = mp.Queue(maxsize=8)
            self._proc = mp.Process(
                target=status_gui_process_main,
                args=(self._queue,),
                name="control-status-gui",
                daemon=True,
            )
            self._proc.start()
            return self._proc.is_alive()
        except Exception:
            self._queue = None
            self._proc = None
            return False

    def update(
        self,
        active_keys: Sequence[str],
        twist: Sequence[float],
        joint_names: Sequence[str],
        joint_positions: Sequence[float],
    ) -> None:
        if self._queue is None:
            return
        key_text = " ".join(active_keys) if active_keys else "(none)"
        twist_text = (
            f"vx={twist[0]:+.3f} vy={twist[1]:+.3f} vz={twist[2]:+.3f} m/s"
            f" | wx={twist[3]:+.3f} wy={twist[4]:+.3f} rad/s"
        )
        joint_lines = [f"{name}: {pos:+.4f}" for name, pos in zip(joint_names, joint_positions)]
        joint_text = "Joint positions:\n" + ("\n".join(joint_lines) if joint_lines else "  (none)")
        msg = {"active_keys": key_text, "twist": twist_text, "joint_locations": joint_text}
        try:
            self._queue.put_nowait(msg)
        except Exception:
            pass

    def close(self) -> None:
        if self._queue is not None:
            try:
                self._queue.put_nowait({"cmd": "quit"})
            except Exception:
                pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=0.5)
            if self._proc.is_alive():
                self._proc.terminate()
        self._queue = None
        self._proc = None


def solve_linear_system(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(a)
    if n == 0 or len(b) != n:
        return None

    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv

        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]

    return [m[i][n] for i in range(n)]


def compute_dls_joint_velocity(
    robot_id: int,
    end_link_idx: int,
    joint_indices: Sequence[int],
    task_twist: Sequence[float],
    damping: float,
    rotation_weight: float,
) -> Optional[List[float]]:
    num_joints = p.getNumJoints(robot_id)
    if num_joints <= 0:
        return None

    q_all = [float(p.getJointState(robot_id, j)[0]) for j in range(num_joints)]
    zeros = [0.0] * num_joints

    jacobian = p.calculateJacobian(
        bodyUniqueId=robot_id,
        linkIndex=end_link_idx,
        localPosition=[0.0, 0.0, 0.0],
        objPositions=q_all,
        objVelocities=zeros,
        objAccelerations=zeros,
    )
    linear_jac = jacobian[0]
    angular_jac = jacobian[1]

    # Task-space rows: vx, vy, vz, wx, wy. Angular rows are weighted higher
    # so orientation corrections survive when the Jacobian must trade off tasks.
    task_rows: List[List[float]] = []
    row_weights = (1.0, 1.0, 1.0, float(rotation_weight), float(rotation_weight))
    for src_row, row_weight in zip(
        (linear_jac[0], linear_jac[1], linear_jac[2], angular_jac[0], angular_jac[1]),
        row_weights,
    ):
        task_rows.append([float(src_row[j]) * row_weight for j in joint_indices])

    n = len(joint_indices)
    if n == 0:
        return None

    jt = [[task_rows[r][c] for r in range(5)] for c in range(n)]
    h = [[0.0 for _ in range(n)] for _ in range(n)]
    g = [0.0 for _ in range(n)]

    for i in range(n):
        for j in range(n):
            acc = 0.0
            for k in range(5):
                acc += jt[i][k] * task_rows[k][j]
            h[i][j] = acc
        h[i][i] += damping * damping

        rhs = 0.0
        for k in range(5):
            rhs += jt[i][k] * float(task_twist[k]) * row_weights[k]
        g[i] = rhs

    return solve_linear_system(h, g)


def clamp_and_step_toward(
    current: Sequence[float],
    target: Sequence[float],
    lowers: Sequence[float],
    uppers: Sequence[float],
    max_step: float,
) -> List[float]:
    out: List[float] = []
    for cur, tgt, lo, hi in zip(current, target, lowers, uppers):
        feasible = min(max(tgt, lo), hi)
        delta = feasible - cur
        if delta > max_step:
            feasible = cur + max_step
        elif delta < -max_step:
            feasible = cur - max_step
        out.append(min(max(feasible, lo), hi))
    return out


def launch_end_controller(
    urdf_path: Path,
    fixed_base: bool,
    extra_search_paths: Iterable[Path],
    end_link_hint: str,
    linear_speed: float,
    angular_speed_deg: float,
    joint_vel_limit: float,
    sim_hz: float,
    speed_factor: float,
    rotation_weight: float,
    joint_force: float,
    ik_damping: float,
) -> None:
    cid = p.connect(p.GUI)
    if cid < 0:
        raise SystemExit("Failed to open PyBullet GUI")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    for path in extra_search_paths:
        if path.exists() and path.is_dir():
            p.setAdditionalSearchPath(str(path))

    p.resetSimulation()
    p.setGravity(0, 0, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    plane_path = Path(pybullet_data.getDataPath()) / "plane.urdf"
    if plane_path.exists():
        try:
            p.loadURDF(str(plane_path))
        except Exception:
            pass

    load_flags = p.URDF_USE_INERTIA_FROM_FILE

    robot_id = p.loadURDF(
        str(urdf_path),
        useFixedBase=fixed_base,
        flags=load_flags,
    )
    aabb_min, aabb_max = p.getAABB(robot_id, -1)
    cx = (aabb_min[0] + aabb_max[0]) * 0.5
    cy = (aabb_min[1] + aabb_max[1]) * 0.5
    cz = (aabb_min[2] + aabb_max[2]) * 0.5
    sx = abs(aabb_max[0] - aabb_min[0])
    sy = abs(aabb_max[1] - aabb_min[1])
    sz = abs(aabb_max[2] - aabb_min[2])
    radius = max(0.2, sx, sy, sz) * 2.5
    p.resetDebugVisualizerCamera(
        cameraDistance=radius,
        cameraYaw=50,
        cameraPitch=-25,
        cameraTargetPosition=[cx, cy, cz],
    )

    joint_indices, lower_limits, upper_limits, joint_ranges, joint_names = get_controllable_joints(robot_id)
    if not joint_indices:
        raise SystemExit("No controllable revolute/prismatic joints found for IK.")

    end_link_idx = resolve_end_link(robot_id, end_link_hint)
    if end_link_idx == -1:
        raise SystemExit("Robot has no links beyond base; cannot run end-effector IK.")

    end_link_name = p.getJointInfo(robot_id, end_link_idx)[12].decode("utf-8", errors="replace")
    all_joint_indices = list(range(p.getNumJoints(robot_id)))
    all_joint_names = [
        p.getJointInfo(robot_id, idx)[1].decode("utf-8", errors="replace") for idx in all_joint_indices
    ]
    print(f"Using end link index {end_link_idx}: {end_link_name}")
    print("Controllable joints:")
    for idx, name in zip(joint_indices, joint_names):
        print(f"  - [{idx}] {name}")

    print("Keyboard velocity control is active:")
    print("  WASD: x/y translation   Q/E: z translation")
    print("  H/K: +/− wx            U/J: +/− wy")
    print("  Orientation uses 2-DOF Jacobian regularized inverse solve.")
    print("PyBullet is running. Close the GUI window to exit.")
    status_gui = ControlStatusWindow()
    if status_gui.start():
        print("Opened separate control status GUI window.")
    else:
        print("Separate control status GUI unavailable (tkinter not found).")

    last_feasible = [float(p.getJointState(robot_id, j)[0]) for j in joint_indices]

    sim_hz = max(1.0, float(sim_hz))
    speed_factor = max(0.0, float(speed_factor))
    dt = 1.0 / sim_hz
    step_sleep = dt / speed_factor if speed_factor > 0.0 else 0.0
    p.setPhysicsEngineParameter(fixedTimeStep=dt)
    linear_speed = max(0.0, float(linear_speed))
    angular_speed = math.radians(max(0.0, float(angular_speed_deg)))
    joint_vel_limit = max(0.0, float(joint_vel_limit))
    rotation_weight = max(1.0, float(rotation_weight))
    joint_force = max(1.0, float(joint_force))
    damping = max(1e-6, ik_damping)

    try:
        while p.isConnected():
            events = p.getKeyboardEvents()
            vx = 0.0
            vy = 0.0
            vz = 0.0
            wx = 0.0
            wy = 0.0
            active_keys: List[str] = []

            if is_key_down(events, "w"):
                vx += linear_speed
                active_keys.append("W")
            if is_key_down(events, "s"):
                vx -= linear_speed
                active_keys.append("S")
            if is_key_down(events, "a"):
                vy += linear_speed
                active_keys.append("A")
            if is_key_down(events, "d"):
                vy -= linear_speed
                active_keys.append("D")
            if is_key_down(events, "q"):
                vz += linear_speed
                active_keys.append("Q")
            if is_key_down(events, "e"):
                vz -= linear_speed
                active_keys.append("E")

            if is_key_down(events, "h"):
                wx -= angular_speed
                active_keys.append("H")
            if is_key_down(events, "k"):
                wx += angular_speed
                active_keys.append("K")
            if is_key_down(events, "u"):
                wy += angular_speed
                active_keys.append("U")
            if is_key_down(events, "j"):
                wy -= angular_speed
                active_keys.append("J")

            current_state = [float(p.getJointState(robot_id, j)[0]) for j in joint_indices]
            all_joint_positions = [float(p.getJointState(robot_id, j)[0]) for j in all_joint_indices]
            status_gui.update(active_keys, [vx, vy, vz, wx, wy], all_joint_names, all_joint_positions)

            # No key pressed: actively command zero joint velocities.
            if abs(vx) + abs(vy) + abs(vz) + abs(wx) + abs(wy) <= 1e-12:
                for j_idx in joint_indices:
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=j_idx,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0.0,
                        force=joint_force,
                    )
                last_feasible = current_state
                p.stepSimulation()
                time.sleep(step_sleep)
                continue

            qdot = compute_dls_joint_velocity(
                robot_id=robot_id,
                end_link_idx=end_link_idx,
                joint_indices=joint_indices,
                task_twist=[vx, vy, vz, wx, wy],
                damping=damping,
                rotation_weight=rotation_weight,
            )
            if qdot is None:
                for j_idx in joint_indices:
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=j_idx,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0.0,
                        force=joint_force,
                    )
                p.stepSimulation()
                time.sleep(step_sleep)
                continue

            commanded_velocity = [float(vel) for vel in qdot]
            if joint_vel_limit > 0.0:
                commanded_velocity = [
                    min(max(vel, -joint_vel_limit), joint_vel_limit) for vel in commanded_velocity
                ]

            commanded_next = [cur + vel * dt for cur, vel in zip(current_state, commanded_velocity)]

            for j_idx, vel in zip(joint_indices, commanded_velocity):
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=j_idx,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=vel,
                    force=joint_force,
                )

            p.stepSimulation()

            last_feasible = commanded_next
            time.sleep(step_sleep)
    finally:
        status_gui.close()


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
        joint_count = len(re.findall(r"<joint\b", urdf_text))
        if joint_count == 0:
            print("Warning: resolved URDF has 0 joints. IK control will not work.")

    search_paths = {
        root,
        selected.parent,
        urdf_path.parent,
        *package_map.values(),
    }

    try:
        launch_end_controller(
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
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
