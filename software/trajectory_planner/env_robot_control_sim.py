"""PyBullet sim: robot + static OBJ environment with end-effector IK control (movement_sim/end_controller).

Robot base is rotated 180° about Z to match the physical cell orientation.
A separate Tk window shows the end-link pose and current joint angles (deg) for plan_targets.json.

Usage — from repo root (Dummy_lab/):
  python software/trajectory_planner/env_robot_control_sim.py

  From software/:
  python trajectory_planner/env_robot_control_sim.py

Keyboard (same as end_controller.py):
  WASD / Q/E — linear velocity   H/K / U/J — angular velocity
  T — home joints [0, 0, 90, 0, 0, 0] deg (robot frame)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    raise SystemExit("Missing dependency: pybullet. Install with: pip install pybullet") from exc

from manual_control.mapping import (  # noqa: E402
    HOME_JOINTS_DEG,
    sim_rad_to_transformed_deg,
    transformed_deg_to_sim_rad,
)
from movement_sim.end_controller import (  # noqa: E402
    ControlStatusWindow,
    compute_dls_joint_velocity,
    get_controllable_joints,
    is_key_down,
    resolve_end_link,
)
from movement_sim.joint_controller import (  # noqa: E402
    build_package_map,
    choose_description,
    discover_descriptions,
    materialize_urdf,
)
from robot_config import DEFAULT_ROBOT, repo_root  # noqa: E402


# Base orientation: 180° about world Z (align sim with physical robot).
_BASE_ORIENTATION_Z_180 = p.getQuaternionFromEuler([0.0, 0.0, math.pi])


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_env = root / "physical_env" / "packaging_station_edit.obj"
    parser = argparse.ArgumentParser(
        description="Sim robot + OBJ environment with end-effector IK control."
    )
    parser.add_argument("--root", type=Path, default=root, help="Workspace root for URDF discovery.")
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT, help="URDF/Xacro path or name filter.")
    parser.add_argument("--env", type=Path, default=default_env, help="Wavefront .obj for the static scene.")
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="Uniform scale applied to the OBJ mesh.")
    parser.add_argument("--fixed-base", action="store_true", default=True, help="Load robot with fixed base.")
    parser.add_argument("--floating-base", action="store_true", help="Allow robot base to move.")
    parser.add_argument("--no-plane", action="store_true", help="Do not load PyBullet's ground plane.")
    parser.add_argument("--end-link", type=str, default="", help="End-effector link name or index.")
    parser.add_argument("--linear-speed", type=float, default=0.15, help="End-effector linear speed (m/s).")
    parser.add_argument("--angular-speed-deg", type=float, default=120.0, help="End-effector angular speed (deg/s).")
    parser.add_argument("--joint-vel-limit", type=float, default=0.0, help="Per-joint velocity limit; 0 disables.")
    parser.add_argument("--sim-hz", type=float, default=480.0, help="Physics/control loop rate in Hz.")
    parser.add_argument("--speed-factor", type=float, default=2.0, help="Wall-clock speed multiplier.")
    parser.add_argument("--rotation-weight", type=float, default=8.0, help="Angular task weight in IK solve.")
    parser.add_argument("--joint-force", type=float, default=2000.0, help="Joint velocity control force limit.")
    parser.add_argument("--ik-damping", type=float, default=0.05, help="DLS damping for Jacobian solve.")
    parser.add_argument("--status-window", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_static_obj_mesh(obj_path: Path, mesh_scale: float) -> int:
    path = obj_path.resolve()
    if not path.is_file():
        raise SystemExit(f"Environment mesh not found: {path}")
    s = max(1e-9, float(mesh_scale))
    scale = [s, s, s]
    mesh_file = str(path)
    concave = getattr(p, "GEOM_FORCE_CONCAVE_TRIMESH", 0)
    try:
        col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file, meshScale=scale, flags=concave)
    except Exception:
        col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file, meshScale=scale)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=mesh_file, meshScale=scale)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


def _aabb_union(
    boxes: Iterable[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    mins: List[Tuple[float, float, float]] = []
    maxs: List[Tuple[float, float, float]] = []
    for mn, mx in boxes:
        mins.append(mn)
        maxs.append(mx)
    if not mins:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    return (
        (min(a[0] for a in mins), min(a[1] for a in mins), min(a[2] for a in mins)),
        (max(a[0] for a in maxs), max(a[1] for a in maxs), max(a[2] for a in maxs)),
    )


def is_key_triggered(events: Dict[int, int], char: str) -> bool:
    lower = ord(char.lower())
    upper = ord(char.upper())
    return ((events.get(lower, 0) & p.KEY_WAS_TRIGGERED) != 0) or ((events.get(upper, 0) & p.KEY_WAS_TRIGGERED) != 0)


def start_end_link_status_process(script_path: Path) -> Optional[subprocess.Popen[str]]:
    cmd = [sys.executable, str(script_path), "--status-window"]
    try:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except Exception:
        return None


def get_controllable_joints_deg(robot_id: int, joint_indices: List[int]) -> List[float]:
    """First six controllable joints in transformed deg (robot frame, matches plan_targets.json)."""
    degs: List[float] = []
    for local_idx, joint_idx in enumerate(joint_indices[:6]):
        sim_rad = float(p.getJointState(robot_id, joint_idx)[0])
        degs.append(sim_rad_to_transformed_deg(local_idx, sim_rad))
    return degs


def send_end_link_status(
    proc: Optional[subprocess.Popen[str]],
    end_link_pose: Optional[Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]],
    note: str = "",
    joints_deg: Optional[Sequence[float]] = None,
) -> None:
    if proc is None or proc.stdin is None:
        return
    payload: Dict[str, object] = {"note": note, "available": end_link_pose is not None}
    if end_link_pose is not None:
        index, pos, rpy = end_link_pose
        payload.update({"index": index, "pos": list(pos), "rpy": list(rpy)})
    if joints_deg is not None:
        rounded = [round(float(v), 4) for v in joints_deg]
        payload["joints_deg"] = rounded
        payload["joints_deg_json"] = json.dumps(rounded)
    try:
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
    except Exception:
        pass


def stop_end_link_status_process(proc: Optional[subprocess.Popen[str]]) -> None:
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


def run_end_link_status_window() -> None:
    try:
        import tkinter as tk
    except Exception as exc:
        raise SystemExit(f"Tkinter is not available: {exc}") from exc

    import queue
    import threading

    root = tk.Tk()
    root.title("End Link Position (env_robot_control_sim)")
    root.geometry("520x260")
    root.attributes("-topmost", True)

    title_var = tk.StringVar(value="End Link: unavailable")
    pos_var = tk.StringVar(value="position (m): -")
    rpy_var = tk.StringVar(value="orientation rpy (rad): -")
    joints_var = tk.StringVar(value="joints (deg): -")
    joints_json_var = tk.StringVar(value="goal_joints_deg: -")
    note_var = tk.StringVar(value="")

    tk.Label(root, textvariable=title_var, font=("Helvetica", 14, "bold")).pack(pady=(10, 2))
    tk.Label(root, textvariable=pos_var, font=("Consolas", 12)).pack()
    tk.Label(root, textvariable=rpy_var, font=("Consolas", 12)).pack()
    tk.Label(root, text="Joint angles (robot frame, deg)", font=("Helvetica", 11, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
    tk.Label(root, textvariable=joints_var, font=("Consolas", 11)).pack(anchor="w", padx=12)
    tk.Label(root, textvariable=joints_json_var, font=("Consolas", 10), foreground="#004488").pack(anchor="w", padx=12, pady=(2, 0))
    tk.Label(root, textvariable=note_var, font=("Helvetica", 10), foreground="#aa0000").pack(pady=(6, 0))

    def apply_payload(payload: Dict[str, object]) -> None:
        note_var.set(str(payload.get("note", "")))
        joints = payload.get("joints_deg")
        if isinstance(joints, list) and joints:
            joints_var.set(
                "joints (deg): "
                + ", ".join(f"J{i+1}={float(v):+.2f}" for i, v in enumerate(joints))
            )
            joints_json_var.set(f"goal_joints_deg: {payload.get('joints_deg_json', json.dumps(joints))}")
        else:
            joints_var.set("joints (deg): -")
            joints_json_var.set("goal_joints_deg: -")
        if not bool(payload.get("available", False)):
            title_var.set("End Link: unavailable")
            pos_var.set("position (m): -")
            rpy_var.set("orientation rpy (rad): -")
            return
        index = int(payload.get("index", -1))
        pos = payload.get("pos", [0.0, 0.0, 0.0])
        rpy = payload.get("rpy", [0.0, 0.0, 0.0])
        title_var.set(f"End link index {index}")
        pos_var.set(f"position (m): ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
        rpy_var.set(f"orientation rpy (rad): ({rpy[0]:+.4f}, {rpy[1]:+.4f}, {rpy[2]:+.4f})")

    message_queue: queue.Queue[Optional[str]] = queue.Queue()

    def read_stdin_thread() -> None:
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    message_queue.put(None)
                    break
                message_queue.put(line.strip())
        except Exception:
            message_queue.put(None)

    threading.Thread(target=read_stdin_thread, daemon=True).start()

    def poll_messages() -> None:
        try:
            while True:
                try:
                    msg = message_queue.get_nowait()
                except queue.Empty:
                    break
                if msg is None:
                    root.destroy()
                    return
                try:
                    payload = json.loads(msg)
                    if isinstance(payload, dict):
                        apply_payload(payload)
                except Exception:
                    pass
        except Exception:
            pass
        root.after(50, poll_messages)

    root.after(50, poll_messages)
    root.mainloop()


def get_end_link_pose(robot_id: int, end_link_idx: int) -> Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]:
    state = p.getLinkState(robot_id, end_link_idx, computeForwardKinematics=True)
    position = tuple(float(v) for v in state[4])
    rotation = tuple(float(v) for v in p.getEulerFromQuaternion(state[5]))
    return end_link_idx, position, rotation


def apply_home_pose(robot_id: int, joint_indices: List[int]) -> None:
    for local_idx, joint_idx in enumerate(joint_indices[:6]):
        home_deg = HOME_JOINTS_DEG[local_idx] if local_idx < len(HOME_JOINTS_DEG) else 0.0
        sim_rad = transformed_deg_to_sim_rad(local_idx, home_deg)
        p.resetJointState(robot_id, joint_idx, targetValue=sim_rad, targetVelocity=0.0)


def run_control_sim(
    urdf_path: Path,
    env_path: Path,
    mesh_scale: float,
    fixed_base: bool,
    extra_search_paths: Iterable[Path],
    load_plane: bool,
    end_link_hint: str,
    linear_speed: float,
    angular_speed_deg: float,
    joint_vel_limit: float,
    sim_hz: float,
    speed_factor: float,
    rotation_weight: float,
    joint_force: float,
    ik_damping: float,
    status_script: Path,
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

    if load_plane:
        plane_path = Path(pybullet_data.getDataPath()) / "plane.urdf"
        if plane_path.exists():
            try:
                p.loadURDF(str(plane_path))
            except Exception:
                pass

    env_id = load_static_obj_mesh(env_path, mesh_scale)

    load_flags = p.URDF_USE_INERTIA_FROM_FILE
    robot_id = p.loadURDF(
        str(urdf_path),
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=_BASE_ORIENTATION_Z_180,
        useFixedBase=fixed_base,
        flags=load_flags,
    )

    boxes = [p.getAABB(robot_id, -1), p.getAABB(env_id, -1)]
    aabb_min, aabb_max = _aabb_union(boxes)
    cx = (aabb_min[0] + aabb_max[0]) * 0.5
    cy = (aabb_min[1] + aabb_max[1]) * 0.5
    cz = (aabb_min[2] + aabb_max[2]) * 0.5
    span = max(abs(aabb_max[0] - aabb_min[0]), abs(aabb_max[1] - aabb_min[1]), abs(aabb_max[2] - aabb_min[2]))
    p.resetDebugVisualizerCamera(
        cameraDistance=max(0.5, span) * 2.2,
        cameraYaw=50,
        cameraPitch=-28,
        cameraTargetPosition=[cx, cy, cz],
    )

    joint_indices, _, _, _, joint_names = get_controllable_joints(robot_id)
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

    print(f"Robot base rotated 180° about Z. End link [{end_link_idx}]: {end_link_name}")
    print("Keyboard: WASD/QE linear, H/K/U/J angular, T -> HOME [0, 0, 90, 0, 0, 0] deg")

    end_pose_proc = start_end_link_status_process(status_script)
    if end_pose_proc is not None:
        print("Opened separate end-link position window.")
    else:
        print("End-link position window unavailable (could not start subprocess).")

    control_gui = ControlStatusWindow()
    if control_gui.start():
        print("Opened end-controller status window (keys + joint angles).")

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

    home_latch_until_s = 0.0
    frame_idx = 0
    last_end_pose_frame = 0

    try:
        while p.isConnected():
            frame_idx += 1
            events = p.getKeyboardEvents()
            vx = vy = vz = wx = wy = 0.0
            active_keys: List[str] = []

            joints_deg = get_controllable_joints_deg(robot_id, joint_indices)

            if is_key_triggered(events, "t"):
                active_keys.append("T")
                apply_home_pose(robot_id, joint_indices)
                home_latch_until_s = time.perf_counter() + 1.0
                joints_deg = list(HOME_JOINTS_DEG[:6])
                send_end_link_status(
                    end_pose_proc,
                    get_end_link_pose(robot_id, end_link_idx),
                    note="HOME [0, 0, 90, 0, 0, 0] deg",
                    joints_deg=joints_deg,
                )

            home_latched = time.perf_counter() < home_latch_until_s

            if not home_latched:
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

            all_joint_positions = [float(p.getJointState(robot_id, j)[0]) for j in all_joint_indices]
            transformed_deg = [
                sim_rad_to_transformed_deg(idx, pos) for idx, pos in enumerate(all_joint_positions)
            ]
            control_gui.update(active_keys, [vx, vy, vz, wx, wy], all_joint_names, transformed_deg)

            if frame_idx - last_end_pose_frame >= 15:
                send_end_link_status(
                    end_pose_proc,
                    get_end_link_pose(robot_id, end_link_idx),
                    joints_deg=joints_deg,
                )
                last_end_pose_frame = frame_idx

            if home_latched or abs(vx) + abs(vy) + abs(vz) + abs(wx) + abs(wy) <= 1e-12:
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
            else:
                commanded_velocity = [float(vel) for vel in qdot]
                if joint_vel_limit > 0.0:
                    commanded_velocity = [
                        min(max(vel, -joint_vel_limit), joint_vel_limit) for vel in commanded_velocity
                    ]
                for j_idx, vel in zip(joint_indices, commanded_velocity):
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=j_idx,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=vel,
                        force=joint_force,
                    )

            p.stepSimulation()
            time.sleep(step_sleep)
    finally:
        control_gui.close()
        stop_end_link_status_process(end_pose_proc)


def main() -> None:
    args = parse_args()
    if args.status_window:
        run_end_link_status_window()
        return

    root = args.root.resolve()
    fixed_base = args.fixed_base and not args.floating_base
    candidates = discover_descriptions(root)
    selected = choose_description(candidates, args.robot, root)
    package_map = build_package_map(root)
    urdf_path, temp_file = materialize_urdf(selected, package_map)
    extra = {root, selected.parent, urdf_path.parent, *package_map.values()}
    script_path = Path(__file__).resolve()

    try:
        run_control_sim(
            urdf_path=urdf_path,
            env_path=args.env,
            mesh_scale=args.mesh_scale,
            fixed_base=fixed_base,
            extra_search_paths=extra,
            load_plane=not args.no_plane,
            end_link_hint=args.end_link,
            linear_speed=args.linear_speed,
            angular_speed_deg=args.angular_speed_deg,
            joint_vel_limit=args.joint_vel_limit,
            sim_hz=args.sim_hz,
            speed_factor=args.speed_factor,
            rotation_weight=args.rotation_weight,
            joint_force=args.joint_force,
            ik_damping=args.ik_damping,
            status_script=script_path,
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
