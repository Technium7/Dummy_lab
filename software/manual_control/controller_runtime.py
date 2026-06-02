from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pybullet as p
import pybullet_data

from ik_core import compute_dls_joint_velocity, get_controllable_joints, is_key_down, resolve_end_link
from mapping import (
    HOME_JOINTS_DEG,
    deg_speed_to_ascii_speed,
    home_stream_ascii_speed,
    sim_rad_to_transformed_deg,
    sim_vel_rad_to_transformed_deg_s,
    transformed_deg_to_sim_rad,
)
from status_window import ControlStatusWindow

_SOFTWARE_DIR = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))
from robot_ascii_interface import RobotAsciiInterface


def is_key_triggered(events: dict[int, int], char: str) -> bool:
    lower = ord(char.lower())
    upper = ord(char.upper())
    return ((events.get(lower, 0) & p.KEY_WAS_TRIGGERED) != 0) or ((events.get(upper, 0) & p.KEY_WAS_TRIGGERED) != 0)


def launch_controller(
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
    port: str,
    baudrate: int,
    timeout: float,
    control_dt_ms: float,
    home_use_firmware: bool = True,
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
    try:
        p.loadURDF(str(Path(pybullet_data.getDataPath()) / "plane.urdf"))
    except Exception:
        pass

    robot_id = p.loadURDF(str(urdf_path), useFixedBase=fixed_base, flags=p.URDF_USE_INERTIA_FROM_FILE)
    aabb_min, aabb_max = p.getAABB(robot_id, -1)
    cx = (aabb_min[0] + aabb_max[0]) * 0.5
    cy = (aabb_min[1] + aabb_max[1]) * 0.5
    cz = (aabb_min[2] + aabb_max[2]) * 0.5
    radius = max(0.2, abs(aabb_max[0] - aabb_min[0]), abs(aabb_max[1] - aabb_min[1]), abs(aabb_max[2] - aabb_min[2])) * 2.5
    p.resetDebugVisualizerCamera(cameraDistance=radius, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=[cx, cy, cz])

    joint_indices, _, _, _, joint_names = get_controllable_joints(robot_id)
    if not joint_indices:
        raise SystemExit("No controllable revolute/prismatic joints found for IK.")
    end_link_idx = resolve_end_link(robot_id, end_link_hint)
    all_joint_indices = list(range(p.getNumJoints(robot_id)))
    all_joint_names = [p.getJointInfo(robot_id, idx)[1].decode("utf-8", errors="replace") for idx in all_joint_indices]

    robot = RobotAsciiInterface(port=port, baudrate=baudrate, timeout=timeout)
    if not robot.connect():
        raise SystemExit(f"Failed to connect robot at {port}")
    print(f"[Robot] STOP -> {robot.stop()}")
    time.sleep(0.2)
    print(f"[Robot] START -> {robot.start()}")
    time.sleep(0.2)
    print(f"[Robot] CMDMODE 2 -> {robot.set_command_mode(2)}")
    time.sleep(0.2)
    print(f"[Robot] HAND_EN -> {robot.hand_enable()}")
    time.sleep(0.1)

    status_gui = ControlStatusWindow()
    status_gui.start()

    dt = 1.0 / max(1.0, float(sim_hz))
    step_sleep = dt / max(1e-9, float(speed_factor)) if float(speed_factor) > 0.0 else 0.0
    p.setPhysicsEngineParameter(fixedTimeStep=dt)
    linear_speed = max(0.0, float(linear_speed))
    angular_speed = math.radians(max(0.0, float(angular_speed_deg)))
    joint_vel_limit = max(0.0, float(joint_vel_limit))
    joint_force = max(1.0, float(joint_force))
    damping = max(1e-6, ik_damping)
    control_dt_s = max(0.02, float(control_dt_ms) / 1000.0)
    next_robot_tick = time.perf_counter()
    last_robot_send_s = time.perf_counter()
    stream_pose_deg: Optional[List[float]] = None
    home_stream_speed = home_stream_ascii_speed()

    speed_scale = 0.4
    speed_scale_min = 0.4
    speed_scale_max = 2.5
    speed_scale_rise_per_sec = 2.5
    hold_target_deg: List[float] | None = None
    home_latch_until_s = 0.0
    last_hand_open_s = 0.0
    last_hand_close_s = 0.0
    hand_cmd_min_interval_s = 0.2

    try:
        while p.isConnected():
            events = p.getKeyboardEvents()
            moving_key_down = any(
                is_key_down(events, k) for k in ("w", "s", "a", "d", "q", "e", "h", "k", "u", "j", "i", "y")
            )
            speed_scale = min(speed_scale_max, speed_scale + speed_scale_rise_per_sec * dt) if moving_key_down else speed_scale_min
            linear_cmd_speed = linear_speed * speed_scale
            angular_cmd_speed = angular_speed * speed_scale

            vx = vy = vz = wx = wy = 0.0
            wrist_vel = 0.0
            active_keys: List[str] = []
            if is_key_down(events, "w"):
                vy -= linear_cmd_speed
                active_keys.append("W")
            if is_key_down(events, "s"):
                vy += linear_cmd_speed
                active_keys.append("S")
            if is_key_down(events, "a"):
                vx -= linear_cmd_speed
                active_keys.append("A")
            if is_key_down(events, "d"):
                vx += linear_cmd_speed
                active_keys.append("D")
            if is_key_down(events, "q"):
                vz += linear_cmd_speed
                active_keys.append("Q")
            if is_key_down(events, "e"):
                vz -= linear_cmd_speed
                active_keys.append("E")
            if is_key_down(events, "h"):
                wy += angular_cmd_speed
                active_keys.append("H")
            if is_key_down(events, "k"):
                wy -= angular_cmd_speed
                active_keys.append("K")
            if is_key_down(events, "u"):
                wx += angular_cmd_speed
                active_keys.append("U")
            if is_key_down(events, "j"):
                wx -= angular_cmd_speed
                active_keys.append("J")
            if is_key_down(events, "i"):
                wrist_vel += angular_cmd_speed
                active_keys.append("I")
            if is_key_down(events, "y"):
                wrist_vel -= angular_cmd_speed
                active_keys.append("Y")
            now_s = time.perf_counter()
            if is_key_down(events, "o") and (now_s - last_hand_open_s) >= hand_cmd_min_interval_s:
                active_keys.append("O")
                robot.hand_enable()
                print(f"[Robot] HAND_O -> {robot.hand_open()}")
                last_hand_open_s = now_s
            if is_key_down(events, "p") and (now_s - last_hand_close_s) >= hand_cmd_min_interval_s:
                active_keys.append("P")
                robot.hand_enable()
                print(f"[Robot] HAND_C -> {robot.hand_close()}")
                last_hand_close_s = now_s
            if is_key_triggered(events, "t"):
                active_keys.append("T")
                for local_idx, joint_idx in enumerate(joint_indices[:6]):
                    p.resetJointState(robot_id, joint_idx, targetValue=transformed_deg_to_sim_rad(local_idx, HOME_JOINTS_DEG[local_idx]), targetVelocity=0.0)
                hold_target_deg = HOME_JOINTS_DEG[:]
                stream_pose_deg = None
                home_latch_until_s = time.perf_counter() + 25.0
                if home_use_firmware:

                    def _firmware_home() -> None:
                        try:
                            resp = robot.home(wait_ms=500, extra_read_sec=20.0)
                            print(f"[Robot] !HOME -> {resp}")
                        except Exception as exc:
                            print(f"[Robot] !HOME failed: {exc}")

                    threading.Thread(target=_firmware_home, name="robot-home", daemon=True).start()

            all_joint_positions = [float(p.getJointState(robot_id, j)[0]) for j in all_joint_indices]
            transformed_joint_positions_deg = [sim_rad_to_transformed_deg(idx, pos) for idx, pos in enumerate(all_joint_positions)]
            home_latched = time.perf_counter() < home_latch_until_s
            if home_latched:
                robot_target_deg = HOME_JOINTS_DEG[:]
            elif moving_key_down:
                hold_target_deg = None
                robot_target_deg = transformed_joint_positions_deg[:6]
                if stream_pose_deg is None:
                    stream_pose_deg = list(transformed_joint_positions_deg[:6])
            else:
                stream_pose_deg = None
                if hold_target_deg is None:
                    hold_target_deg = transformed_joint_positions_deg[:6]
                robot_target_deg = hold_target_deg

            status_gui.update(active_keys, [vx, vy, vz, wx, wy, wrist_vel], all_joint_names, transformed_joint_positions_deg)

            if abs(vx) + abs(vy) + abs(vz) + abs(wx) + abs(wy) + abs(wrist_vel) <= 1e-12:
                commanded_velocity = [0.0] * len(joint_indices)
            else:
                qdot = compute_dls_joint_velocity(
                    robot_id, end_link_idx, joint_indices, [vx, vy, vz, wx, wy], damping, rotation_weight
                )
                commanded_velocity = [float(v) for v in qdot] if qdot is not None else [0.0] * len(joint_indices)
                if joint_vel_limit > 0.0:
                    commanded_velocity = [min(max(v, -joint_vel_limit), joint_vel_limit) for v in commanded_velocity]
            if len(joint_indices) >= 6:
                commanded_velocity[5] = wrist_vel

            for j_idx, vel in zip(joint_indices, commanded_velocity):
                p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=j_idx, controlMode=p.VELOCITY_CONTROL, targetVelocity=vel, force=joint_force)
            p.stepSimulation()

            transformed_vel_deg_s = [sim_vel_rad_to_transformed_deg_s(idx, vel) for idx, vel in enumerate(commanded_velocity)]
            speed_cmd = deg_speed_to_ascii_speed(transformed_vel_deg_s + [0.0] * max(0, 6 - len(transformed_vel_deg_s)))
            now = time.perf_counter()
            if now >= next_robot_tick and len(robot_target_deg) >= 6:
                if moving_key_down and not home_latched and stream_pose_deg is not None:
                    dt_send = max(1e-6, now - last_robot_send_s)
                    for j in range(6):
                        stream_pose_deg[j] += (
                            sim_vel_rad_to_transformed_deg_s(j, commanded_velocity[j]) * dt_send
                        )
                    send_pose = stream_pose_deg
                else:
                    send_pose = robot_target_deg

                send_speed = home_stream_speed if home_latched else speed_cmd
                robot.send_trajectory_point_async(
                    send_pose[0],
                    send_pose[1],
                    send_pose[2],
                    send_pose[3],
                    send_pose[4],
                    send_pose[5],
                    send_speed,
                )
                last_robot_send_s = now
                next_robot_tick = now + control_dt_s

            time.sleep(step_sleep)
    finally:
        status_gui.close()
        try:
            print(f"[Robot] STOP -> {robot.stop()}")
        except Exception:
            pass
        robot.disconnect()

