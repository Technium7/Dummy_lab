from __future__ import annotations

import sys
import time
from pathlib import Path

_SOFTWARE_DIR = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from robot_ascii_interface import RobotAsciiInterface


def connect_robot_for_teleop(
    *,
    port: str,
    baudrate: int,
    timeout: float,
    sim_only: bool = False,
    debug_serial: bool = False,
    joint_current_limit_a: float = 0.0,
) -> RobotAsciiInterface | None:
    """
    Open serial, apply Linux low-latency tty settings, and run the standard init sequence.

    Returns None in sim-only mode or when the port cannot be opened (PyBullet-only fallback).
    """
    if sim_only:
        print("Sim-only mode: PyBullet teleop preview, no serial connection.")
        return None

    robot = RobotAsciiInterface(port=port, baudrate=baudrate, timeout=timeout, debug_serial=debug_serial)
    if not robot.connect():
        print(
            f"Warning: could not open robot at {port}. "
            "Continuing in PyBullet-only mode (no hardware streaming).",
            file=sys.stderr,
        )
        return None

    print(f"[Robot] START -> {robot.start()}")
    time.sleep(0.15)
    if joint_current_limit_a > 0.0:
        for joint, resp in robot.set_arm_current_limits(joint_current_limit_a):
            print(f"[Robot] I_LIMIT_J {joint} {joint_current_limit_a:g}A -> {resp}")
        time.sleep(0.1)
    print(f"[Robot] CMDMODE 2 -> {robot.set_command_mode(2)}")
    time.sleep(0.2)
    print(f"[Robot] HAND_EN -> {robot.hand_enable()}")
    time.sleep(0.1)
    joints = robot.get_joint_pos()
    if joints is not None:
        print(f"[Robot] GETJPOS -> {[round(v, 2) for v in joints[:6]]}")
    else:
        print("[Robot] Warning: GETJPOS failed after init.", file=sys.stderr)
    return robot


def disconnect_robot(robot: RobotAsciiInterface | None) -> None:
    if robot is None:
        return
    try:
        print(f"[Robot] STOP -> {robot.stop()}")
    except Exception:
        pass
    robot.disconnect()
