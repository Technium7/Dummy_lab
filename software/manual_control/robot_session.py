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

    if joint_current_limit_a > 0.0:
        print(f"[Robot] I_LIMIT_J 1-6 -> {joint_current_limit_a:g}A (before START)")
        apply_arm_current_limits(robot, joint_current_limit_a, label="")
    print(f"[Robot] START -> {robot.start()}")
    time.sleep(0.15)
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


def apply_arm_current_limits(
    robot: RobotAsciiInterface,
    current_a: float,
    *,
    label: str = "",
) -> None:
    """Set J1–J6 motor current limit via firmware ``#I_LIMIT_J`` (before ``!START``)."""
    if current_a <= 0.0:
        return
    for joint in range(1, 7):
        cmd = f"#I_LIMIT_J {joint} {current_a:g}"
        if not robot.send_command_async(cmd):
            print(f"{label}[Robot] I_LIMIT_J {joint} {current_a:g}A -> send failed")
        else:
            print(f"{label}[Robot] I_LIMIT_J {joint} {current_a:g}A -> sent")
        time.sleep(0.06)
    time.sleep(0.25)


def disconnect_robot(robot: RobotAsciiInterface | None, *, label: str = "") -> None:
    """Send ``!STOP`` then close serial (safe shutdown — do not call ``disconnect()`` alone)."""
    if robot is None:
        return
    try:
        resp = robot.stop()
        print(f"{label}[Robot] STOP -> {resp}")
    except Exception as exc:
        print(f"{label}[Robot] STOP failed: {exc}", file=sys.stderr)
    try:
        robot.disconnect()
    except Exception as exc:
        print(f"{label}[Robot] disconnect failed: {exc}", file=sys.stderr)
