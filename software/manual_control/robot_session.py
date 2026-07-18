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

    # Let CDC / motor CAN settle after open — blasting #I_LIMIT_J too early drops USB.
    time.sleep(0.35)

    # Sync targetJoints ← currentJoints before enabling, otherwise FixUpdate
    # immediately MoveJoints(stale target) — often HOME after a prior T + DISABLE.
    sync_target_to_current(robot)
    time.sleep(0.05)
    print(f"[Robot] START -> {robot.start()}")
    time.sleep(0.2)
    print(f"[Robot] CMDMODE 2 -> {robot.set_command_mode(2)}")
    time.sleep(0.25)

    # Current limits AFTER enable: each #I_LIMIT_J triggers a motor EEPROM write on CAN.
    # Doing this before START (or too fast) is what made "set current" break connect.
    if joint_current_limit_a > 0.0:
        print(f"[Robot] I_LIMIT_J 1-6 -> {joint_current_limit_a:g}A (after START)")
        apply_arm_current_limits(robot, joint_current_limit_a, label="")

    print(f"[Robot] HAND_EN -> {robot.hand_enable()}")
    time.sleep(0.1)
    joints = robot.get_joint_pos()
    if joints is not None:
        print(f"[Robot] GETJPOS -> {[round(v, 2) for v in joints[:6]]}")
    else:
        print("[Robot] Warning: GETJPOS failed after init.", file=sys.stderr)
    return robot


def sync_target_to_current(robot: RobotAsciiInterface, *, label: str = "") -> None:
    """Make firmware ``targetJoints`` match ``currentJoints`` (via ``!STOP`` / EmergencyStop).

    Needed before ``!START``: once enabled, the control loop immediately
    ``MoveJoints(targetJoints)``. After ``!DISABLE`` that target is often still HOME
    from a prior ``T``, so the arm appears to auto-home on the next launch.
    """
    try:
        if robot._session is not None:
            robot._session.discard_stream_acks(max_lines=512)
        resp = robot.stop()
        print(f"{label}[Robot] sync target←current (!STOP) -> {resp}")
    except Exception as exc:
        print(f"{label}[Robot] sync before START failed: {exc}", file=sys.stderr)


def apply_arm_current_limits(
    robot: RobotAsciiInterface,
    current_a: float,
    *,
    label: str = "",
) -> None:
    """Set J1–J6 motor current limit via firmware ``#I_LIMIT_J`` (each writes motor EEPROM)."""
    if current_a <= 0.0:
        return
    for joint in range(1, 7):
        cmd = f"#I_LIMIT_J {joint} {current_a:g}"
        # Prefer waited send so we do not pile EEPROM writes onto a busy CAN/USB link.
        resp = robot.send_command(cmd, wait_ms=120, max_read_s=0.6)
        if resp is None:
            print(f"{label}[Robot] I_LIMIT_J {joint} {current_a:g}A -> no ack")
        else:
            print(f"{label}[Robot] I_LIMIT_J {joint} {current_a:g}A -> {resp}")
        time.sleep(0.15)
    time.sleep(0.2)


def disconnect_robot(
    robot: RobotAsciiInterface | None,
    *,
    label: str = "",
    already_disabled: bool = False,
) -> None:
    """Close serial after a safe shutdown command.

    Default: ``!STOP`` (firmware EmergencyStop — holds ``currentJoints`` then soft-disable).
    If the operator already sent ``!DISABLE`` (``already_disabled=True``), skip ``!STOP``:
    EmergencyStop calls ``MoveJoints`` once and can re-torque the arm (often looking like a
    return toward home if joint state is stale).
    """
    if robot is None:
        return
    try:
        if already_disabled:
            # Keep motors torque-off; do not command a hold move.
            if robot._session is not None:
                robot._session.discard_stream_acks(max_lines=512)
            print(f"{label}[Robot] shutdown: already disabled — skip !STOP, closing serial")
        else:
            resp = robot.stop()
            print(f"{label}[Robot] STOP -> {resp}")
    except Exception as exc:
        print(f"{label}[Robot] shutdown command failed: {exc}", file=sys.stderr)
    try:
        robot.disconnect()
    except Exception as exc:
        print(f"{label}[Robot] disconnect failed: {exc}", file=sys.stderr)
