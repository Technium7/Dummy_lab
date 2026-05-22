from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


def _v(ctx: Mapping[str, Any], msg: str) -> None:
    if ctx.get("verbose"):
        print(msg)


def _get_or_create_robot(ctx: dict[str, Any], port: str, baudrate: int):
    from robot_ascii_interface import RobotAsciiInterface

    robot = ctx.get("robot")
    if robot is None:
        robot = RobotAsciiInterface(port=port, baudrate=baudrate)
        ctx["robot"] = robot
    return robot


def robot_check_connected(params: Mapping[str, Any], ctx: dict[str, Any]) -> None:
    """
    Ensure the serial link is open. Params (all optional):
      port, baudrate — override ctx robot_port / robot_baudrate
      auto_connect — default True: call connect() if not open
    """
    port = str(params.get("port") or ctx.get("robot_port") or "COM3")
    baud = int(params.get("baudrate") or ctx.get("robot_baudrate") or 115200)
    auto = bool(params.get("auto_connect", True))

    if ctx.get("dry_run"):
        _v(ctx, f"[dry-run] robot_check_connected: would connect to {port} @ {baud}")
        return

    robot = _get_or_create_robot(ctx, port, baud)
    if robot.ser and robot.ser.is_open:
        _v(ctx, f"Robot serial already open ({port}).")
        return
    if not auto:
        raise RuntimeError(f"Serial not open to {port} and auto_connect=false")
    if not robot.connect():
        raise RuntimeError(f"Failed to open serial port {port!r} (is the cable plugged in and port correct?)")
    _v(ctx, f"Connected to robot on {port}.")


def robot_ensure_enabled(params: Mapping[str, Any], ctx: dict[str, Any]) -> None:
    """
    Ensure drives are enabled by sending !START (firmware: SetEnable(true)).
    Requires an open serial session (run robot_check_connected first).
    """
    if ctx.get("dry_run"):
        _v(ctx, "[dry-run] robot_ensure_enabled: would send !START")
        return

    robot = ctx.get("robot")
    if robot is None or not robot.ser or not robot.ser.is_open:
        raise RuntimeError("Robot not connected; add a robot_check_connected step before robot_ensure_enabled")

    resp = robot.start()
    if resp is None:
        raise RuntimeError("!START returned no response (timeout or serial error)")
    if "ok" not in resp.lower():
        raise RuntimeError(f"!START unexpected response: {resp!r}")
    _v(ctx, f"Robot enabled (!START): {resp}")


def robot_home(params: Mapping[str, Any], ctx: dict[str, Any]) -> None:
    """
    Send firmware !HOME (Homing to controller home pose).
    Requires connected + enabled serial session.
    Params (optional):
      command_wait_ms — pre-read delay before polling for ``!HOME`` reply (default 1500).
      read_extra_sec — extra poll time after that (default 30 seconds).
      wait_motion — if true, poll until joints stable after command (default False)
    """
    if ctx.get("dry_run"):
        _v(ctx, "[dry-run] robot_home: would send !HOME")
        return

    robot = ctx.get("robot")
    if robot is None or not robot.ser or not robot.ser.is_open:
        raise RuntimeError("Robot not connected")

    cmd_wait = int(params.get("command_wait_ms", 1500))
    extra_read = float(params.get("read_extra_sec", 30.0))
    resp = robot.home(wait_ms=cmd_wait, extra_read_sec=extra_read)
    if resp is None:
        raise RuntimeError("!HOME returned no response")
    if "ok" not in resp.lower():
        raise RuntimeError(f"!HOME unexpected response: {resp!r}")
    _v(ctx, f"Home command sent: {resp}")

    if bool(params.get("wait_motion", False)):
        ok = robot.wait_for_motion_complete(timeout_sec=float(params.get("motion_timeout_sec", 60.0)))
        if not ok:
            raise RuntimeError("Timed out waiting for motion to complete after !HOME")
        _v(ctx, "Motion settle complete.")


ROBOT_SUBTASKS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], None]] = {
    "robot_check_connected": robot_check_connected,
    "robot_ensure_enabled": robot_ensure_enabled,
    "robot_home": robot_home,
}
