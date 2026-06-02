"""Default robot description paths and cross-platform runtime settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# USB CDC IDs from hardware/99-dummy-robot-serial.rules (Robot Embedded Framework REF 1.0)
_DUMMY_USB_VID = 0x1209
_DUMMY_USB_PID = 0x0D32

DEFAULT_ROBOT_PACKAGE = Path("robot_design") / "Dummy_ros2_gripper_description"
DEFAULT_ROBOT_XACRO = DEFAULT_ROBOT_PACKAGE / "urdf" / "Dummy_ros2_gripper.xacro"
DEFAULT_ROBOT_NAME = "Dummy_ros2_gripper"
# Relative to repo root; passed to choose_description() as --robot hint.
DEFAULT_ROBOT = DEFAULT_ROBOT_XACRO.as_posix()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_robot_xacro() -> Path:
    return repo_root() / DEFAULT_ROBOT_XACRO


def find_dummy_robot_port() -> str | None:
    """Return the serial device path for a connected Dummy robot, or None."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    for info in list_ports.comports():
        if info.vid == _DUMMY_USB_VID and info.pid == _DUMMY_USB_PID and info.device:
            return info.device
    return None


def default_robot_port() -> str:
    """
    Default serial port for robot_ascii_interface and CLIs.

    Order: ``ROBOT_PORT`` env → auto-detect Dummy USB VID/PID → platform fallback
    (``COM3`` on Windows, ``/dev/ttyACM0`` on Linux, first ``/dev/tty.usb*`` on macOS).
    """
    env = os.environ.get("ROBOT_PORT", "").strip()
    if env:
        return env
    found = find_dummy_robot_port()
    if found:
        return found
    if sys.platform == "win32":
        return "COM3"
    if sys.platform == "darwin":
        try:
            for entry in sorted(Path("/dev").glob("tty.usb*")):
                return str(entry)
        except OSError:
            pass
        return "/dev/tty.usbmodem14101"
    return "/dev/ttyACM0"
