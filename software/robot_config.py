"""Default robot description paths for software tools."""

from __future__ import annotations

from pathlib import Path

DEFAULT_ROBOT_PACKAGE = Path("robot_design") / "Dummy_ros2_gripper_description"
DEFAULT_ROBOT_XACRO = DEFAULT_ROBOT_PACKAGE / "urdf" / "Dummy_ros2_gripper.xacro"
DEFAULT_ROBOT_NAME = "Dummy_ros2_gripper"
# Relative to repo root; passed to choose_description() as --robot hint.
DEFAULT_ROBOT = DEFAULT_ROBOT_XACRO.as_posix()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_robot_xacro() -> Path:
    return repo_root() / DEFAULT_ROBOT_XACRO
