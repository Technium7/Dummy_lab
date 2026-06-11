from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from high_level_jobs.subtasks.grab_sensor.transforms import pose_to_matrix

GraspPoint = Literal["corner_tip", "edge_center", "far_corner_tip"]
BoardMount = Literal["gripper_xy", "gripper_xz"]


def grasp_offset_in_board_frame(
    *,
    squares_x: int,
    squares_y: int,
    square_size_m: float,
    grasp_point: GraspPoint = "corner_tip",
) -> tuple[float, float, float]:
    """
    Vector from checkerboard origin to gripper TCP, expressed in the board frame.

    corner_tip: pinch the outer corner nearest the pattern origin (default).
    edge_center: pinch the midpoint of the edge below the pattern origin row.
    far_corner_tip: pinch the outer corner opposite the pattern origin.
    """
    s = float(square_size_m)
    if grasp_point == "corner_tip":
        return (s, s, 0.0)
    if grasp_point == "edge_center":
        half_w = 0.5 * squares_x * s
        return (half_w, 0.0, 0.0)
    if grasp_point == "far_corner_tip":
        return (squares_x * s, squares_y * s, 0.0)
    raise ValueError(f"Unknown grasp_point: {grasp_point!r}")


def default_gripper_to_board_rpy_deg(
    board_mount: BoardMount = "gripper_xz",
) -> tuple[float, float, float]:
    """
    Board frame orientation relative to the end link.

    gripper_xy: board lies in the gripper XY plane (pattern Y -> gripper Y).
    gripper_xz: board lies in the gripper XZ plane (pattern Y -> gripper Z).
    """
    if board_mount == "gripper_xy":
        return (0.0, 0.0, 0.0)
    if board_mount == "gripper_xz":
        return (-90.0, 0.0, 0.0)
    raise ValueError(f"Unknown board_mount: {board_mount!r}")


def default_gripper_to_board_xyz(
    *,
    squares_x: int,
    squares_y: int,
    square_size_m: float,
    grasp_point: GraspPoint = "corner_tip",
    gripper_to_board_rpy_deg: Sequence[float] = (-90.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Gripper-frame translation to checkerboard origin (OpenCV first inner corner)."""
    offset_board = np.asarray(
        grasp_offset_in_board_frame(
            squares_x=squares_x,
            squares_y=squares_y,
            square_size_m=square_size_m,
            grasp_point=grasp_point,
        ),
        dtype=np.float64,
    )
    rpy_rad = np.radians(np.asarray(gripper_to_board_rpy_deg, dtype=np.float64))
    rot = pose_to_matrix((0.0, 0.0, 0.0), rpy_rad)[:3, :3]
    translated = rot @ offset_board
    return (float(translated[0]), float(translated[1]), float(translated[2]))


@dataclass(frozen=True)
class CalibrationCardConfig:
    """
    Checkerboard held in the gripper.

    A 10x10 square board with 10.128 mm cells has 9x9 inner corners (OpenCV pattern).
    """

    inner_corners_x: int = 9
    inner_corners_y: int = 9
    square_size_m: float = 0.010128
    board_mount: BoardMount = "gripper_xz"
    grasp_point: GraspPoint = "corner_tip"
    gripper_to_board_xyz_m: tuple[float, float, float] = (0.010128, 0.0, 0.010128)
    gripper_to_board_rpy_deg: tuple[float, float, float] = (-90.0, 0.0, 0.0)

    @classmethod
    def from_squares(
        cls,
        squares_x: int = 10,
        squares_y: int = 10,
        square_size_m: float = 0.010128,
        board_mount: BoardMount = "gripper_xz",
        grasp_point: GraspPoint = "corner_tip",
        gripper_to_board_xyz_m: Sequence[float] | None = None,
        gripper_to_board_rpy_deg: Sequence[float] | None = None,
    ) -> CalibrationCardConfig:
        if gripper_to_board_rpy_deg is None:
            gripper_to_board_rpy_deg = default_gripper_to_board_rpy_deg(board_mount)
        if gripper_to_board_xyz_m is None:
            gripper_to_board_xyz_m = default_gripper_to_board_xyz(
                squares_x=squares_x,
                squares_y=squares_y,
                square_size_m=square_size_m,
                grasp_point=grasp_point,
                gripper_to_board_rpy_deg=gripper_to_board_rpy_deg,
            )
        return cls(
            board_mount=board_mount,
            grasp_point=grasp_point,
            inner_corners_x=max(1, squares_x - 1),
            inner_corners_y=max(1, squares_y - 1),
            square_size_m=float(square_size_m),
            gripper_to_board_xyz_m=tuple(float(v) for v in gripper_to_board_xyz_m),
            gripper_to_board_rpy_deg=tuple(float(v) for v in gripper_to_board_rpy_deg),
        )

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.inner_corners_x, self.inner_corners_y

    @property
    def board_width_m(self) -> float:
        return (self.inner_corners_x + 1) * self.square_size_m

    @property
    def board_height_m(self) -> float:
        return (self.inner_corners_y + 1) * self.square_size_m

    def gripper_to_board_matrix(self) -> np.ndarray:
        rpy_rad = np.radians(np.asarray(self.gripper_to_board_rpy_deg, dtype=np.float64))
        return pose_to_matrix(self.gripper_to_board_xyz_m, rpy_rad)
