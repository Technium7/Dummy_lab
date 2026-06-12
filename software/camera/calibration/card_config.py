from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from high_level_jobs.subtasks.grab_sensor.transforms import pose_to_matrix

GraspPoint = Literal["corner_tip", "origin_corner_tip", "link_side_edge", "edge_center", "far_corner_tip"]
BoardMount = Literal["gripper_xy", "gripper_xz", "gripper_yz"]

# TCP is 6.5 cm along link6 −Y from the wrist joint frame.
DEFAULT_END_LINK_TO_TCP_AXIS = (0.0, -1.0, 0.0)
DEFAULT_END_LINK_TO_TCP_DISTANCE_M = 0.065
# Optional URDF mesh origin (used only with --tcp-from-urdf-visual).
URDF_LINK6_VISUAL_XYZ_M = (0.001009, 0.106879, -0.172457)


def end_link_tcp_offset_m(
    direction_xyz: Sequence[float] | None = None,
    *,
    distance_m: float = DEFAULT_END_LINK_TO_TCP_DISTANCE_M,
) -> tuple[float, float, float]:
    """Unit direction in the end-link frame, scaled to ``distance_m`` (rotates with the link)."""
    if float(distance_m) == 0.0:
        return (0.0, 0.0, 0.0)
    direction = np.asarray(
        DEFAULT_END_LINK_TO_TCP_AXIS if direction_xyz is None else direction_xyz,
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("TCP direction must be non-zero.")
    scaled = direction * (float(distance_m) / norm)
    return (float(scaled[0]), float(scaled[1]), float(scaled[2]))


DEFAULT_END_LINK_TO_TCP_XYZ_M = (0.0, -0.065, 0.0)


def grasp_offset_in_board_frame(
    *,
    squares_x: int,
    squares_y: int,
    square_size_m: float,
    grasp_point: GraspPoint = "corner_tip",
) -> tuple[float, float, float]:
    """
    Vector from checkerboard origin to gripper TCP, expressed in the board frame.

    corner_tip: outer corner on the wrist-facing side (+board X toward the link). Default.
    origin_corner_tip: outer corner nearest the OpenCV pattern origin (old/wrong end for gripper_yz).
    link_side_edge: midpoint of the outer edge on the wrist-facing side.
    edge_center: midpoint of the outer edge below the pattern origin row.
    far_corner_tip: alias for origin_corner_tip (pattern-origin side corner).
    """
    s = float(square_size_m)
    inner_x = max(1, squares_x - 1)
    inner_y = max(1, squares_y - 1)
    if grasp_point == "corner_tip":
        return (inner_x * s, inner_y * s, 0.0)
    if grasp_point == "origin_corner_tip":
        return (s, s, 0.0)
    if grasp_point == "link_side_edge":
        return (inner_x * s, 0.5 * s * (inner_y - 1), 0.0)
    if grasp_point == "edge_center":
        half_w = 0.5 * squares_x * s
        return (half_w, 0.0, 0.0)
    if grasp_point == "far_corner_tip":
        return (s, s, 0.0)
    raise ValueError(f"Unknown grasp_point: {grasp_point!r}")


def default_gripper_to_board_rpy_deg(
    board_mount: BoardMount = "gripper_yz",
) -> tuple[float, float, float]:
    """
    Board frame orientation relative to the end link.

    gripper_xy: board lies in the gripper XY plane (pattern Y -> gripper Y).
    gripper_xz: board lies in the gripper XZ plane (pattern Y -> gripper Z).
    gripper_yz: board lies in the gripper YZ plane (pattern X -> gripper Y, pattern Y -> gripper −Z).
    """
    if board_mount == "gripper_xy":
        return (0.0, 0.0, 0.0)
    if board_mount == "gripper_xz":
        return (-90.0, 0.0, 0.0)
    if board_mount == "gripper_yz":
        return (-90.0, 0.0, 90.0)
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
    # Board origin in TCP frame: tcp sits at board_origin + offset in board frame.
    translated = -(rot @ offset_board)
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
    board_mount: BoardMount = "gripper_yz"
    grasp_point: GraspPoint = "corner_tip"
    end_link_to_tcp_xyz_m: tuple[float, float, float] = DEFAULT_END_LINK_TO_TCP_XYZ_M
    end_link_to_tcp_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_to_board_xyz_m: tuple[float, float, float] = (0.0, -0.091152, 0.091152)
    gripper_to_board_rpy_deg: tuple[float, float, float] = (-90.0, 0.0, 90.0)

    @classmethod
    def from_squares(
        cls,
        squares_x: int = 10,
        squares_y: int = 10,
        square_size_m: float = 0.010128,
        board_mount: BoardMount = "gripper_yz",
        grasp_point: GraspPoint = "corner_tip",
        end_link_to_tcp_xyz_m: Sequence[float] | None = None,
        end_link_to_tcp_distance_m: float | None = None,
        end_link_to_tcp_rpy_deg: Sequence[float] | None = None,
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
        if end_link_to_tcp_xyz_m is None:
            end_link_to_tcp_xyz_m = end_link_tcp_offset_m(
                distance_m=(
                    DEFAULT_END_LINK_TO_TCP_DISTANCE_M
                    if end_link_to_tcp_distance_m is None
                    else float(end_link_to_tcp_distance_m)
                )
            )
        if end_link_to_tcp_rpy_deg is None:
            end_link_to_tcp_rpy_deg = (0.0, 0.0, 0.0)
        return cls(
            board_mount=board_mount,
            grasp_point=grasp_point,
            inner_corners_x=max(1, squares_x - 1),
            inner_corners_y=max(1, squares_y - 1),
            square_size_m=float(square_size_m),
            end_link_to_tcp_xyz_m=tuple(float(v) for v in end_link_to_tcp_xyz_m),
            end_link_to_tcp_rpy_deg=tuple(float(v) for v in end_link_to_tcp_rpy_deg),
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

    def tcp_to_board_matrix(self) -> np.ndarray:
        """Transform from gripper TCP (pinch point) to checkerboard origin."""
        rpy_rad = np.radians(np.asarray(self.gripper_to_board_rpy_deg, dtype=np.float64))
        return pose_to_matrix(self.gripper_to_board_xyz_m, rpy_rad)

    def end_link_to_tcp_matrix(self) -> np.ndarray:
        """Transform from PyBullet end-link frame to gripper TCP (translation in link frame)."""
        rpy_rad = np.radians(np.asarray(self.end_link_to_tcp_rpy_deg, dtype=np.float64))
        return pose_to_matrix(self.end_link_to_tcp_xyz_m, rpy_rad)

    def gripper_to_board_matrix(self) -> np.ndarray:
        """Full transform from PyBullet end link to checkerboard origin."""
        return self.end_link_to_tcp_matrix() @ self.tcp_to_board_matrix()

    def board_point_in_link_frame_m(
        self,
        point_board: Sequence[float],
    ) -> tuple[float, float, float]:
        """Map a board-frame point into the end-link frame (board extends along link Y/Z for gripper_yz)."""
        transform = self.gripper_to_board_matrix()
        point = np.asarray(point_board, dtype=np.float64).reshape(3)
        mapped = transform[:3, :3] @ point + transform[:3, 3]
        return (float(mapped[0]), float(mapped[1]), float(mapped[2]))

    @property
    def board_origin_in_link_frame_m(self) -> tuple[float, float, float]:
        """OpenCV pattern origin (first inner corner) in the end-link frame."""
        return self.board_point_in_link_frame_m((0.0, 0.0, 0.0))

    def board_outline_corners_board_frame(
        self,
    ) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        """Outer physical edge and inner OpenCV pattern rectangle in the board frame."""
        s = float(self.square_size_m)
        cols, rows = self.pattern_size
        outer = [
            (-s, -s, 0.0),
            (cols * s, -s, 0.0),
            (cols * s, rows * s, 0.0),
            (-s, rows * s, 0.0),
        ]
        inner = [
            (0.0, 0.0, 0.0),
            ((cols - 1) * s, 0.0, 0.0),
            ((cols - 1) * s, (rows - 1) * s, 0.0),
            (0.0, (rows - 1) * s, 0.0),
        ]
        return outer, inner

    @classmethod
    def from_cli_or_json(
        cls,
        *,
        squares_x: int = 10,
        squares_y: int = 10,
        square_size_m: float = 0.010128,
        board_mount: BoardMount = "gripper_yz",
        grasp_point: GraspPoint = "corner_tip",
        end_link_to_tcp_xyz_m: Sequence[float] | None = None,
        end_link_to_tcp_distance_m: float | None = None,
        gripper_to_board_xyz_m: Sequence[float] | None = None,
        gripper_to_board_rpy_deg: Sequence[float] | None = None,
        json_block: dict | None = None,
    ) -> CalibrationCardConfig:
        """
        Build the card model used by calibration capture and the PyBullet preview.

        JSON may override board dimensions and mount presets only; TCP and board offsets
        are always recomputed from the current geometry rules (not stale saved xyz).
        """
        block = json_block or {}
        return cls.from_squares(
            squares_x=int(block.get("squares_x", squares_x)),
            squares_y=int(block.get("squares_y", squares_y)),
            square_size_m=float(block.get("square_size_m", square_size_m)),
            board_mount=block.get("board_mount", board_mount),
            grasp_point=block.get("grasp_point", grasp_point),
            end_link_to_tcp_xyz_m=end_link_to_tcp_xyz_m,
            end_link_to_tcp_distance_m=end_link_to_tcp_distance_m,
            gripper_to_board_xyz_m=gripper_to_board_xyz_m,
            gripper_to_board_rpy_deg=gripper_to_board_rpy_deg,
        )
