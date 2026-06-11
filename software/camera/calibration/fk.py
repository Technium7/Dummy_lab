from __future__ import annotations

from typing import Sequence

import numpy as np
import pybullet as p

from high_level_jobs.subtasks.grab_sensor.transforms import pose_to_matrix


def link_pose_matrix(robot_id: int, link_idx: int) -> np.ndarray:
    state = p.getLinkState(robot_id, link_idx, computeForwardKinematics=True)
    pos = tuple(float(v) for v in state[4])
    rpy = tuple(float(v) for v in p.getEulerFromQuaternion(state[5]))
    return pose_to_matrix(pos, rpy)


def base_to_gripper_matrix(robot_id: int, end_link_idx: int) -> np.ndarray:
    return link_pose_matrix(robot_id, end_link_idx)


def base_to_board_matrix(
    robot_id: int,
    end_link_idx: int,
    gripper_to_board: np.ndarray,
) -> np.ndarray:
    t_base_gripper = base_to_gripper_matrix(robot_id, end_link_idx)
    return t_base_gripper @ gripper_to_board


def rotation_translation_from_matrix(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return transform[:3, :3].copy(), transform[:3, 3].copy()


def joints_deg_to_list(joints_deg: Sequence[float]) -> list[float]:
    return [float(v) for v in joints_deg[:6]]
