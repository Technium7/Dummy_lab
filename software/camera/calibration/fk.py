from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pybullet as p

from high_level_jobs.subtasks.grab_sensor.transforms import pose_to_matrix

if TYPE_CHECKING:
    from camera.calibration.card_config import CalibrationCardConfig


def set_robot_joints_deg(robot_id: int, joints_deg: Sequence[float]) -> None:
    from manual_control.ik_core import get_controllable_joints
    from manual_control.mapping import transformed_deg_to_sim_rad

    joint_indices, _, _, _, _ = get_controllable_joints(robot_id)
    for local_idx, joint_idx in enumerate(joint_indices[:6]):
        p.resetJointState(
            robot_id,
            joint_idx,
            targetValue=transformed_deg_to_sim_rad(local_idx, joints_deg[local_idx]),
            targetVelocity=0.0,
        )


def end_link_tcp_xyz_from_visual(robot_id: int, end_link_idx: int) -> tuple[float, float, float] | None:
    """First visual shape local offset on the end link (matches URDF mesh origin)."""
    for shape in p.getVisualShapeData(robot_id):
        if int(shape[1]) == int(end_link_idx):
            pos = shape[5]
            return (float(pos[0]), float(pos[1]), float(pos[2]))
    return None


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


def base_to_board_from_card(
    robot_id: int,
    end_link_idx: int,
    card_cfg: CalibrationCardConfig,
) -> np.ndarray:
    """FK board pose using the same T_link_board chain as calibration capture."""
    return base_to_board_matrix(robot_id, end_link_idx, card_cfg.gripper_to_board_matrix())


def rotation_translation_from_matrix(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return transform[:3, :3].copy(), transform[:3, 3].copy()


def joints_deg_to_list(joints_deg: Sequence[float]) -> list[float]:
    return [float(v) for v in joints_deg[:6]]
