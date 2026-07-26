"""Kinematic trajectory playback in PyBullet GUI."""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence

import numpy as np
import pybullet as p

from .compare_window import send_compare_update
from .scene import PlanningScene, get_end_link_pose, pose_error_rad, set_joint_positions


def playback_trajectory(
    scene: PlanningScene,
    trajectory: np.ndarray,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    compare_proc,
    hz: float = 40.0,
    max_joint_step_rad: float = 0.012,
    settle_s: float = 1.0,
) -> None:
    """Follow ``trajectory`` waypoints; terminal pose is ``trajectory[-1]`` (no separate goal override)."""
    dt = 1.0 / max(1.0, hz)
    traj = np.asarray(trajectory, dtype=np.float64)
    if len(traj) == 0:
        raise ValueError("Cannot play back an empty trajectory.")
    goal_arr = traj[-1]
    p.setGravity(0.0, 0.0, 0.0)
    p.setRealTimeSimulation(0)

    for j_idx in scene.joint_indices:
        p.setJointMotorControl2(
            scene.robot_id,
            jointIndex=j_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
        )

    print("Playing trajectory (kinematic). Close PyBullet GUI to exit.")
    set_joint_positions(scene, traj[0])
    pos, rpy = get_end_link_pose(scene)
    send_compare_update(compare_proc, "Settled at HOME", target_pos, target_rpy, pos, rpy)
    t_end = time.perf_counter() + settle_s
    while time.perf_counter() < t_end and p.isConnected():
        set_joint_positions(scene, traj[0])
        time.sleep(dt)

    frame = 0
    for i in range(len(traj) - 1):
        if not p.isConnected():
            break
        q0 = traj[i]
        q1 = traj[i + 1]
        delta = float(np.max(np.abs(q1 - q0)))
        n_sub = max(1, int(math.ceil(delta / max_joint_step_rad)))
        for k in range(1, n_sub + 1):
            if not p.isConnected():
                break
            t = k / n_sub
            q = (1.0 - t) * q0 + t * q1
            set_joint_positions(scene, q)
            frame += 1
            if frame % 3 == 0:
                pos, rpy = get_end_link_pose(scene)
                send_compare_update(
                    compare_proc,
                    f"Playback {i+1}/{len(traj)-1}",
                    target_pos,
                    target_rpy,
                    pos,
                    rpy,
                )
            time.sleep(dt)

    t_end = time.perf_counter() + 2.5
    while time.perf_counter() < t_end and p.isConnected():
        set_joint_positions(scene, goal_arr)
        pos, rpy = get_end_link_pose(scene)
        send_compare_update(compare_proc, "At goal", target_pos, target_rpy, pos, rpy)
        time.sleep(dt)

    pos, rpy = get_end_link_pose(scene)
    pos_err, ori_err = pose_error_rad(pos, rpy, target_pos, target_rpy)
    send_compare_update(
        compare_proc,
        "At goal",
        target_pos,
        target_rpy,
        pos,
        rpy,
        note=f"Final: {pos_err*1000:.3f} mm, {math.degrees(ori_err):.2f} deg",
    )
    print(
        f"Final end-link error: position {pos_err*1000:.3f} mm, "
        f"orientation {math.degrees(ori_err):.2f} deg"
    )

    while p.isConnected():
        set_joint_positions(scene, goal_arr)
        time.sleep(dt)
