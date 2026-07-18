from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import pybullet as p

from manual_control.ik_core import compute_dls_joint_velocity, is_key_down
from manual_control.mapping import (
    HOME_JOINTS_DEG,
    deg_speed_to_ascii_speed,
    home_stream_ascii_speed,
    sim_rad_to_transformed_deg,
    sim_vel_rad_to_transformed_deg_s,
    transformed_deg_to_sim_rad,
)
from manual_control.robot_control_thread import RobotControlThread


class KeyInput(Protocol):
    def is_down(self, events: dict[int, int], char: str) -> bool: ...

    def was_triggered(self, events: dict[int, int], char: str) -> bool: ...

    def any_movement_down(self, events: dict[int, int]) -> bool: ...


@dataclass
class TeleopConfig:
    linear_speed: float
    angular_speed_deg: float
    joint_vel_limit: float
    sim_dt: float
    rotation_weight: float
    joint_force: float
    ik_damping: float
    control_dt_s: float
    home_use_firmware: bool = False


@dataclass
class TeleopState:
    hold_target_deg: Optional[List[float]] = None
    home_latch_until_s: float = 0.0
    speed_scale: float = 0.4
    was_moving: bool = False


@dataclass
class TeleopTickResult:
    active_keys: List[str]
    twist: List[float]
    transformed_joint_positions_deg: List[float]
    robot_target_deg: List[float]
    moving_key_down: bool
    home_latched: bool
    commanded_velocity: List[float]


class PyBulletKeyInput:
    """Keyboard input from the PyBullet GUI only."""

    _MOVEMENT_KEYS = frozenset("wsadqehkujiy")

    def is_down(self, events: dict[int, int], char: str) -> bool:
        return is_key_down(events, char)

    def was_triggered(self, events: dict[int, int], char: str) -> bool:
        lower = ord(char.lower())
        upper = ord(char.upper())
        return ((events.get(lower, 0) & p.KEY_WAS_TRIGGERED) != 0) or (
            (events.get(upper, 0) & p.KEY_WAS_TRIGGERED) != 0
        )

    def any_movement_down(self, events: dict[int, int]) -> bool:
        return any(self.is_down(events, k) for k in self._MOVEMENT_KEYS)


def _arm_joint_positions_deg(robot_id: int, joint_indices: Sequence[int]) -> List[float]:
    return [
        sim_rad_to_transformed_deg(i, float(p.getJointState(robot_id, joint_indices[i])[0]))
        for i in range(min(6, len(joint_indices)))
    ]


def run_teleop_tick(
    *,
    events: dict[int, int],
    keys: KeyInput,
    state: TeleopState,
    config: TeleopConfig,
    robot_id: int,
    end_link_idx: int,
    joint_indices: Sequence[int],
    all_joint_indices: Sequence[int],
    robot_control: RobotControlThread | None,
    include_joint_telemetry: bool = True,
) -> TeleopTickResult:
    """One teleop iteration — logic matches stable Windows manual_control (64acdd4)."""
    speed_scale_min = 0.35
    speed_scale_max = 1.8
    speed_scale_rise_per_sec = 1.6
    home_stream_speed = home_stream_ascii_speed()
    angular_speed = math.radians(config.angular_speed_deg)

    moving_key_down = keys.any_movement_down(events)
    if moving_key_down and not state.was_moving:
        state.speed_scale = speed_scale_min
    state.speed_scale = (
        min(speed_scale_max, state.speed_scale + speed_scale_rise_per_sec * config.sim_dt)
        if moving_key_down
        else speed_scale_min
    )
    linear_cmd_speed = config.linear_speed * state.speed_scale
    angular_cmd_speed = angular_speed * state.speed_scale

    vx = vy = vz = wx = wy = 0.0
    wrist_vel = 0.0
    active_keys: List[str] = []

    if keys.is_down(events, "w"):
        vy -= linear_cmd_speed
        active_keys.append("W")
    if keys.is_down(events, "s"):
        vy += linear_cmd_speed
        active_keys.append("S")
    if keys.is_down(events, "a"):
        vx += linear_cmd_speed
        active_keys.append("A")
    if keys.is_down(events, "d"):
        vx -= linear_cmd_speed
        active_keys.append("D")
    if keys.is_down(events, "q"):
        vz += linear_cmd_speed
        active_keys.append("Q")
    if keys.is_down(events, "e"):
        vz -= linear_cmd_speed
        active_keys.append("E")
    if keys.is_down(events, "h"):
        wy += angular_cmd_speed
        active_keys.append("H")
    if keys.is_down(events, "k"):
        wy -= angular_cmd_speed
        active_keys.append("K")
    if keys.is_down(events, "u"):
        wx += angular_cmd_speed
        active_keys.append("U")
    if keys.is_down(events, "j"):
        wx -= angular_cmd_speed
        active_keys.append("J")
    if keys.is_down(events, "i"):
        wrist_vel += angular_cmd_speed
        active_keys.append("I")
    if keys.is_down(events, "y"):
        wrist_vel -= angular_cmd_speed
        active_keys.append("Y")

    if robot_control is not None and keys.was_triggered(events, "o"):
        active_keys.append("O")
        robot_control.request_hand_open()
    if robot_control is not None and keys.was_triggered(events, "p"):
        active_keys.append("P")
        robot_control.request_hand_close()
    if robot_control is not None and keys.was_triggered(events, "b"):
        active_keys.append("B")
        robot_control.request_disable()
    if robot_control is not None and keys.was_triggered(events, "n"):
        active_keys.append("N")
        robot_control.request_enable()
    if keys.was_triggered(events, "t"):
        active_keys.append("T")
        for local_idx, joint_idx in enumerate(joint_indices[:6]):
            p.resetJointState(
                robot_id,
                joint_idx,
                targetValue=transformed_deg_to_sim_rad(local_idx, HOME_JOINTS_DEG[local_idx]),
                targetVelocity=0.0,
            )
        state.hold_target_deg = HOME_JOINTS_DEG[:]
        state.home_latch_until_s = time.perf_counter() + 3.0
        if robot_control is not None and config.home_use_firmware:
            robot_control.request_home_firmware()

    arm_deg = _arm_joint_positions_deg(robot_id, joint_indices)
    if include_joint_telemetry:
        all_joint_positions = [float(p.getJointState(robot_id, j)[0]) for j in all_joint_indices]
        transformed_joint_positions_deg = [
            sim_rad_to_transformed_deg(idx, pos) for idx, pos in enumerate(all_joint_positions)
        ]
    else:
        transformed_joint_positions_deg = []

    home_latched = time.perf_counter() < state.home_latch_until_s
    if moving_key_down and home_latched:
        state.home_latch_until_s = 0.0
        home_latched = False
    if moving_key_down:
        state.hold_target_deg = None
    elif state.was_moving:
        state.hold_target_deg = list(arm_deg[:6])

    if abs(vx) + abs(vy) + abs(vz) + abs(wx) + abs(wy) + abs(wrist_vel) <= 1e-12:
        commanded_velocity = [0.0] * len(joint_indices)
    else:
        qdot = compute_dls_joint_velocity(
            robot_id,
            end_link_idx,
            joint_indices,
            [vx, vy, vz, wx, wy],
            config.ik_damping,
            config.rotation_weight,
        )
        commanded_velocity = [float(v) for v in qdot] if qdot is not None else [0.0] * len(joint_indices)
        if config.joint_vel_limit > 0.0:
            commanded_velocity = [
                min(max(v, -config.joint_vel_limit), config.joint_vel_limit) for v in commanded_velocity
            ]
    if len(joint_indices) >= 6:
        commanded_velocity[5] = wrist_vel

    for j_idx, vel in zip(joint_indices, commanded_velocity):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=j_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=vel,
            force=config.joint_force,
        )
    p.stepSimulation()

    transformed_vel_deg_s = [
        sim_vel_rad_to_transformed_deg_s(idx, vel) for idx, vel in enumerate(commanded_velocity)
    ]
    speed_cmd = deg_speed_to_ascii_speed(transformed_vel_deg_s + [0.0] * max(0, 6 - len(transformed_vel_deg_s)))

    if home_latched:
        robot_target_deg = HOME_JOINTS_DEG[:]
    elif moving_key_down:
        robot_target_deg = list(arm_deg[:6])
    else:
        if state.hold_target_deg is None:
            state.hold_target_deg = list(arm_deg)
        robot_target_deg = state.hold_target_deg

    if robot_control is not None and len(robot_target_deg) >= 6:
        send_speed = home_stream_speed if home_latched else speed_cmd
        robot_control.set_stream_intent(
            target_deg=robot_target_deg,
            vel_deg_s=transformed_vel_deg_s[:6],
            speed=send_speed,
            moving_key_down=moving_key_down,
            home_latched=home_latched,
            timestamp=time.perf_counter(),
        )

    state.was_moving = moving_key_down
    return TeleopTickResult(
        active_keys=active_keys,
        twist=[vx, vy, vz, wx, wy, wrist_vel],
        transformed_joint_positions_deg=transformed_joint_positions_deg,
        robot_target_deg=list(robot_target_deg),
        moving_key_down=moving_key_down,
        home_latched=home_latched,
        commanded_velocity=commanded_velocity,
    )
