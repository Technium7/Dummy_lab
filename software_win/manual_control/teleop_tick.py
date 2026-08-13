from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple

import pybullet as p

from manual_control.ik_core import (
    compute_dls_joint_velocity,
    get_end_link_pose_robot_frame,
    is_key_down,
)
from manual_control.mapping import (
    deg_speed_to_ascii_speed,
    home_stream_ascii_speed,
    sim_rad_to_transformed_deg,
    sim_vel_rad_to_transformed_deg_s,
    transformed_deg_to_sim_rad,
    work_joints_deg,
)
from manual_control.robot_control_thread import RobotControlThread

# WASDQE: short click → one epoxy-style creep; hold past threshold → old velocity jog.
_CREEP_BINDS: Tuple[Tuple[str, int, float], ...] = (
    ("w", 1, -1.0),  # -Y
    ("s", 1, +1.0),  # +Y
    ("a", 0, +1.0),  # +X
    ("d", 0, -1.0),  # -X
    ("q", 2, +1.0),  # +Z
    ("e", 2, -1.0),  # -Z
)
_CREEP_KEY_MAP = {k: (axis, sign) for k, axis, sign in _CREEP_BINDS}


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
    # Press → creep; hold past creep_hold_threshold_s → legacy velocity jog.
    creep_enabled: bool = True
    creep_step_m: float = 0.00012
    creep_linear_speed: float = 0.00165
    creep_accel_time: float = 0.20
    creep_settle_tol: float = 1e-5
    creep_hold_threshold_s: float = 0.20


@dataclass
class ActiveCreep:
    """One-axis click step (same ramp idea as epoxy IkCartesianMover)."""

    axis: int
    target: float
    sign: float
    t0: float
    speed: float
    accel_t: float
    decel_dist: float
    tol: float
    key: str


@dataclass
class TeleopState:
    hold_target_deg: Optional[List[float]] = None
    home_latch_until_s: float = 0.0  # T → WORK stream latch
    speed_scale: float = 0.18
    was_moving: bool = False
    creep: Optional[ActiveCreep] = None
    cart_press_t: Dict[str, float] = field(default_factory=dict)
    cart_hold_mode: Set[str] = field(default_factory=set)
    cart_down_prev: Set[str] = field(default_factory=set)


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


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _start_creep(
    *,
    robot_id: int,
    end_link_idx: int,
    key: str,
    axis: int,
    sign: float,
    config: TeleopConfig,
) -> ActiveCreep:
    pos, _ = get_end_link_pose_robot_frame(robot_id, end_link_idx)
    distance = abs(float(config.creep_step_m)) * float(sign)
    speed = abs(float(config.creep_linear_speed))
    accel_t = max(float(config.creep_accel_time), 1e-3)
    tol = min(abs(float(config.creep_settle_tol)), max(abs(distance) * 0.05, 1e-6))
    decel_dist = max(speed * accel_t * 0.5, abs(distance) * 0.35, 2.0 * tol)
    return ActiveCreep(
        axis=int(axis),
        target=float(pos[axis]) + distance,
        sign=1.0 if sign > 0 else -1.0,
        t0=time.perf_counter(),
        speed=speed,
        accel_t=accel_t,
        decel_dist=decel_dist,
        tol=tol,
        key=key,
    )


def _creep_axis_velocity(
    *,
    robot_id: int,
    end_link_idx: int,
    creep: ActiveCreep,
    dt: float,
) -> Tuple[Optional[float], bool]:
    """Return (signed axis speed, done)."""
    pos, _ = get_end_link_pose_robot_frame(robot_id, end_link_idx)
    err = creep.target - float(pos[creep.axis])
    elapsed = time.perf_counter() - creep.t0
    if elapsed > 0.02 and (abs(err) <= creep.tol or creep.sign * err <= 0):
        return None, True

    ramp_up = _smoothstep(elapsed / creep.accel_t)
    ramp_down = _smoothstep(abs(err) / max(creep.decel_dist, 1e-9))
    v = creep.speed * ramp_up * ramp_down
    v = max(v, min(creep.speed * 0.08, abs(err) / max(dt, 1e-3)))
    v = min(v, abs(err) / max(dt, 1e-3))
    if v < 1e-8:
        return None, True
    return creep.sign * v, False


def _apply_creep_velocity(axis: int, axis_v: float) -> Tuple[float, float, float]:
    vx = vy = vz = 0.0
    if axis == 0:
        vx = axis_v
    elif axis == 1:
        vy = axis_v
    else:
        vz = axis_v
    return vx, vy, vz


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
    """One teleop tick: WASDQE click-creep / hold-jog, HKUJ/IY rotate, T→WORK."""
    speed_scale_min = 0.18
    speed_scale_max = 1.8
    speed_scale_rise_per_sec = 1.6
    home_stream_speed = home_stream_ascii_speed()
    angular_speed = math.radians(config.angular_speed_deg)
    now = time.perf_counter()

    use_creep = bool(config.creep_enabled) and float(config.creep_step_m) > 1e-9
    hold_thresh = max(float(config.creep_hold_threshold_s), 0.05)
    creep_key_down = any(keys.is_down(events, k) for k, _, _ in _CREEP_BINDS)
    rot_key_down = any(keys.is_down(events, k) for k in "hkujiy")

    vx = vy = vz = wx = wy = 0.0
    wrist_vel = 0.0
    active_keys: List[str] = []
    creep_active = False
    cart_hold_active = False

    if use_creep:
        # Press starts creep immediately; hold past threshold switches to velocity jog.
        down_now = {k for k, _, _ in _CREEP_BINDS if keys.is_down(events, k)}
        pressed_now: List[Tuple[str, int, float]] = []
        for key, axis, sign in _CREEP_BINDS:
            rising = keys.was_triggered(events, key) or (
                key in down_now and key not in state.cart_down_prev
            )
            if rising:
                state.cart_press_t[key] = now
                state.cart_hold_mode.discard(key)
                pressed_now.append((key, axis, sign))

        for key in list(state.cart_press_t.keys()):
            if key in down_now:
                if (now - state.cart_press_t[key]) >= hold_thresh:
                    state.cart_hold_mode.add(key)
                    if state.creep is not None and state.creep.key == key:
                        state.creep = None
            else:
                state.cart_press_t.pop(key, None)
                state.cart_hold_mode.discard(key)

        state.cart_down_prev = set(down_now)

        if pressed_now and not any(k in state.cart_hold_mode for k, _, _ in pressed_now):
            key, axis, sign = pressed_now[0]
            state.creep = _start_creep(
                robot_id=robot_id,
                end_link_idx=end_link_idx,
                key=key,
                axis=axis,
                sign=sign,
                config=config,
            )

        if state.creep is not None:
            axis_v, done = _creep_axis_velocity(
                robot_id=robot_id,
                end_link_idx=end_link_idx,
                creep=state.creep,
                dt=config.sim_dt,
            )
            if done or axis_v is None:
                state.creep = None
            else:
                creep_active = True
                active_keys.append(state.creep.key.upper())
                vx, vy, vz = _apply_creep_velocity(state.creep.axis, axis_v)

        hold_keys = {k for k in state.cart_hold_mode if k in down_now}
        if hold_keys and not creep_active:
            cart_hold_active = True
            if not state.was_moving:
                state.speed_scale = speed_scale_min
            state.speed_scale = min(
                speed_scale_max, state.speed_scale + speed_scale_rise_per_sec * config.sim_dt
            )
            linear_cmd_speed = config.linear_speed * state.speed_scale
            for key in ("w", "s", "a", "d", "q", "e"):
                if key not in hold_keys:
                    continue
                axis, sign = _CREEP_KEY_MAP[key]
                active_keys.append(key.upper())
                if axis == 0:
                    vx += sign * linear_cmd_speed
                elif axis == 1:
                    vy += sign * linear_cmd_speed
                else:
                    vz += sign * linear_cmd_speed
    else:
        moving_cart = creep_key_down or rot_key_down
        if moving_cart and not state.was_moving:
            state.speed_scale = speed_scale_min
        if moving_cart:
            state.speed_scale = min(
                speed_scale_max, state.speed_scale + speed_scale_rise_per_sec * config.sim_dt
            )
        else:
            state.speed_scale = speed_scale_min
        linear_cmd_speed = config.linear_speed * state.speed_scale
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

    angular_moving = rot_key_down
    if angular_moving and not cart_hold_active:
        if not state.was_moving:
            state.speed_scale = speed_scale_min
        state.speed_scale = min(
            speed_scale_max, state.speed_scale + speed_scale_rise_per_sec * config.sim_dt
        )
    elif not cart_hold_active and not creep_active and not angular_moving:
        state.speed_scale = speed_scale_min
    angular_cmd_speed = angular_speed * state.speed_scale

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
        work_deg = work_joints_deg()
        for local_idx, joint_idx in enumerate(joint_indices[:6]):
            p.resetJointState(
                robot_id,
                joint_idx,
                targetValue=transformed_deg_to_sim_rad(local_idx, work_deg[local_idx]),
                targetVelocity=0.0,
            )
        state.hold_target_deg = list(work_deg)
        state.home_latch_until_s = time.perf_counter() + 3.0
        state.creep = None
        state.cart_press_t.clear()
        state.cart_hold_mode.clear()

    arm_deg = _arm_joint_positions_deg(robot_id, joint_indices)
    if include_joint_telemetry:
        all_joint_positions = [float(p.getJointState(robot_id, j)[0]) for j in all_joint_indices]
        transformed_joint_positions_deg = [
            sim_rad_to_transformed_deg(idx, pos) for idx, pos in enumerate(all_joint_positions)
        ]
    else:
        transformed_joint_positions_deg = []

    moving_key_down = (
        creep_active
        or cart_hold_active
        or rot_key_down
        or (not use_creep and creep_key_down)
    )
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
    elif abs(vx) + abs(vy) + abs(vz) + abs(wx) + abs(wy) <= 1e-12:
        commanded_velocity = [0.0] * len(joint_indices)
        if len(joint_indices) >= 6:
            commanded_velocity[5] = wrist_vel
    else:
        arm_joints = list(joint_indices[:5])
        arm_weights = (1.15, 1.1, 1.05, 1.0, 1.0)[: len(arm_joints)]
        qdot = compute_dls_joint_velocity(
            robot_id,
            end_link_idx,
            arm_joints,
            [vx, vy, vz, wx, wy],
            config.ik_damping,
            config.rotation_weight,
            joint_weights=arm_weights,
        )
        commanded_velocity = [0.0] * len(joint_indices)
        if qdot is not None:
            for i, vel in enumerate(qdot):
                if i < len(commanded_velocity):
                    commanded_velocity[i] = float(vel)
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
    speed_cmd = deg_speed_to_ascii_speed(
        transformed_vel_deg_s + [0.0] * max(0, 6 - len(transformed_vel_deg_s))
    )

    if home_latched:
        robot_target_deg = work_joints_deg()
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
