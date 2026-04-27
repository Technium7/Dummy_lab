from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple


MIN_SPEED = 10.0
MAX_SPEED = 30.0
ACCELERATION = 15.0
LOOKAHEAD_GAIN = 10.0
GATE_TOLERANCE_DEG = 0.25
JOINT_REDUCTION = [30.0, 30.0, 30.0, 24.0, 30.0, 50.0]
DOMINANT_REDUCTION = max(JOINT_REDUCTION)


@dataclass
class JointProfile:
    start: float
    target: float
    direction: float
    distance: float
    t_acc: float
    t_flat: float
    t_dec: float
    t_total: float
    v_start: float
    v_peak: float
    accel: float
    pre_pad: float
    post_pad: float


@dataclass
class PlanState:
    target: List[float]
    profiles: List[JointProfile]
    total_runtime: float
    t_ref: float
    last_cmd: List[float]
    frozen_cycles: int


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def deg_speed_to_ascii_speed(v_deg_per_s_list: Sequence[float]) -> float:
    demand = max(abs(v_deg_per_s_list[j]) * JOINT_REDUCTION[j] for j in range(6)) / 36.0
    min_ascii = MIN_SPEED * DOMINANT_REDUCTION / 36.0
    max_ascii = MAX_SPEED * DOMINANT_REDUCTION / 36.0
    return clamp(demand, min_ascii, max_ascii)


def make_joint_profile(start: float, target: float, accel: float, vmax: float, joint_index: int, v_start_signed: float) -> JointProfile:
    reduction_scale = DOMINANT_REDUCTION / JOINT_REDUCTION[joint_index]
    effective_vmax = vmax * reduction_scale
    effective_accel = accel * reduction_scale

    distance = abs(target - start)
    direction = 1.0 if target >= start else -1.0

    v0 = max(0.0, direction * v_start_signed)
    v0 = min(v0, effective_vmax)

    if distance < 1e-6:
        return JointProfile(start, target, direction, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effective_accel, 0.0, 0.0)

    t_acc_to_vmax = (effective_vmax - v0) / effective_accel
    d_acc_to_vmax = (effective_vmax * effective_vmax - v0 * v0) / (2.0 * effective_accel)
    d_dec_from_vmax = (effective_vmax * effective_vmax) / (2.0 * effective_accel)

    if d_acc_to_vmax + d_dec_from_vmax <= distance:
        v_peak = effective_vmax
        t_acc = t_acc_to_vmax
        d_flat = distance - d_acc_to_vmax - d_dec_from_vmax
        t_flat = d_flat / v_peak if v_peak > 1e-9 else 0.0
    else:
        v_peak_sq = max(0.0, effective_accel * distance + 0.5 * v0 * v0)
        v_peak = min(effective_vmax, math.sqrt(v_peak_sq))
        if v_peak < v0:
            v_peak = v0
        t_acc = (v_peak - v0) / effective_accel
        t_flat = 0.0

    t_dec = v_peak / effective_accel
    t_total = t_acc + t_flat + t_dec

    return JointProfile(start, target, direction, distance, t_acc, t_flat, t_dec, t_total, v0, v_peak, effective_accel, 0.0, 0.0)


def add_symmetric_padding(profiles: Sequence[JointProfile]) -> float:
    global_total = max(profile.t_total for profile in profiles)
    for profile in profiles:
        pad = max(0.0, global_total - profile.t_total)
        profile.pre_pad = 0.5 * pad
        profile.post_pad = 0.5 * pad
    return global_total


def sample_local_profile(profile: JointProfile, tau: float) -> Tuple[float, float]:
    accel = profile.accel
    if tau <= 0.0:
        return 0.0, profile.v_start
    if tau >= profile.t_total:
        return profile.distance, 0.0

    d_acc = profile.v_start * profile.t_acc + 0.5 * accel * profile.t_acc * profile.t_acc
    d_flat = profile.v_peak * profile.t_flat

    if tau < profile.t_acc:
        velocity = profile.v_start + accel * tau
        position = profile.v_start * tau + 0.5 * accel * tau * tau
    elif tau < profile.t_acc + profile.t_flat:
        dt = tau - profile.t_acc
        velocity = profile.v_peak
        position = d_acc + profile.v_peak * dt
    else:
        dt = tau - profile.t_acc - profile.t_flat
        velocity = max(0.0, profile.v_peak - accel * dt)
        position = d_acc + d_flat + profile.v_peak * dt - 0.5 * accel * dt * dt

    return position, velocity


def sample_joint_at_time(profile: JointProfile, t_global: float) -> Tuple[float, float]:
    if t_global < profile.pre_pad:
        return profile.start, 0.0

    t_motion_end = profile.pre_pad + profile.t_total
    if t_global > t_motion_end:
        return profile.target, 0.0

    tau = t_global - profile.pre_pad
    x_rel, v_mag = sample_local_profile(profile, tau)
    return profile.start + profile.direction * x_rel, profile.direction * v_mag


def is_on_track(feedback: float, x_traj: float, v_traj: float, dt_sec: float, direction: float) -> bool:
    threshold = max(LOOKAHEAD_GAIN * abs(v_traj) * dt_sec, GATE_TOLERANCE_DEG)
    if direction >= 0:
        return feedback >= (x_traj - threshold)
    return feedback <= (x_traj + threshold)


def make_cmd_from_feedback(feedback: float, v_traj: float, dt_sec: float, target: float, direction: float) -> float:
    cmd = feedback + LOOKAHEAD_GAIN * v_traj * dt_sec
    if direction >= 0:
        return min(cmd, target)
    return max(cmd, target)


class ContinuousTracker:
    def __init__(self, dt_sec: float, finish_tolerance: float = 0.6) -> None:
        self.dt_sec = float(dt_sec)
        self.finish_tolerance = float(finish_tolerance)
        self._plan: Optional[PlanState] = None

    def replan(self, start_position: Sequence[float], start_velocity: Sequence[float], target_position: Sequence[float]) -> None:
        profiles = [
            make_joint_profile(
                start=float(start_position[j]),
                target=float(target_position[j]),
                accel=ACCELERATION,
                vmax=MAX_SPEED,
                joint_index=j,
                v_start_signed=float(start_velocity[j]),
            )
            for j in range(6)
        ]

        total_runtime = add_symmetric_padding(profiles)
        t_ref = self.dt_sec

        v_boot: List[float] = []
        cmd_boot: List[float] = []
        for j in range(6):
            _, v_value = sample_joint_at_time(profiles[j], t_ref)
            v_boot.append(v_value)
            cmd_j = make_cmd_from_feedback(
                feedback=float(start_position[j]),
                v_traj=v_value,
                dt_sec=self.dt_sec,
                target=float(target_position[j]),
                direction=profiles[j].direction,
            )
            low = min(float(start_position[j]), float(target_position[j]))
            high = max(float(start_position[j]), float(target_position[j]))
            cmd_boot.append(clamp(cmd_j, low, high))

        self._plan = PlanState(
            target=[float(v) for v in target_position],
            profiles=profiles,
            total_runtime=total_runtime,
            t_ref=t_ref,
            last_cmd=cmd_boot,
            frozen_cycles=0,
        )

    def clear(self) -> None:
        self._plan = None

    def has_plan(self) -> bool:
        return self._plan is not None

    def step(self, feedback: Sequence[float]) -> Tuple[List[float], List[float], bool, str]:
        if self._plan is None:
            return [float(v) for v in feedback], [0.0] * 6, True, "IDLE"

        plan = self._plan
        remain = max(abs(plan.target[j] - float(feedback[j])) for j in range(6))
        if plan.t_ref >= plan.total_runtime and remain <= self.finish_tolerance:
            cmd = [float(v) for v in plan.target]
            self._plan = None
            return cmd, [0.0] * 6, True, "DONE"

        x_traj: List[float] = []
        v_traj: List[float] = []
        for profile in plan.profiles:
            x, v = sample_joint_at_time(profile, plan.t_ref)
            x_traj.append(x)
            v_traj.append(v)

        on_track = True
        for j in range(6):
            if not is_on_track(float(feedback[j]), x_traj[j], v_traj[j], self.dt_sec, plan.profiles[j].direction):
                on_track = False
                break

        if on_track:
            cmd: List[float] = []
            for j in range(6):
                cmd_j = make_cmd_from_feedback(
                    feedback=float(feedback[j]),
                    v_traj=v_traj[j],
                    dt_sec=self.dt_sec,
                    target=plan.target[j],
                    direction=plan.profiles[j].direction,
                )
                low = min(plan.profiles[j].start, plan.target[j])
                high = max(plan.profiles[j].start, plan.target[j])
                cmd.append(clamp(cmd_j, low, high))

            plan.last_cmd = cmd[:]
            plan.t_ref += self.dt_sec
            return cmd, v_traj, False, "RUN"

        plan.frozen_cycles += 1
        return plan.last_cmd[:], v_traj, False, "HOLD"
