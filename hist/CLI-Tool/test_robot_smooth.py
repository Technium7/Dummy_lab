#!/usr/bin/env python3
"""Trajectory tracker with frozen-reference-time lag handling.

Control rule (per loop):
- Sample trajectory at reference time t_ref: x_traj(t_ref), v_traj(t_ref)
- If robot is sufficiently on-track, send cmd = x_fb + LOOKAHEAD_GAIN * v_traj * dt and advance t_ref
- Else, keep sending previous cmd and DO NOT advance t_ref
"""

from dataclasses import dataclass
from robot_ascii_interface import RobotAsciiInterface
import math
import time


HOME_POSITION = [0, 0, 90, 0, 0, 0]
TARGET_POSITION = [60, -40, 60, 20, 0, 0]
MIN_SPEED = 10.0      # deg/s (joint space)
MAX_SPEED = 30.0      # deg/s (joint space)
ACCELERATION = 15.0   # deg/s^2 (joint space)
CONTROL_DT_MS = 200

# From firmware DummyRobot ctor: reductions for J1..J6
JOINT_REDUCTION = [30.0, 30.0, 30.0, 24.0, 30.0, 50.0]
DOMINANT_REDUCTION = max(JOINT_REDUCTION)

LOG_INTERVAL = 0.4
LOOKAHEAD_GAIN = 8.0
GATE_TOLERANCE_DEG = 0.25  # prevents freeze due to sensor noise / tiny mismatch
PRINT_LOG = False  # Set False to disable intermediate trajectory logs.


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
    v_peak: float
    accel: float
    pre_pad: float
    post_pad: float


def fmt(pos):
    if not pos:
        return "N/A"
    return ", ".join(f"{v:.2f}" for v in pos)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def parse_joint_input(raw_text):
    tokens = raw_text.replace(",", " ").split()
    if len(tokens) != 6:
        raise ValueError("Please enter exactly 6 values (J1..J6).")
    return [float(token) for token in tokens]


def prompt_next_target(default_position):
    prompt = (
        "Next target J1..J6 (comma or space separated). "
        f"Press Enter for default [{fmt(default_position)}], or type 'q' to quit: "
    )
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            print("[Input] No interactive input detected, stopping target loop.")
            return None

        if not raw:
            return default_position[:]

        if raw.lower() in {"q", "quit", "exit"}:
            return None

        try:
            return parse_joint_input(raw)
        except ValueError as exc:
            print(f"[Input] {exc}")


def deg_speed_to_ascii_speed(v_deg_per_s_list):
    """
    Convert per-joint deg/s velocity demand into firmware ASCII speed scalar.
    ascii ~= max_j(|v_j| * reduction_j) / 36
    """
    demand = max(abs(v_deg_per_s_list[j]) * JOINT_REDUCTION[j] for j in range(6)) / 36.0

    red_max = DOMINANT_REDUCTION
    min_ascii = MIN_SPEED * red_max / 36.0
    max_ascii = MAX_SPEED * red_max / 36.0
    return clamp(demand, min_ascii, max_ascii)


def make_joint_profile(start, target, accel, vmax, joint_index):
    reduction_scale = DOMINANT_REDUCTION / JOINT_REDUCTION[joint_index]
    effective_vmax = vmax * reduction_scale
    effective_accel = accel * reduction_scale

    distance = abs(target - start)
    direction = 1.0 if target >= start else -1.0

    if distance < 1e-6:
        return JointProfile(start, target, direction, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effective_accel, 0.0, 0.0)

    t_to_vmax = effective_vmax / effective_accel
    d_acc_to_vmax = 0.5 * effective_accel * t_to_vmax * t_to_vmax

    if 2.0 * d_acc_to_vmax >= distance:
        t_acc = math.sqrt(distance / effective_accel)
        t_flat = 0.0
        v_peak = effective_accel * t_acc
    else:
        t_acc = t_to_vmax
        d_flat = distance - 2.0 * d_acc_to_vmax
        t_flat = d_flat / effective_vmax
        v_peak = effective_vmax

    t_dec = t_acc
    t_total = t_acc + t_flat + t_dec
    return JointProfile(start, target, direction, distance, t_acc, t_flat, t_dec, t_total, v_peak, effective_accel, 0.0, 0.0)


def add_symmetric_padding(profiles):
    global_total = max(profile.t_total for profile in profiles)
    for profile in profiles:
        pad = max(0.0, global_total - profile.t_total)
        profile.pre_pad = 0.5 * pad
        profile.post_pad = 0.5 * pad
    return global_total


def sample_local_profile(profile, tau):
    accel = profile.accel

    if tau <= 0.0:
        return 0.0, 0.0
    if tau >= profile.t_total:
        return profile.distance, 0.0

    d_acc = 0.5 * accel * profile.t_acc * profile.t_acc
    d_flat = profile.v_peak * profile.t_flat

    if tau < profile.t_acc:
        velocity = accel * tau
        position = 0.5 * accel * tau * tau
    elif tau < profile.t_acc + profile.t_flat:
        dt = tau - profile.t_acc
        velocity = profile.v_peak
        position = d_acc + profile.v_peak * dt
    else:
        dt = tau - profile.t_acc - profile.t_flat
        velocity = max(0.0, profile.v_peak - accel * dt)
        position = d_acc + d_flat + profile.v_peak * dt - 0.5 * accel * dt * dt

    return position, velocity


def sample_joint_at_time(profile, t_global):
    if t_global < profile.pre_pad:
        return profile.start, 0.0

    t_motion_end = profile.pre_pad + profile.t_total
    if t_global > t_motion_end:
        return profile.target, 0.0

    tau = t_global - profile.pre_pad
    x_rel, v_mag = sample_local_profile(profile, tau)
    x = profile.start + profile.direction * x_rel
    v = profile.direction * v_mag
    return x, v


def is_on_track(feedback, x_traj, v_traj, dt_sec, direction):
    """Direction-aware lag gate derived from user's condition."""
    threshold = max(LOOKAHEAD_GAIN * abs(v_traj) * dt_sec, GATE_TOLERANCE_DEG)
    if direction >= 0:
        return feedback >= (x_traj - threshold)
    return feedback <= (x_traj + threshold)


def make_cmd_from_feedback(feedback, v_traj, dt_sec, target, direction):
    cmd = feedback + LOOKAHEAD_GAIN * v_traj * dt_sec
    if direction >= 0:
        return min(cmd, target)
    return max(cmd, target)


def stop_and_hold(robot):
    print("[Shutdown] Stop and hold")
    print(f"           STOP: {robot.stop()}")
    time.sleep(0.12)
    print(f"           MODE2: {robot.set_command_mode(2)}")
    time.sleep(0.12)

    pos = robot.get_joint_pos()
    if pos and len(pos) >= 6:
        robot.move_joint(
            pos[0], pos[1], pos[2], pos[3], pos[4], pos[5],
            speed=max(MIN_SPEED, 6.0),
            wait_ms=200,
        )


def run_tracking(robot, target_position, leg_name="TRACK"):
    dt_sec = CONTROL_DT_MS / 1000.0

    current = robot.get_joint_pos()
    if not current or len(current) < 6:
        raise RuntimeError("Failed to read current joint position")

    profiles = [
        make_joint_profile(current[j], target_position[j], ACCELERATION, MAX_SPEED, j)
        for j in range(6)
    ]
    total_runtime = add_symmetric_padding(profiles)

    if PRINT_LOG:
        print(f"\n=== {leg_name}: Planned runtime per joint (s) ===")
        for j, profile in enumerate(profiles, start=1):
            print(
                f"J{j}: total={profile.t_total:.3f}, "
                f"pad(pre/post)=({profile.pre_pad:.3f}/{profile.post_pad:.3f}), "
                f"v_peak={profile.v_peak:.2f}"
            )
        print(f"Global synchronized runtime: {total_runtime:.3f}s\n")

    wall_start = time.perf_counter()
    next_tick = wall_start
    last_log = -LOG_INTERVAL

    # Bootstrap with one-step-ahead command so HOLD state still has a forward-driving command.
    t_ref = dt_sec
    x_boot = []
    v_boot = []
    for profile in profiles:
        x_value, v_value = sample_joint_at_time(profile, t_ref)
        x_boot.append(x_value)
        v_boot.append(v_value)

    last_cmd = []
    for j in range(6):
        cmd_j = make_cmd_from_feedback(
            feedback=current[j],
            v_traj=v_boot[j],
            dt_sec=dt_sec,
            target=target_position[j],
            direction=profiles[j].direction,
        )
        low = min(profiles[j].start, target_position[j])
        high = max(profiles[j].start, target_position[j])
        last_cmd.append(clamp(cmd_j, low, high))

    frozen_cycles = 0

    while True:
        now = time.perf_counter()
        wall_elapsed = now - wall_start

        # End when reference trajectory fully consumed and close to target
        if t_ref >= total_runtime:
            feedback_now = robot.get_joint_pos() or last_cmd
            remain = max(abs(target_position[j] - feedback_now[j]) for j in range(6))
            if remain <= 0.6:
                break
            if wall_elapsed > total_runtime + 2.0:
                break

        if now < next_tick:
            time.sleep(next_tick - now)

        feedback = robot.get_joint_pos()
        if not feedback or len(feedback) < 6:
            feedback = last_cmd

        x_traj = []
        v_traj = []
        for profile in profiles:
            x, v = sample_joint_at_time(profile, t_ref)
            x_traj.append(x)
            v_traj.append(v)

        on_track_all = True
        for j in range(6):
            if not is_on_track(feedback[j], x_traj[j], v_traj[j], dt_sec, profiles[j].direction):
                on_track_all = False
                break

        if on_track_all:
            cmd = []
            for j in range(6):
                cmd_j = make_cmd_from_feedback(
                    feedback=feedback[j],
                    v_traj=v_traj[j],
                    dt_sec=dt_sec,
                    target=target_position[j],
                    direction=profiles[j].direction,
                )
                low = min(profiles[j].start, target_position[j])
                high = max(profiles[j].start, target_position[j])
                cmd.append(clamp(cmd_j, low, high))

            last_cmd = cmd
            t_ref += dt_sec
            frozen_flag = "RUN"
        else:
            # Hold last command and freeze trajectory reference time
            cmd = last_cmd
            frozen_cycles += 1
            frozen_flag = "HOLD"

        speed_cmd = deg_speed_to_ascii_speed(v_traj)
        ok = robot.send_trajectory_point_async(
            cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], speed_cmd
        )
        if not ok:
            print("Send failed, aborting.")
            break

        if PRINT_LOG and (wall_elapsed - last_log >= LOG_INTERVAL):
            print(
                f"tw={wall_elapsed:5.2f}s | tr={t_ref:5.2f}s | {frozen_flag} | "
                f"fb=[{fmt(feedback)}] | traj=[{fmt(x_traj)}] | cmd=[{fmt(cmd)}] | spd={speed_cmd:5.2f}"
            )
            last_log = wall_elapsed

        next_tick += dt_sec

    print(f"\n[Finish:{leg_name}] Sending exact final target")
    robot.move_joint(
        target_position[0], target_position[1], target_position[2],
        target_position[3], target_position[4], target_position[5],
        speed=MIN_SPEED,
        wait_ms=250,
    )
    robot.wait_for_motion_complete(timeout_sec=8)
    print(f"Final position: {fmt(robot.get_joint_pos())}")
    print(f"Frozen cycles: {frozen_cycles}")


def run():
    selected_home = HOME_POSITION[:]
    default_target = TARGET_POSITION[:]

    robot = RobotAsciiInterface()
    if not robot.connect():
        return

    try:
        print("\n[Init] STOP")
        print(f"       {robot.stop()}")
        time.sleep(0.2)

        print("[Init] START")
        print(f"       {robot.start()}")
        time.sleep(0.2)

        print("[Init] CMDMODE 2 (Interruptable)")
        print(f"       {robot.set_command_mode(2)}")
        time.sleep(0.2)

        print(f"[Init] Move to HOME: {fmt(selected_home)}")
        robot.move_joint(
            selected_home[0], selected_home[1], selected_home[2],
            selected_home[3], selected_home[4], selected_home[5],
            speed=MIN_SPEED,
            wait_ms=250,
        )
        robot.wait_for_motion_complete(timeout_sec=10)
        print(f"[Init] At HOME: {fmt(robot.get_joint_pos())}")

        pos = robot.get_joint_pos()
        print(f"[Init] Current position: {fmt(pos)}")
        print(f"[Init] Home position   : {fmt(selected_home)}")
        print(f"[Init] Default target  : {fmt(default_target)}")
        print(
            f"[Init] Inputs: min_speed={MIN_SPEED}, max_speed={MAX_SPEED}, "
            f"acc={ACCELERATION}, dt={CONTROL_DT_MS}ms"
        )
        print("[Run] Enter targets one by one. Type 'q' to stop.")

        target_index = 1
        while True:
            selected_target = prompt_next_target(default_target)
            if selected_target is None:
                print("[Run] Target loop ended by user.")
                print(f"[Run] Returning to HOME: {fmt(selected_home)}")
                robot.move_joint(
                    selected_home[0], selected_home[1], selected_home[2],
                    selected_home[3], selected_home[4], selected_home[5],
                    speed=MIN_SPEED,
                    wait_ms=250,
                )
                robot.wait_for_motion_complete(timeout_sec=10)
                print(f"[Run] At HOME: {fmt(robot.get_joint_pos())}")
                break

            print(f"[Run] Target #{target_index}: {fmt(selected_target)}")
            run_tracking(robot, selected_target, leg_name=f"MOVE #{target_index}")

            default_target = selected_target[:]
            target_index += 1

    finally:
        try:
            stop_and_hold(robot)
        finally:
            robot.disconnect()


if __name__ == "__main__":
    run()
