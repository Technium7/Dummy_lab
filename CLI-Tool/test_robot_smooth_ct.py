#!/usr/bin/env python3
"""Continuous trajectory tracker with asynchronous target input.

Behavior:
- Homes once at startup.
- Accepts new joint targets from console at any time (no waiting for current move to finish).
- On each new target, replans from latest measured joint position and last sent per-joint velocity.
- Uses the same freeze-cycle logic as test_robot_smooth.py:
  - If on track, advance reference time.
  - If off track, hold previous command and freeze reference time.
"""

from dataclasses import dataclass
from queue import Empty, Queue
from robot_ascii_interface import RobotAsciiInterface
import math
import threading
import time


HOME_POSITION = [0, 0, 90, 0, 0, 0]
TARGET_POSITION = [60, -40, 60, 20, 0, 0]
MIN_SPEED = 10.0      # deg/s (joint space)
MAX_SPEED = 30.0      # deg/s (joint space)
ACCELERATION = 15.0   # deg/s^2 (joint space)
CONTROL_DT_MS = 150

# From firmware DummyRobot ctor: reductions for J1..J6
JOINT_REDUCTION = [30.0, 30.0, 30.0, 24.0, 30.0, 50.0]
DOMINANT_REDUCTION = max(JOINT_REDUCTION)

LOG_INTERVAL = 0.4
LOOKAHEAD_GAIN = 10.0
GATE_TOLERANCE_DEG = 0.25
PRINT_LOG = False


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


def input_worker(target_queue, stop_event, default_target):
    current_default = default_target[:]
    prompt = (
        "[Input] Enter target J1..J6 (comma/space). "
        "Press Enter to repeat previous target, or type 'q' to go HOME and quit: "
    )

    while not stop_event.is_set():
        try:
            raw = input(prompt).strip()
        except EOFError:
            target_queue.put({"type": "quit"})
            return

        if not raw:
            target_queue.put({"type": "target", "target": current_default[:]})
            continue

        if raw.lower() in {"q", "quit", "exit"}:
            target_queue.put({"type": "quit"})
            return

        try:
            target = parse_joint_input(raw)
        except ValueError as exc:
            print(f"[Input] {exc}")
            continue

        current_default = target[:]
        target_queue.put({"type": "target", "target": target})


def deg_speed_to_ascii_speed(v_deg_per_s_list):
    demand = max(abs(v_deg_per_s_list[j]) * JOINT_REDUCTION[j] for j in range(6)) / 36.0

    red_max = DOMINANT_REDUCTION
    min_ascii = MIN_SPEED * red_max / 36.0
    max_ascii = MAX_SPEED * red_max / 36.0
    return clamp(demand, min_ascii, max_ascii)


def make_joint_profile(start, target, accel, vmax, joint_index, v_start_signed):
    reduction_scale = DOMINANT_REDUCTION / JOINT_REDUCTION[joint_index]
    effective_vmax = vmax * reduction_scale
    effective_accel = accel * reduction_scale

    distance = abs(target - start)
    direction = 1.0 if target >= start else -1.0

    # Use only component of initial velocity that helps movement toward target.
    v0 = max(0.0, direction * v_start_signed)
    v0 = min(v0, effective_vmax)

    if distance < 1e-6:
        return JointProfile(
            start, target, direction, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, effective_accel,
            0.0, 0.0,
        )

    # Distances for accel (from v0 to vmax) and decel (vmax to 0)
    t_acc_to_vmax = (effective_vmax - v0) / effective_accel
    d_acc_to_vmax = (effective_vmax * effective_vmax - v0 * v0) / (2.0 * effective_accel)
    d_dec_from_vmax = (effective_vmax * effective_vmax) / (2.0 * effective_accel)

    if d_acc_to_vmax + d_dec_from_vmax <= distance:
        # Trapezoidal profile
        v_peak = effective_vmax
        t_acc = t_acc_to_vmax
        d_flat = distance - d_acc_to_vmax - d_dec_from_vmax
        t_flat = d_flat / v_peak if v_peak > 1e-9 else 0.0
    else:
        # Triangular-like profile with non-zero start velocity.
        # distance = (v_peak^2 - v0^2)/(2a) + v_peak^2/(2a)
        v_peak_sq = max(0.0, effective_accel * distance + 0.5 * v0 * v0)
        v_peak = min(effective_vmax, math.sqrt(v_peak_sq))
        if v_peak < v0:
            v_peak = v0
        t_acc = (v_peak - v0) / effective_accel
        t_flat = 0.0

    t_dec = v_peak / effective_accel
    t_total = t_acc + t_flat + t_dec

    return JointProfile(
        start, target, direction, distance,
        t_acc, t_flat, t_dec, t_total,
        v0, v_peak, effective_accel,
        0.0, 0.0,
    )


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


def build_plan(start_position, start_velocity, target_position, dt_sec):
    profiles = [
        make_joint_profile(
            start=start_position[j],
            target=target_position[j],
            accel=ACCELERATION,
            vmax=MAX_SPEED,
            joint_index=j,
            v_start_signed=start_velocity[j],
        )
        for j in range(6)
    ]

    total_runtime = add_symmetric_padding(profiles)
    t_ref = dt_sec

    x_boot = []
    v_boot = []
    for profile in profiles:
        x_value, v_value = sample_joint_at_time(profile, t_ref)
        x_boot.append(x_value)
        v_boot.append(v_value)

    cmd_boot = []
    for j in range(6):
        cmd_j = make_cmd_from_feedback(
            feedback=start_position[j],
            v_traj=v_boot[j],
            dt_sec=dt_sec,
            target=target_position[j],
            direction=profiles[j].direction,
        )
        low = min(start_position[j], target_position[j])
        high = max(start_position[j], target_position[j])
        cmd_boot.append(clamp(cmd_j, low, high))

    if PRINT_LOG:
        print(f"\n[Plan] New target: {fmt(target_position)}")
        print(f"[Plan] Start pos : {fmt(start_position)}")
        print(f"[Plan] Start vel : {fmt(start_velocity)}")
        print(f"[Plan] Runtime   : {total_runtime:.3f}s")

    return {
        "target": target_position[:],
        "profiles": profiles,
        "total_runtime": total_runtime,
        "t_ref": t_ref,
        "last_cmd": cmd_boot,
        "frozen_cycles": 0,
    }


def run_continuous_tracking(robot, home_position):
    dt_sec = CONTROL_DT_MS / 1000.0

    target_queue = Queue()
    stop_event = threading.Event()
    input_thread = threading.Thread(
        target=input_worker,
        args=(target_queue, stop_event, TARGET_POSITION),
        daemon=True,
    )
    input_thread.start()

    last_feedback = robot.get_joint_pos()
    if not last_feedback or len(last_feedback) < 6:
        raise RuntimeError("Failed to read current joint position")

    last_cmd = last_feedback[:]
    last_measured_velocity = [0.0] * 6
    last_sent_velocity = [0.0] * 6

    plan = None
    quitting = False

    next_tick = time.perf_counter()
    wall_start = next_tick
    last_log = -LOG_INTERVAL

    print("[Run] Continuous mode started.")
    print("[Run] You can input next target while robot is still moving.")

    # Reuse same finish tolerance as the non-continuous script.
    finish_tolerance = 0.6

    try:
        while True:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)

            wall_elapsed = time.perf_counter() - wall_start

            prev_feedback = last_feedback[:]
            feedback = robot.get_joint_pos()
            if not feedback or len(feedback) < 6:
                feedback = last_feedback
            last_feedback = feedback[:]

            # Estimate actual joint velocity from feedback for replanning.
            measured_velocity = [
                (feedback[j] - prev_feedback[j]) / dt_sec for j in range(6)
            ]
            # Light smoothing improves robustness against encoder noise.
            last_measured_velocity = [
                0.5 * last_measured_velocity[j] + 0.5 * measured_velocity[j]
                for j in range(6)
            ]

            latest_target = None
            got_quit = False

            # Consume all pending input; newest target wins.
            while True:
                try:
                    item = target_queue.get_nowait()
                except Empty:
                    break

                if item.get("type") == "quit":
                    got_quit = True
                elif item.get("type") == "target":
                    latest_target = item.get("target")

            if got_quit and not quitting:
                quitting = True
                latest_target = home_position[:]
                print("[Run] Quit requested, replanning to HOME.")

            if latest_target is not None:
                # Send stop command to current position with minimum speed.
                stop_speed = deg_speed_to_ascii_speed([MIN_SPEED] * 6)
                ok = robot.send_trajectory_point_async(
                    feedback[0], feedback[1], feedback[2],
                    feedback[3], feedback[4], feedback[5],
                    stop_speed,
                )
                if not ok:
                    print("[Run] Stop command failed, aborting.")
                    break
                
                # Replan from current position with zero start velocity to new target.
                plan = build_plan(
                    start_position=feedback,
                    start_velocity=[0.0] * 6,
                    target_position=latest_target,
                    dt_sec=dt_sec,
                )
                last_cmd = plan["last_cmd"][:]
                print(f"[Run] Replanning to target: {fmt(latest_target)}")

            if plan is None:
                next_tick += dt_sec
                continue

            t_ref = plan["t_ref"]
            target = plan["target"]
            profiles = plan["profiles"]
            total_runtime = plan["total_runtime"]
            remain = max(abs(target[j] - feedback[j]) for j in range(6))

            # When quitting, allow immediate completion at HOME even if t_ref is frozen.
            if quitting and remain <= finish_tolerance:
                print(f"[Plan] Reached target: {fmt(target)}")
                plan = None
                last_sent_velocity = [0.0] * 6
                break

            # If plan consumed, hold final point and wait for next plan.
            if t_ref >= total_runtime:
                if remain <= finish_tolerance:
                    print(f"[Plan] Reached target: {fmt(target)}")
                    plan = None
                    last_sent_velocity = [0.0] * 6
                    if quitting:
                        break
                    next_tick += dt_sec
                    continue

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
                        target=target[j],
                        direction=profiles[j].direction,
                    )
                    low = min(profiles[j].start, target[j])
                    high = max(profiles[j].start, target[j])
                    cmd.append(clamp(cmd_j, low, high))

                last_cmd = cmd
                plan["last_cmd"] = cmd[:]
                plan["t_ref"] += dt_sec
                frozen_flag = "RUN"
            else:
                cmd = plan["last_cmd"][:]
                plan["frozen_cycles"] += 1
                frozen_flag = "HOLD"

            speed_cmd = deg_speed_to_ascii_speed(v_traj)
            ok = robot.send_trajectory_point_async(
                cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], speed_cmd
            )
            if not ok:
                print("[Run] Send failed, aborting continuous control loop.")
                break

            last_sent_velocity = v_traj[:]

            if PRINT_LOG and (wall_elapsed - last_log >= LOG_INTERVAL):
                print(
                    f"tw={wall_elapsed:5.2f}s | tr={plan['t_ref']:5.2f}s | {frozen_flag} | "
                    f"fb=[{fmt(feedback)}] | traj=[{fmt(x_traj)}] | cmd=[{fmt(cmd)}] | spd={speed_cmd:5.2f}"
                )
                last_log = wall_elapsed

            next_tick += dt_sec

    finally:
        stop_event.set()


def run():
    selected_home = HOME_POSITION[:]

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

        print(
            f"[Init] Inputs: min_speed={MIN_SPEED}, max_speed={MAX_SPEED}, "
            f"acc={ACCELERATION}, dt={CONTROL_DT_MS}ms"
        )

        run_continuous_tracking(robot, selected_home)

    finally:
        try:
            stop_and_hold(robot)
        finally:
            robot.disconnect()


if __name__ == "__main__":
    run()
