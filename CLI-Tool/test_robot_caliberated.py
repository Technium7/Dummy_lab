#!/usr/bin/env python3
"""Calibrated trajectory tracker based on measured per-joint speed mismatch."""

from dataclasses import dataclass
from robot_ascii_interface import RobotAsciiInterface
import time


# Inputs requested
HOME_POSITION = [0, 0, 90, 0, 0, 0]
TARGET_POSITION = [60, -40, 60, 20, 0, 0]
MIN_SPEED = 10.0      # deg/s (joint space)
MAX_SPEED = 30.0      # deg/s (joint space)
ACCELERATION = 10.0   # deg/s^2 (joint space)
CONTROL_DT_MS = 50

# From firmware DummyRobot ctor: reductions for J1..J6
JOINT_REDUCTION = [30.0, 30.0, 30.0, 24.0, 30.0, 50.0]
DOMINANT_REDUCTION = max(JOINT_REDUCTION)

# --- Calibration from your test summary ---
# Desired speed in that test was 5 deg/s.
CAL_TEST_DESIRED_SPEED = 5.0
CAL_TEST_ACHIEVED_SPEED = [2.55, 2.58, 2.56, 2.10, 2.56, 5.57]

# Conservative blend toward full correction: 0=no correction, 1=full correction.
CAL_BLEND = 0.6
CAL_MIN = 0.90
CAL_MAX = 2.50

LOG_INTERVAL = 0.4
LOOKAHEAD_GAIN = 5


def build_calibration_factors():
    factors = []
    for achieved in CAL_TEST_ACHIEVED_SPEED:
        raw_gain = CAL_TEST_DESIRED_SPEED / max(achieved, 1e-6)
        conservative_gain = 1.0 + CAL_BLEND * (raw_gain - 1.0)
        factors.append(max(CAL_MIN, min(CAL_MAX, conservative_gain)))
    return factors


CAL_GAIN = build_calibration_factors()


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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def deg_speed_to_ascii_speed(v_deg_per_s_list):
    """
    Convert per-joint deg/s demand to firmware ASCII scalar speed.
    Includes per-joint calibration gain (boost where measured speed is low).
    """
    demand = max(
        abs(v_deg_per_s_list[j]) * CAL_GAIN[j] * JOINT_REDUCTION[j]
        for j in range(6)
    ) / 36.0

    red_max = DOMINANT_REDUCTION
    gain_max = max(CAL_GAIN)
    min_ascii = MIN_SPEED * red_max * gain_max / 36.0
    max_ascii = MAX_SPEED * red_max * gain_max / 36.0
    return clamp(demand, min_ascii, max_ascii)


def make_joint_profile(start, target, accel, vmax, joint_index):
    """
    Build profile in joint-space degrees with:
    - reduction compensation
    - calibration-aware effective model (slower model when joint measured slow)
    """
    reduction_scale = DOMINANT_REDUCTION / JOINT_REDUCTION[joint_index]

    # If joint was measured slower than expected (gain>1), model it conservatively
    # by reducing effective kinematic capability (longer planned runtime).
    model_scale = reduction_scale / CAL_GAIN[joint_index]

    effective_vmax = vmax * model_scale
    effective_accel = accel * model_scale

    distance = abs(target - start)
    direction = 1.0 if target >= start else -1.0

    if distance < 1e-6:
        return JointProfile(start, target, direction, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effective_accel, 0.0, 0.0)

    t_to_vmax = effective_vmax / effective_accel
    d_acc_to_vmax = 0.5 * effective_accel * t_to_vmax * t_to_vmax

    if 2.0 * d_acc_to_vmax >= distance:
        t_acc = (distance / effective_accel) ** 0.5
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
    global_total = max(p.t_total for p in profiles)
    for p in profiles:
        pad = max(0.0, global_total - p.t_total)
        p.pre_pad = 0.5 * pad
        p.post_pad = 0.5 * pad
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
        v = accel * tau
        x = 0.5 * accel * tau * tau
    elif tau < profile.t_acc + profile.t_flat:
        dt = tau - profile.t_acc
        v = profile.v_peak
        x = d_acc + profile.v_peak * dt
    else:
        dt = tau - profile.t_acc - profile.t_flat
        v = max(0.0, profile.v_peak - accel * dt)
        x = d_acc + d_flat + profile.v_peak * dt - 0.5 * accel * dt * dt

    return x, v


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


def build_command(feedback_x, x_traj, v_traj, dt_sec, target_x, direction):
    if direction >= 0.0:
        x_cmd = max(x_traj, feedback_x) + LOOKAHEAD_GAIN * dt_sec * max(0.0, v_traj)
        x_cmd = min(x_cmd, target_x)
    else:
        x_cmd = min(x_traj, feedback_x) + LOOKAHEAD_GAIN * dt_sec * min(0.0, v_traj)
        x_cmd = max(x_cmd, target_x)
    return x_cmd


def stop_and_hold(robot):
    print("[Shutdown] Stop and hold")
    print(f"           STOP: {robot.stop()}")
    time.sleep(0.12)
    print(f"           MODE2: {robot.set_command_mode(2)}")
    time.sleep(0.12)

    pos = robot.get_joint_pos()
    if pos and len(pos) >= 6:
        robot.move_joint(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], speed=max(MIN_SPEED, 6.0), wait_ms=200)


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

    print(f"\n=== {leg_name}: Planned runtime per joint (s) ===")
    for j, p in enumerate(profiles, start=1):
        print(
            f"J{j}: total={p.t_total:.3f}, pad(pre/post)=({p.pre_pad:.3f}/{p.post_pad:.3f}), "
            f"v_peak={p.v_peak:.2f}, cal_gain={CAL_GAIN[j-1]:.2f}"
        )
    print(f"Global synchronized runtime: {total_runtime:.3f}s\n")

    last_feedback = current[:]
    start = time.perf_counter()
    next_tick = start
    last_log = -LOG_INTERVAL

    while True:
        now = time.perf_counter()
        elapsed = now - start
        if elapsed >= total_runtime:
            break

        if now < next_tick:
            time.sleep(next_tick - now)

        feedback = robot.get_joint_pos()
        if feedback and len(feedback) >= 6:
            last_feedback = feedback
        else:
            feedback = last_feedback

        x_traj = []
        v_traj = []
        for p in profiles:
            x, v = sample_joint_at_time(p, elapsed)
            x_traj.append(x)
            v_traj.append(v)

        cmd = []
        for j in range(6):
            cmd_j = build_command(
                feedback_x=feedback[j],
                x_traj=x_traj[j],
                v_traj=v_traj[j],
                dt_sec=dt_sec,
                target_x=target_position[j],
                direction=profiles[j].direction,
            )
            low = min(profiles[j].start, target_position[j])
            high = max(profiles[j].start, target_position[j])
            cmd.append(clamp(cmd_j, low, high))

        speed_cmd = deg_speed_to_ascii_speed(v_traj)

        ok = robot.send_trajectory_point_async(
            cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], speed_cmd
        )
        if not ok:
            print("Send failed, aborting.")
            break

        if elapsed - last_log >= LOG_INTERVAL:
            print(
                f"t={elapsed:5.2f}s | fb=[{fmt(feedback)}] | traj=[{fmt(x_traj)}] | "
                f"cmd=[{fmt(cmd)}] | spd={speed_cmd:5.2f} (ascii)"
            )
            last_log = elapsed

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


def run():
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

        print(f"[Init] Move to HOME: {fmt(HOME_POSITION)}")
        robot.move_joint(
            HOME_POSITION[0], HOME_POSITION[1], HOME_POSITION[2],
            HOME_POSITION[3], HOME_POSITION[4], HOME_POSITION[5],
            speed=MIN_SPEED,
            wait_ms=250,
        )
        robot.wait_for_motion_complete(timeout_sec=10)
        print(f"[Init] At HOME: {fmt(robot.get_joint_pos())}")

        pos = robot.get_joint_pos()
        print(f"[Init] Current position: {fmt(pos)}")
        print(f"[Init] Home position   : {fmt(HOME_POSITION)}")
        print(f"[Init] Target position : {fmt(TARGET_POSITION)}")
        print(
            f"[Init] Inputs: min_speed={MIN_SPEED}, max_speed={MAX_SPEED}, "
            f"acc={ACCELERATION}, dt={CONTROL_DT_MS}ms"
        )
        print(f"[Init] Calibration gain: {fmt(CAL_GAIN)}")

        run_tracking(robot, TARGET_POSITION, leg_name="HOME -> TARGET")
        run_tracking(robot, HOME_POSITION, leg_name="TARGET -> HOME")

    finally:
        try:
            stop_and_hold(robot)
        finally:
            robot.disconnect()


if __name__ == "__main__":
    run()
