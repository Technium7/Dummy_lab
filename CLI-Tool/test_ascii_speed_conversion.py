#!/usr/bin/env python3
"""Validate ASCII speed conversion by single-joint timing tests.

Test method:
1) Start from HOME.
2) Move one joint at a time by a fixed offset using '&' command speed.
3) Compute ASCII speed from desired joint speed (deg/s) and reduction.
4) Measure actual move time and compare achieved deg/s.
"""

import time
from robot_ascii_interface import RobotAsciiInterface


HOME = [0, 0, 90, 0, 0, 0]
JOINT_REDUCTION = [30.0, 30.0, 30.0, 24.0, 30.0, 50.0]  # J1..J6

DESIRED_JOINT_SPEED_DEG_S = 5.0
TEST_OFFSETS_DEG = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]

HOME_SPEED_ASCII = 20.0
TIMEOUT_SEC = 20


def fmt(values):
    return ", ".join(f"{value:.2f}" for value in values)


def deg_speed_to_ascii_speed_single_joint(joint_speed_deg_s, reduction):
    # From firmware path:
    # motor_rps = joint_deg_s * reduction / 360
    # dynamicJointSpeed ~= 0.1 * jointSpeed(ascii)
    # => ascii ~= (joint_deg_s * reduction) / 36
    return (joint_speed_deg_s * reduction) / 36.0


def move_and_wait(robot, target, speed_ascii, timeout_sec=TIMEOUT_SEC):
    robot.move_joint(
        target[0], target[1], target[2], target[3], target[4], target[5],
        speed=speed_ascii,
        wait_ms=200,
    )
    return robot.wait_for_motion_complete(timeout_sec=timeout_sec, poll_interval_ms=50, tolerance=0.2)


def run():
    robot = RobotAsciiInterface()
    if not robot.connect():
        return

    results = []

    try:
        print("[Init] STOP")
        print(f"       {robot.stop()}")
        time.sleep(0.2)

        print("[Init] START")
        print(f"       {robot.start()}")
        time.sleep(0.2)

        print("[Init] CMDMODE 2")
        print(f"       {robot.set_command_mode(2)}")
        time.sleep(0.2)

        print(f"[Init] Move HOME: {fmt(HOME)}")
        move_and_wait(robot, HOME, HOME_SPEED_ASCII, timeout_sec=10)
        print(f"[Init] At HOME: {fmt(robot.get_joint_pos() or HOME)}")

        print("\n=== Per-Joint Speed Conversion Test ===")
        print(f"Desired joint speed: {DESIRED_JOINT_SPEED_DEG_S:.2f} deg/s\n")

        for joint_idx in range(6):
            joint_name = f"J{joint_idx + 1}"
            reduction = JOINT_REDUCTION[joint_idx]
            delta_deg = TEST_OFFSETS_DEG[joint_idx]
            ascii_speed = deg_speed_to_ascii_speed_single_joint(DESIRED_JOINT_SPEED_DEG_S, reduction)

            # Return to HOME before each trial
            move_and_wait(robot, HOME, HOME_SPEED_ASCII, timeout_sec=10)

            target = HOME[:]
            target[joint_idx] = HOME[joint_idx] + delta_deg

            print(
                f"[{joint_name}] reduction={reduction:.1f}, delta={delta_deg:.1f}deg, "
                f"ascii_speed={ascii_speed:.3f}"
            )

            t0 = time.perf_counter()
            completed = move_and_wait(robot, target, ascii_speed, timeout_sec=TIMEOUT_SEC)
            elapsed = time.perf_counter() - t0

            final_pos = robot.get_joint_pos() or target
            actual_delta = abs(final_pos[joint_idx] - HOME[joint_idx])
            achieved_speed = (actual_delta / elapsed) if elapsed > 1e-6 else 0.0

            print(
                f"     done={completed}, time={elapsed:.3f}s, "
                f"actual_delta={actual_delta:.2f}deg, achieved={achieved_speed:.2f} deg/s"
            )

            results.append({
                "joint": joint_name,
                "reduction": reduction,
                "ascii_speed": ascii_speed,
                "elapsed": elapsed,
                "actual_delta": actual_delta,
                "achieved_speed": achieved_speed,
                "error": achieved_speed - DESIRED_JOINT_SPEED_DEG_S,
                "completed": completed,
            })

        # Return HOME at end
        move_and_wait(robot, HOME, HOME_SPEED_ASCII, timeout_sec=10)

        print("\n=== Summary ===")
        print("Joint | Red | ASCIIspd | Time(s) | Delta(deg) | Achieved(deg/s) | Error")
        for item in results:
            print(
                f"{item['joint']:>4} | {item['reduction']:>3.0f} | {item['ascii_speed']:>8.3f} | "
                f"{item['elapsed']:>7.3f} | {item['actual_delta']:>10.2f} | "
                f"{item['achieved_speed']:>15.2f} | {item['error']:>+6.2f}"
            )

    finally:
        try:
            robot.stop()
            robot.set_command_mode(2)
        finally:
            robot.disconnect()


if __name__ == "__main__":
    run()
