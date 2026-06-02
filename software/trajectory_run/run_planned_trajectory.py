"""Stream ``last_plan.npz`` to the robot: home -> goal -> home, repeated."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from manual_control.mapping import (  # noqa: E402
    HOME_JOINTS_DEG,
    MAX_SPEED,
    MIN_SPEED,
    deg_speed_to_ascii_speed,
    sim_rad_to_transformed_deg,
)
from robot_ascii_interface import RobotAsciiInterface  # noqa: E402
from robot_config import default_robot_port  # noqa: E402
from trajectory_run.plan_loader import default_plan_path, load_plan_npz, load_planned_trajectory_rad  # noqa: E402
from trajectory_run.quintic_spline import (  # noqa: E402
    smooth_and_retime_waypoints,
    summarize_smooth_info,
)
from trajectory_run.run_results import default_run_results_path, save_run_results  # noqa: E402


def sim_trajectory_to_robot_deg(trajectory_rad: np.ndarray) -> np.ndarray:
    """Convert planner joint samples (sim rad) to robot ``&`` command degrees."""
    traj = np.asarray(trajectory_rad, dtype=np.float64)
    n, nj = traj.shape
    out = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        for j in range(min(6, nj)):
            out[i, j] = sim_rad_to_transformed_deg(j, float(traj[i, j]))
    return out


def format_joints_deg(joints: Sequence[float]) -> str:
    labels = ["J1", "J2", "J3", "J4", "J5", "J6"]
    parts = [f"{labels[i]}={joints[i]:8.4f}" for i in range(min(6, len(joints)))]
    return "  ".join(parts)


def _joint_errors(
    measured: Sequence[float],
    reference: Sequence[float],
) -> tuple[float, np.ndarray]:
    """Return (max abs error, per-joint errors) in degrees."""
    ref = np.asarray(reference[:6], dtype=np.float64)
    act = np.asarray(measured[:6], dtype=np.float64)
    per_joint = act - ref
    return float(np.max(np.abs(per_joint))), per_joint


def print_cycle_summary(
    records: list[dict],
    *,
    start_deg: Sequence[float],
    goal_deg: Sequence[float],
) -> None:
    """Print aggregate statistics after all cycles."""
    if not records:
        return

    n = len(records)
    goal_errs = [r["goal_max_err"] for r in records if r.get("goal_max_err") is not None]
    home_errs = [r["home_max_err"] for r in records if r.get("home_max_err") is not None]

    print(f"\n{'=' * 60}")
    print(f"FINAL SUMMARY ({n} cycle{'s' if n != 1 else ''})")
    print("=" * 60)
    print(f"Planned start: {format_joints_deg(start_deg)}")
    print(f"Planned goal:  {format_joints_deg(goal_deg)}")
    print()
    print(f"{'Cycle':>5}  {'Goal max |Δ|':>12}  {'Home max |Δ|':>12}")
    print("-" * 34)
    for r in records:
        g = f"{r['goal_max_err']:.4f}" if r.get("goal_max_err") is not None else "   n/a"
        h = f"{r['home_max_err']:.4f}" if r.get("home_max_err") is not None else "   n/a"
        print(f"{r['cycle']:>5}  {g:>12}  {h:>12}")

    if goal_errs:
        print()
        print(
            f"Goal error (deg):  min={min(goal_errs):.4f}  "
            f"max={max(goal_errs):.4f}  mean={float(np.mean(goal_errs)):.4f}"
        )
    if home_errs:
        print(
            f"Home error (deg):  min={min(home_errs):.4f}  "
            f"max={max(home_errs):.4f}  mean={float(np.mean(home_errs)):.4f}"
        )

    last = records[-1]
    if last.get("at_home") is not None:
        print()
        print(f"Final physical joints: {format_joints_deg(last['at_home'])}")
        if last.get("home_max_err") is not None:
            print(f"Final max |Δ| from start (deg): {last['home_max_err']:.4f}")
    print("=" * 60)


def wait_until_joints(
    robot: RobotAsciiInterface,
    target_deg: Sequence[float],
    *,
    timeout_sec: float,
    tol_deg: float = 0.5,
    poll_interval_s: float = 0.25,
    stable_reads: int = 3,
    label: str = "",
    name: str = "HOME",
) -> tuple[bool, List[float] | None]:
    """Poll ``#GETJPOS`` until joints stay within ``tol_deg`` of ``target_deg``."""
    target = np.asarray(target_deg[:6], dtype=np.float64)
    deadline = time.perf_counter() + max(1.0, float(timeout_sec))
    stable = 0
    last: List[float] | None = None

    while time.perf_counter() < deadline:
        current = read_physical_joints(robot)
        if current is None:
            time.sleep(poll_interval_s)
            continue
        last = current
        err = float(np.max(np.abs(np.asarray(current[:6]) - target)))
        if err <= tol_deg:
            stable += 1
            if stable >= stable_reads:
                return True, current
        else:
            stable = 0
        time.sleep(poll_interval_s)

    if last is not None:
        err = float(np.max(np.abs(np.asarray(last[:6]) - target)))
        print(
            f"{label}Timed out waiting for {name} "
            f"(max |Δ|={err:.3f} deg, tol={tol_deg:.2f} deg)."
        )
    else:
        print(f"{label}Timed out waiting for {name} (no joint reads).")
    return False, last


def _drain_serial(robot: RobotAsciiInterface, max_lines: int = 32) -> None:
    if not robot.ser or not robot.ser.is_open:
        return
    try:
        for _ in range(max_lines):
            if robot.ser.in_waiting <= 0:
                break
            robot.ser.readline()
    except Exception:
        pass


def firmware_home(
    robot: RobotAsciiInterface,
    target_deg: Sequence[float],
    *,
    cmd_mode: int = 2,
    settle_timeout_sec: float = 180.0,
    home_wait_ms: int = 1500,
    home_read_sec: float = 120.0,
    home_arrive_tol_deg: float = 0.5,
    label: str = "",
) -> tuple[bool, List[float] | None]:
    """Send ``!HOME``, poll joints until at plan start, then enable streaming CMDMODE."""
    target = [float(target_deg[j]) for j in range(6)]
    current = read_physical_joints(robot)
    if current is not None:
        err = float(np.max(np.abs(np.asarray(current[:6]) - np.asarray(target))))
        print(f"{label}!HOME (max |Δ|={err:.2f} deg before homing)")
    else:
        print(f"{label}!HOME")

    # Sequential mode during homing — avoid CMDMODE 2 fighting stale stream targets.
    print(f"{label}#CMDMODE 1 -> {robot.set_command_mode(1)}")
    time.sleep(0.2)

    resp = robot.home(wait_ms=home_wait_ms, extra_read_sec=home_read_sec)
    if resp is None:
        print(f"{label}Note: !HOME serial reply not received (may still be moving).")
    elif "ok" not in str(resp).lower():
        print(f"{label}Warning: !HOME response: {resp}")

    print(f"{label}Waiting until joints reach HOME (up to {settle_timeout_sec:.0f} s)...")
    ok, final = wait_until_joints(
        robot,
        target,
        timeout_sec=settle_timeout_sec,
        tol_deg=home_arrive_tol_deg,
        label=label,
        name="HOME",
    )
    _drain_serial(robot)

    if not ok and final is not None:
        print(f"{label}Pin HOME: fine correction after !HOME")
        pin_ok, pinned = pin_to_joint_target(
            robot,
            target,
            settle_timeout_sec=min(60.0, settle_timeout_sec),
            tol_deg=home_arrive_tol_deg,
            label=label,
            name="HOME",
        )
        if pin_ok and pinned is not None:
            ok, final = True, pinned

    print(f"{label}#CMDMODE {cmd_mode} -> {robot.set_command_mode(cmd_mode)}")
    time.sleep(0.2)

    if final is None:
        final = read_physical_joints(robot)
    if final is None:
        return False, None
    err = float(np.max(np.abs(np.asarray(final[:6]) - np.asarray(target))))
    print(f"{label}  at HOME: {format_joints_deg(final)} (max |Δ|={err:.3f} deg)")
    if err > home_arrive_tol_deg:
        print(f"{label}Failed: not at HOME within {home_arrive_tol_deg:.2f} deg.")
        return False, final
    return True, final


def pin_to_joint_target(
    robot: RobotAsciiInterface,
    target_deg: Sequence[float],
    *,
    max_joint_vel_deg_s: float = MAX_SPEED,
    settle_timeout_sec: float = 60.0,
    tol_deg: float = 0.15,
    label: str = "",
    name: str = "target",
) -> tuple[bool, List[float] | None]:
    """Send a final ``&`` move so the robot settles on commanded joint angles."""
    target = [float(target_deg[j]) for j in range(6)]
    current = read_physical_joints(robot)
    if current is None:
        print(f"{label}Could not read joints before pin to {name}.")
        return False, None

    err_before = float(np.max(np.abs(np.asarray(current[:6]) - np.asarray(target))))
    if err_before <= tol_deg:
        print(f"{label}Pin {name}: within {tol_deg:.2f} deg (max |Δ|={err_before:.3f}).")
        return True, current

    print(
        f"{label}Pin {name}: correcting max |Δ|={err_before:.3f} deg -> "
        f"{format_joints_deg(target)}"
    )
    speed = deg_speed_to_ascii_speed([max_joint_vel_deg_s] * 6)
    robot.move_joint(
        target[0], target[1], target[2], target[3], target[4], target[5],
        speed=speed,
        wait_ms=250,
    )
    if not robot.wait_for_motion_complete(timeout_sec=settle_timeout_sec):
        print(f"{label}Warning: pin to {name} did not settle within {settle_timeout_sec:.0f} s.")

    final = read_physical_joints(robot)
    if final is None:
        return False, None
    err = float(np.max(np.abs(np.asarray(final[:6]) - np.asarray(target))))
    print(f"{label}  pinned: {format_joints_deg(final)} (max |Δ|={err:.3f} deg)")
    return True, final


def move_to_joint_target(
    robot: RobotAsciiInterface,
    target_deg: Sequence[float],
    *,
    cmd_mode: int = 2,
    settle_timeout_sec: float = 180.0,
    home_wait_ms: int = 1500,
    home_read_sec: float = 120.0,
    home_arrive_tol_deg: float = 0.5,
    label: str = "",
) -> bool:
    """Move to HOME via firmware ``!HOME`` (always), verify joints, then CMDMODE for streaming."""
    ok, _ = firmware_home(
        robot,
        target_deg,
        cmd_mode=cmd_mode,
        settle_timeout_sec=settle_timeout_sec,
        home_wait_ms=home_wait_ms,
        home_read_sec=home_read_sec,
        home_arrive_tol_deg=home_arrive_tol_deg,
        label=label,
    )
    return ok


def stream_waypoints_deg(
    robot: RobotAsciiInterface,
    waypoints_deg: np.ndarray,
    *,
    send_interval_ms: float = 150.0,
    max_joint_vel_deg_s: float = MAX_SPEED,
    speed_scale: float = 1.0,
    planned_duration_s: float | None = None,
    settle_timeout_sec: float = 120.0,
    label: str = "",
) -> bool:
    """
    Stream ``&`` targets in interruptible mode (CMDMODE 2).

    Speed uses ``deg_speed_to_ascii_speed`` from ``manual_control.mapping`` (same as
    teleop): per-segment joint velocity = Δq/Δt, scaled by ``speed_scale``.
    """
    pts = np.asarray(waypoints_deg, dtype=np.float64)
    if len(pts) < 2:
        print(f"{label}Need at least 2 waypoints (got {len(pts)}).")
        return False

    n_seg = len(pts) - 1
    if planned_duration_s is not None and planned_duration_s > 1e-3:
        dt_s = float(planned_duration_s) / n_seg
    else:
        dt_s = max(1e-3, send_interval_ms / 1000.0)

    vmax = max(MIN_SPEED, float(max_joint_vel_deg_s))
    scale = max(0.25, float(speed_scale))

    print(
        f"{label}Streaming {len(pts)} waypoints "
        f"({dt_s * 1000:.0f} ms/seg, speed_scale={scale:.2f})..."
    )

    for i in range(1, len(pts)):
        q_prev = pts[i - 1]
        q = pts[i]
        vels_seg = [(q[j] - q_prev[j]) / dt_s for j in range(6)]
        vels_seg = [max(-vmax, min(vmax, v * scale)) for v in vels_seg]
        speed = deg_speed_to_ascii_speed(vels_seg)

        ok = robot.send_trajectory_point_async(
            q[0], q[1], q[2], q[3], q[4], q[5], speed
        )
        if not ok:
            print(f"{label}Failed to send waypoint {i}/{n_seg}.")
            return False
        time.sleep(dt_s)

    print(f"{label}Waiting for motion to settle...")
    if not robot.wait_for_motion_complete(timeout_sec=settle_timeout_sec):
        print(f"{label}Warning: settle timeout ({settle_timeout_sec:.0f} s).")
    return True


def read_physical_joints(robot: RobotAsciiInterface, retries: int = 5) -> List[float] | None:
    for _ in range(retries):
        joints = robot.get_joint_pos()
        if joints is not None and len(joints) >= 6:
            return [float(v) for v in joints[:6]]
        time.sleep(0.05)
    return None


def prepare_robot_trajectory(
    trajectory_rad: np.ndarray,
    *,
    smooth: bool,
    sample_dt_s: float,
    max_vel_deg_s: float,
    max_acc_deg_s2: float,
    max_knots: int = 80,
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Convert sim-rad plan to robot deg; optionally quintic smooth + retime."""
    waypoints_deg = sim_trajectory_to_robot_deg(trajectory_rad)
    info: dict = {"n_plan_samples": len(waypoints_deg)}

    if not smooth:
        info["smoothed"] = False
        return waypoints_deg, None, info

    pos, vel, sinfo = smooth_and_retime_waypoints(
        waypoints_deg,
        sample_dt=sample_dt_s,
        max_vel_deg_s=max_vel_deg_s,
        max_acc_deg_s2=max_acc_deg_s2,
        max_knots=max_knots,
    )
    info["smoothed"] = True
    info.update(sinfo)
    return pos, vel, info


def run_trajectory(
    robot: RobotAsciiInterface,
    forward_pos: np.ndarray,
    start_deg: Sequence[float],
    goal_deg: Sequence[float],
    *,
    planned_duration_s: float | None,
    repeats: int,
    send_interval_ms: float,
    max_joint_vel_deg_s: float,
    settle_timeout_sec: float,
    speed_scale: float,
    skip_home: bool = False,
    pin_endpoints: bool = True,
    pin_tol_deg: float = 0.15,
    cmd_mode: int = 2,
    home_wait_ms: int = 1500,
    home_read_sec: float = 120.0,
    home_arrive_tol_deg: float = 0.5,
    home_timeout_sec: float = 180.0,
) -> list[dict]:
    """Execute HOME -> target -> HOME cycles; print physical joints after each leg."""
    reverse_pos = forward_pos[::-1]
    multi = repeats > 1
    stream_kw = dict(
        send_interval_ms=send_interval_ms,
        max_joint_vel_deg_s=max_joint_vel_deg_s,
        planned_duration_s=planned_duration_s,
        settle_timeout_sec=settle_timeout_sec,
        speed_scale=speed_scale,
    )
    pin_kw = dict(
        max_joint_vel_deg_s=max_joint_vel_deg_s,
        settle_timeout_sec=settle_timeout_sec,
        tol_deg=pin_tol_deg,
    )
    cycle_records: list[dict] = []

    for cycle in range(1, repeats + 1):
        prefix = f"[Cycle {cycle}/{repeats}] " if multi else ""

        if multi:
            print(f"\n{'=' * 60}")
        else:
            print()

        record: dict = {"cycle": cycle, "goal_max_err": None, "home_max_err": None, "at_home": None}

        if not skip_home:
            print(f"{prefix}Step 1: go to HOME (!HOME)")
            if not move_to_joint_target(
                robot,
                start_deg,
                cmd_mode=cmd_mode,
                settle_timeout_sec=home_timeout_sec,
                home_wait_ms=home_wait_ms,
                home_read_sec=home_read_sec,
                home_arrive_tol_deg=home_arrive_tol_deg,
                label=prefix,
            ):
                raise SystemExit(f"{prefix}Failed to reach HOME.".strip())

        print(f"{prefix}Step 2: HOME -> target")
        if multi:
            print("=" * 60)

        if not stream_waypoints_deg(robot, forward_pos, label=prefix, **stream_kw):
            raise SystemExit(f"{prefix}Forward motion failed.".strip())

        if pin_endpoints:
            ok, at_target = pin_to_joint_target(
                robot, goal_deg, label=prefix, name="TARGET", **pin_kw
            )
            if not ok:
                raise SystemExit(f"{prefix}Failed to pin TARGET.".strip())
        else:
            at_target = read_physical_joints(robot)

        if at_target is None:
            print(f"{prefix}Could not read joints at target.".strip())
        else:
            goal_max_err, goal_per_joint = _joint_errors(at_target, goal_deg)
            record["goal_max_err"] = goal_max_err
            record["goal_joint_err"] = goal_per_joint.tolist()
            record["at_target"] = list(at_target[:6])
            print(f"{prefix}Physical joints at TARGET: {format_joints_deg(at_target)}".strip())
            print(f"{prefix}Max |Δ| from planned goal (deg): {goal_max_err:.4f}".strip())

        print(f"\n{prefix}Step 3: target -> HOME".strip())
        if not stream_waypoints_deg(robot, reverse_pos, label=prefix, **stream_kw):
            raise SystemExit(f"{prefix}Reverse motion failed.".strip())

        if pin_endpoints:
            ok, at_home = pin_to_joint_target(
                robot, start_deg, label=prefix, name="HOME", **pin_kw
            )
            if not ok:
                raise SystemExit(f"{prefix}Failed to pin HOME.".strip())
        else:
            at_home = read_physical_joints(robot)

        if at_home is None:
            print(f"{prefix}Could not read joints at end of cycle.".strip())
        else:
            home_max_err, home_per_joint = _joint_errors(at_home, start_deg)
            record["home_max_err"] = home_max_err
            record["home_joint_err"] = home_per_joint.tolist()
            record["at_home"] = list(at_home[:6])
            if multi:
                print(f"\n{prefix}--- End of cycle {cycle}/{repeats} ---")
            print(f"{prefix}Physical joints: {format_joints_deg(at_home)}".strip())
            print(f"{prefix}Max |Δ| from planned start (deg): {home_max_err:.4f}".strip())

        cycle_records.append(record)

    if cycle_records:
        print_cycle_summary(cycle_records, start_deg=start_deg, goal_deg=goal_deg)

    return cycle_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cached planner trajectory on the physical robot (home<->goal repeats)."
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port (default: ROBOT_PORT env or platform default)",
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=1.0, help="Serial read timeout (s)")
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help=f"Plan npz (default: {default_plan_path()})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="HOME->target->HOME cycles (default 5; use 1 or --once for a single test)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single cycle HOME->target->HOME (same as --repeats 1)",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Deprecated alias for --once (single HOME->target->HOME cycle)",
    )
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=150.0,
        help="Stream period when no planned duration (default 150 ms, same as teleop)",
    )
    parser.add_argument(
        "--cmd-mode",
        type=int,
        default=2,
        choices=(1, 2, 3, 4),
        help="Robot CMDMODE (default 2 = interruptible, same as manual_control)",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=120.0,
        help="Max wait for motion to settle after each leg (default 120 s)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable quintic spline smoothing (raw planner waypoints)",
    )
    parser.add_argument(
        "--sample-dt",
        type=float,
        default=0.05,
        metavar="SEC",
        help="Resample period after quintic fit (default 0.05 s = 50 ms)",
    )
    parser.add_argument(
        "--max-vel",
        type=float,
        default=25.0,
        help="Retiming limit: max joint speed (deg/s, default 25)",
    )
    parser.add_argument(
        "--max-accel",
        type=float,
        default=25.0,
        help="Retiming limit: max joint acceleration (deg/s^2, default 25)",
    )
    parser.add_argument(
        "--max-knots",
        type=int,
        default=150,
        help="Max spline knots from planner path (default 150 = use full plan)",
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=1.5,
        help="Multiply streamed speed commands (>1 faster on robot, default 1.5)",
    )
    parser.add_argument(
        "--skip-home",
        action="store_true",
        help="Skip initial move to planned HOME/start joints",
    )
    parser.add_argument(
        "--home-tol",
        type=float,
        default=0.15,
        metavar="DEG",
        help="Deprecated (unused): !HOME is always sent at cycle start",
    )
    parser.add_argument(
        "--pin-tol",
        type=float,
        default=0.15,
        metavar="DEG",
        help="After each leg, pin if error exceeds this (default 0.15 deg)",
    )
    parser.add_argument(
        "--no-pin",
        action="store_true",
        help="Disable final pin move after streamed legs (stream-only tracking)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Save run log npz (default: {default_run_results_path()})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write run log npz",
    )
    parser.add_argument(
        "--home-wait-ms",
        type=int,
        default=1500,
        help="Pre-read wait for !HOME reply (default 1500 ms)",
    )
    parser.add_argument(
        "--home-read-sec",
        type=float,
        default=120.0,
        help="Serial read window for !HOME reply (default 120 s)",
    )
    parser.add_argument(
        "--home-timeout",
        type=float,
        default=180.0,
        metavar="SEC",
        help="Max wait for joints to reach HOME after !HOME (default 180 s)",
    )
    parser.add_argument(
        "--home-arrive-tol",
        type=float,
        default=0.5,
        metavar="DEG",
        help="HOME success tolerance on joint readout (default 0.5 deg)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load plan and print summary without connecting to the robot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    port = args.port or default_robot_port()
    if args.once or args.forward_only:
        args.repeats = 1
    if args.interval_ms <= 0:
        raise SystemExit("--interval-ms must be positive")
    if args.sample_dt <= 0:
        raise SystemExit("--sample-dt must be positive")

    try:
        trajectory_rad, start_q, goal_q = load_planned_trajectory_rad(args.plan)
        plan_cache = load_plan_npz(args.plan)
    except FileNotFoundError as exc:
        raise SystemExit(f"No plan at {exc}. Run plan_and_demo.py first.") from exc
    smooth = not args.no_smooth
    sample_dt = args.sample_dt if smooth else args.interval_ms / 1000.0

    forward_pos, forward_vel, info = prepare_robot_trajectory(
        trajectory_rad,
        smooth=smooth,
        sample_dt_s=sample_dt,
        max_vel_deg_s=args.max_vel,
        max_acc_deg_s2=args.max_accel,
        max_knots=max(4, int(args.max_knots)),
    )

    start_deg = sim_trajectory_to_robot_deg(start_q.reshape(1, -1))[0]
    goal_deg = sim_trajectory_to_robot_deg(goal_q.reshape(1, -1))[0]

    print(f"Plan: {args.plan or default_plan_path()}")
    print(f"Planner samples: {info['n_plan_samples']}")
    if info.get("smoothed"):
        print(summarize_smooth_info(info))
    else:
        print(f"Execution samples: {len(forward_pos)} (no smoothing)")
    print(f"Start (deg): {format_joints_deg(start_deg)}")
    print(f"Goal  (deg): {format_joints_deg(goal_deg)}")
    if args.repeats == 1:
        print("Mode: single cycle HOME -> target -> HOME")
    else:
        print(f"Mode: {args.repeats} cycles HOME -> target -> HOME")

    send_interval_ms = args.interval_ms
    planned_duration_s = info.get("duration_s") if info.get("smoothed") else None
    if planned_duration_s and args.speed_scale > 1.0:
        planned_duration_s = float(planned_duration_s) / float(args.speed_scale)

    if args.dry_run:
        print(
            f"Stream: CMDMODE {args.cmd_mode}, interval={send_interval_ms:.0f} ms, "
            f"max_joint_vel={args.max_vel:.1f} deg/s, speed_scale={args.speed_scale}"
        )
        if planned_duration_s:
            print(f"Planned leg duration: {planned_duration_s:.2f} s")
        return

    robot = RobotAsciiInterface(port=port, baudrate=args.baudrate, timeout=args.timeout)
    if not robot.connect():
        raise SystemExit(f"Failed to connect on {port}")

    try:
        print(f"!STOP  -> {robot.stop()}")
        time.sleep(0.2)
        print(f"!START -> {robot.start()}")
        time.sleep(0.2)

        initial = read_physical_joints(robot)
        if initial is not None:
            print(f"Initial physical joints: {format_joints_deg(initial)}")

        cycle_records = run_trajectory(
            robot,
            forward_pos,
            start_deg,
            goal_deg,
            planned_duration_s=planned_duration_s,
            repeats=max(1, int(args.repeats)),
            send_interval_ms=send_interval_ms,
            max_joint_vel_deg_s=args.max_vel,
            settle_timeout_sec=args.settle_timeout,
            speed_scale=max(0.25, float(args.speed_scale)),
            skip_home=args.skip_home,
            pin_endpoints=not args.no_pin,
            pin_tol_deg=args.pin_tol,
            cmd_mode=args.cmd_mode,
            home_wait_ms=args.home_wait_ms,
            home_read_sec=args.home_read_sec,
            home_arrive_tol_deg=args.home_arrive_tol,
            home_timeout_sec=args.home_timeout,
        )

        if not args.no_save and cycle_records:
            out_path = args.output or default_run_results_path()
            saved = save_run_results(
                out_path,
                cycle_records=cycle_records,
                start_deg=start_deg,
                goal_deg=goal_deg,
                start_q=start_q,
                goal_q=goal_q,
                target_pos=plan_cache.get("target_pos"),
                target_rpy=plan_cache.get("target_rpy"),
                plan_path=(args.plan or default_plan_path()).resolve(),
                initial_joints_deg=initial,
                config={
                    "repeats": args.repeats,
                    "cmd_mode": args.cmd_mode,
                    "max_vel": args.max_vel,
                    "max_accel": args.max_accel,
                    "speed_scale": args.speed_scale,
                    "pin_endpoints": not args.no_pin,
                    "pin_tol_deg": args.pin_tol,
                },
            )
            print(f"\nRun log saved: {saved}")
            print("Analyze: python software/trajectory_run/analyze_run_results.py")

        print("\nDone.")
    except KeyboardInterrupt:
        print("\nInterrupted — sending !STOP")
        robot.stop()
        raise
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
