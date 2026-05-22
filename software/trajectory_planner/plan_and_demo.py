"""Plan HOME -> target end pose with RRTConnect + CHOMP, play back in PyBullet.

Collision checking uses PyBullet contacts against the static environment mesh.
RRTConnect and CHOMP are Python implementations matching OMPL/MoveIt algorithms
(official ``ompl`` pip package is not available on Windows).

Edit targets in plan_targets.json for future runs.

Usage (repo root):
  python software/trajectory_planner/plan_and_demo.py

Usage (software/):
  python trajectory_planner/plan_and_demo.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

try:
    import pybullet as p
except ImportError as exc:
    raise SystemExit("Missing dependency: pybullet. Install with: pip install pybullet") from exc

from manual_control.mapping import HOME_JOINTS_DEG  # noqa: E402
from movement_sim.joint_controller import (  # noqa: E402
    build_package_map,
    choose_description,
    discover_descriptions,
    materialize_urdf,
)
from trajectory_planner.chomp import chomp_smooth, pin_trajectory_endpoints, resample_trajectory  # noqa: E402
from trajectory_planner.collision import collision_cost, is_state_valid  # noqa: E402
from trajectory_planner.rrt_connect import RRTConnectPlanner, interpolate_path_dense, joint_distance  # noqa: E402
from trajectory_planner.scene import (  # noqa: E402
    PlanningScene,
    build_planning_scene,
    frame_camera_on_scene,
    get_end_link_pose,
    joint_array_from_transformed_deg,
    measure_pose_error,
    refine_joint_config_to_pose,
    set_joint_positions,
    solve_ik,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _targets_path() -> Path:
    return Path(__file__).resolve().parent / "plan_targets.json"


def _plan_cache_path() -> Path:
    return Path(__file__).resolve().parent / "last_plan.npz"


def save_plan_cache(
    trajectory: np.ndarray,
    goal_q: Sequence[float],
    start_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    start_deg: Sequence[float],
    seed: int,
) -> None:
    np.savez(
        _plan_cache_path(),
        trajectory=trajectory,
        goal_q=np.asarray(goal_q, dtype=np.float64),
        start_q=np.asarray(start_q, dtype=np.float64),
        target_pos=np.asarray(target_pos, dtype=np.float64),
        target_rpy=np.asarray(target_rpy, dtype=np.float64),
        start_deg=np.asarray(start_deg, dtype=np.float64),
        seed=np.int64(seed),
    )


def load_plan_cache() -> Optional[Dict[str, Any]]:
    path = _plan_cache_path()
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    return {
        "trajectory": data["trajectory"],
        "goal_q": data["goal_q"].tolist(),
        "start_q": data["start_q"].tolist(),
        "target_pos": tuple(float(v) for v in data["target_pos"]),
        "target_rpy": tuple(float(v) for v in data["target_rpy"]),
        "start_deg": [float(v) for v in data["start_deg"]],
        "seed": int(data["seed"]),
    }


def prepare_playback_trajectory(
    trajectory: np.ndarray,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    hold_steps: int = 15,
) -> np.ndarray:
    """Pin endpoints and append goal hold samples so playback ends exactly at goal_q."""
    traj = pin_trajectory_endpoints(trajectory, start_q, goal_q)
    goal_row = np.asarray(goal_q, dtype=np.float64)
    hold = np.tile(goal_row, (hold_steps, 1))
    return np.vstack([traj, hold])


def load_plan_targets(
    path: Path,
) -> Tuple[List[float], Tuple[float, float, float], Tuple[float, float, float], Optional[List[float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    start_deg = [float(v) for v in data.get("start_joints_deg", HOME_JOINTS_DEG)]
    tgt = data["target_end_link"]
    pos = tuple(float(v) for v in tgt["position_m"])
    rpy = tuple(float(v) for v in tgt["orientation_rpy_rad"])
    goal_deg_raw = data.get("goal_joints_deg")
    goal_deg = [float(v) for v in goal_deg_raw] if goal_deg_raw is not None else None
    return start_deg, pos, rpy, goal_deg


def pose_error(
    current_pos: Sequence[float],
    current_rpy: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
) -> Tuple[float, float]:
    pos_err_m = float(np.linalg.norm(np.asarray(current_pos) - np.asarray(target_pos)))
    # Orientation error: angle between rotation matrices via quaternion dot.
    cq = p.getQuaternionFromEuler(list(current_rpy))
    tq = p.getQuaternionFromEuler(list(target_rpy))
    dot = abs(sum(a * b for a, b in zip(cq, tq)))
    dot = min(1.0, max(-1.0, dot))
    ori_err_rad = 2.0 * math.acos(dot)
    return pos_err_m, ori_err_rad


def solve_goal_configuration(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    start_q: Sequence[float],
    goal_joints_deg: Optional[Sequence[float]] = None,
    pos_tol_m: float = 1e-4,
) -> Optional[List[float]]:
    """Goal joints: prefer teleop ``goal_joints_deg`` (matches env_robot_control_sim), else Cartesian IK."""
    if goal_joints_deg is not None and len(goal_joints_deg) >= len(scene.joint_indices):
        q_goal = joint_array_from_transformed_deg(scene, goal_joints_deg)
        if not is_state_valid(scene, q_goal):
            print("Warning: goal_joints_deg is in collision; falling back to Cartesian IK.")
        else:
            pos_err, ori_err = measure_pose_error(scene, target_pos, target_rpy, q_goal)
            print(
                f"Goal from joint angles (teleop): pos_err={pos_err*1000:.3f} mm, "
                f"ori_err={math.degrees(ori_err):.2f} deg"
            )
            return q_goal

    target_orn = p.getQuaternionFromEuler(list(target_rpy))
    rest_candidates: List[List[float]] = [list(start_q)]
    rng = np.random.default_rng(42)
    low = np.asarray(scene.lower_limits)
    high = np.asarray(scene.upper_limits)
    best_q: Optional[List[float]] = None
    best_pos_err = float("inf")

    for i in range(60):
        if i > 0:
            rest_candidates.append((low + rng.random(len(low)) * (high - low)).tolist())
        try:
            q_seed = solve_ik(scene, target_pos, target_orn, rest_candidates[-1])
            q_goal = refine_joint_config_to_pose(scene, target_pos, target_rpy, q_seed, pos_tol_m=pos_tol_m)
        except Exception:
            continue
        if not is_state_valid(scene, q_goal):
            continue
        set_joint_positions(scene, q_goal)
        pos, rpy = get_end_link_pose(scene)
        pos_err, ori_err = pose_error(pos, rpy, target_pos, target_rpy)
        if pos_err < best_pos_err:
            best_pos_err = pos_err
            best_q = q_goal
        if pos_err <= pos_tol_m and ori_err < 0.35:
            print(f"Goal IK: pos_err={pos_err*1000:.3f} mm, ori_err={math.degrees(ori_err):.2f} deg")
            return q_goal

    if best_q is not None:
        set_joint_positions(scene, best_q)
        pos, rpy = get_end_link_pose(scene)
        pos_err, ori_err = pose_error(pos, rpy, target_pos, target_rpy)
        print(f"Goal IK (best effort): pos_err={pos_err*1000:.3f} mm, ori_err={math.degrees(ori_err):.2f} deg")
        if pos_err <= pos_tol_m * 1.5:
            return best_q
    return None


def plan_trajectory(
    scene: PlanningScene,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    seed: int = 42,
) -> Optional[np.ndarray]:
    valid = lambda q: is_state_valid(scene, q)  # noqa: E731

    print(f"Planning with RRTConnect (seed={seed}, PyBullet collision checker)...")
    planner = RRTConnectPlanner(
        scene, valid, step_size=0.10, max_iterations=12000, goal_tolerance=0.12, rng=random.Random(seed)
    )
    path = planner.plan(start_q, goal_q)
    if path is None:
        print("RRTConnect failed to find a path.")
        return None
    print(f"RRTConnect: {len(path)} waypoints, path length={sum(joint_distance(path[i], path[i+1]) for i in range(len(path)-1)):.3f} rad")

    start_arr = np.asarray(start_q, dtype=np.float64)
    goal_arr = np.asarray(goal_q, dtype=np.float64)
    path[0] = start_arr
    path[-1] = goal_arr

    sparse = interpolate_path_dense(path, samples_per_segment=8)
    sparse[0] = start_arr
    sparse[-1] = goal_arr
    print(f"CHOMP smoothing ({len(sparse)} samples)...")
    smooth = chomp_smooth(
        scene,
        sparse,
        collision_cost_fn=lambda q: collision_cost(scene, q),
        is_valid=valid,
        start_q=start_arr,
        goal_q=goal_arr,
        n_iterations=60,
    )
    dense = resample_trajectory(smooth, n_samples=150)
    dense = pin_trajectory_endpoints(dense, start_arr, goal_arr)
    if not all(valid(dense[i]) for i in range(len(dense))):
        print("Warning: CHOMP path has collisions; using RRTConnect path resampled.")
        sparse[0] = start_arr
        sparse[-1] = goal_arr
        dense = pin_trajectory_endpoints(resample_trajectory(sparse, n_samples=150), start_arr, goal_arr)

    set_joint_positions(scene, dense[-1])
    pos, rpy = get_end_link_pose(scene)
    pos_err, ori_err = pose_error(pos, rpy, target_pos, target_rpy)
    print(f"Planned goal FK: pos_err={pos_err*1000:.3f} mm, ori_err={math.degrees(ori_err):.2f} deg")
    print(f"Final trajectory: {len(dense)} samples")
    return dense


# --- Compare GUI (target vs current end link) ---------------------------------

def run_compare_status_window() -> None:
    try:
        import tkinter as tk
    except Exception as exc:
        raise SystemExit(f"Tkinter not available: {exc}") from exc

    import queue
    import threading

    root = tk.Tk()
    root.title("End Link: Target vs Current")
    root.geometry("520x320")
    root.attributes("-topmost", True)

    target_pos_var = tk.StringVar(value="Target pos (m): -")
    target_rpy_var = tk.StringVar(value="Target rpy (rad): -")
    current_pos_var = tk.StringVar(value="Current pos (m): -")
    current_rpy_var = tk.StringVar(value="Current rpy (rad): -")
    err_var = tk.StringVar(value="Error: -")
    phase_var = tk.StringVar(value="Phase: -")
    note_var = tk.StringVar(value="")

    frame = tk.Frame(root, padx=12, pady=10)
    frame.pack(fill="both", expand=True)
    tk.Label(frame, textvariable=phase_var, font=("Helvetica", 12, "bold")).pack(anchor="w")
    tk.Label(frame, text="Target", font=("Helvetica", 11, "bold"), foreground="#004488").pack(anchor="w", pady=(8, 0))
    tk.Label(frame, textvariable=target_pos_var, font=("Consolas", 11)).pack(anchor="w")
    tk.Label(frame, textvariable=target_rpy_var, font=("Consolas", 11)).pack(anchor="w")
    tk.Label(frame, text="Current", font=("Helvetica", 11, "bold"), foreground="#228822").pack(anchor="w", pady=(8, 0))
    tk.Label(frame, textvariable=current_pos_var, font=("Consolas", 11)).pack(anchor="w")
    tk.Label(frame, textvariable=current_rpy_var, font=("Consolas", 11)).pack(anchor="w")
    tk.Label(frame, textvariable=err_var, font=("Consolas", 12, "bold"), foreground="#aa4400").pack(anchor="w", pady=(10, 0))
    tk.Label(frame, textvariable=note_var, font=("Helvetica", 10), foreground="#880000").pack(anchor="w", pady=(4, 0))

    message_queue: queue.Queue[Optional[str]] = queue.Queue()

    def read_stdin_thread() -> None:
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    message_queue.put(None)
                    break
                message_queue.put(line.strip())
        except Exception:
            message_queue.put(None)

    threading.Thread(target=read_stdin_thread, daemon=True).start()

    def apply_payload(payload: Dict[str, object]) -> None:
        phase_var.set(str(payload.get("phase", "")))
        note_var.set(str(payload.get("note", "")))
        tp = payload.get("target_pos", [0, 0, 0])
        tr = payload.get("target_rpy", [0, 0, 0])
        cp = payload.get("current_pos", [0, 0, 0])
        cr = payload.get("current_rpy", [0, 0, 0])
        target_pos_var.set(f"Target pos (m): ({tp[0]:+.4f}, {tp[1]:+.4f}, {tp[2]:+.4f})")
        target_rpy_var.set(f"Target rpy (rad): ({tr[0]:+.4f}, {tr[1]:+.4f}, {tr[2]:+.4f})")
        current_pos_var.set(f"Current pos (m): ({cp[0]:+.4f}, {cp[1]:+.4f}, {cp[2]:+.4f})")
        current_rpy_var.set(f"Current rpy (rad): ({cr[0]:+.4f}, {cr[1]:+.4f}, {cr[2]:+.4f})")
        pe = float(payload.get("pos_err_mm", 0.0))
        oe = float(payload.get("ori_err_deg", 0.0))
        err_var.set(f"Error: position {pe:.2f} mm  |  orientation {oe:.2f} deg")

    def poll() -> None:
        try:
            while True:
                try:
                    msg = message_queue.get_nowait()
                except queue.Empty:
                    break
                if msg is None:
                    root.destroy()
                    return
                try:
                    payload = json.loads(msg)
                    if isinstance(payload, dict):
                        apply_payload(payload)
                except Exception:
                    pass
        except Exception:
            pass
        root.after(50, poll)

    root.after(50, poll)
    root.mainloop()


def start_compare_window(script_path: Path) -> Optional[subprocess.Popen[str]]:
    cmd = [sys.executable, str(script_path), "--compare-window"]
    try:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except Exception:
        return None


def send_compare_update(
    proc: Optional[subprocess.Popen[str]],
    phase: str,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    current_pos: Sequence[float],
    current_rpy: Sequence[float],
    note: str = "",
) -> None:
    if proc is None or proc.stdin is None:
        return
    pos_err_m, ori_err_rad = pose_error(current_pos, current_rpy, target_pos, target_rpy)
    payload = {
        "phase": phase,
        "note": note,
        "target_pos": list(target_pos),
        "target_rpy": list(target_rpy),
        "current_pos": list(current_pos),
        "current_rpy": list(current_rpy),
        "pos_err_mm": pos_err_m * 1000.0,
        "ori_err_deg": math.degrees(ori_err_rad),
    }
    try:
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
    except Exception:
        pass


def stop_compare_window(proc: Optional[subprocess.Popen[str]]) -> None:
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass


def playback_trajectory(
    scene: PlanningScene,
    trajectory: np.ndarray,
    goal_q: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    compare_proc: Optional[subprocess.Popen[str]],
    hz: float = 40.0,
    max_joint_step_rad: float = 0.012,
    settle_s: float = 1.0,
) -> None:
    """Kinematic playback only (no motors, no gravity) with capped joint step per frame."""
    dt = 1.0 / max(1.0, hz)
    goal_arr = np.asarray(goal_q, dtype=np.float64)
    trajectory = prepare_playback_trajectory(trajectory, trajectory[0], goal_arr)

    # Gravity was moving the arm away from commanded joint angles between frames.
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

    print("Playing trajectory (kinematic, deterministic endpoints). Close GUI to exit.")
    set_joint_positions(scene, trajectory[0])
    pos, rpy = get_end_link_pose(scene)
    send_compare_update(compare_proc, "Settled at HOME", target_pos, target_rpy, pos, rpy)
    t_end = time.perf_counter() + settle_s
    while time.perf_counter() < t_end and p.isConnected():
        set_joint_positions(scene, trajectory[0])
        time.sleep(dt)

    frame = 0
    for i in range(len(trajectory) - 1):
        if not p.isConnected():
            break
        q0 = trajectory[i]
        q1 = trajectory[i + 1]
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
                    f"Playback {i+1}/{len(trajectory)-1}",
                    target_pos,
                    target_rpy,
                    pos,
                    rpy,
                )
            time.sleep(dt)

    # Hard snap to planned goal joints (same config used for planning verification).
    t_end = time.perf_counter() + 2.5
    while time.perf_counter() < t_end and p.isConnected():
        set_joint_positions(scene, goal_arr)
        pos, rpy = get_end_link_pose(scene)
        send_compare_update(compare_proc, "At goal", target_pos, target_rpy, pos, rpy)
        time.sleep(dt)

    pos, rpy = get_end_link_pose(scene)
    pos_err, ori_err = pose_error(pos, rpy, target_pos, target_rpy)
    send_compare_update(
        compare_proc,
        "At goal",
        target_pos,
        target_rpy,
        pos,
        rpy,
        note=f"Final: {pos_err*1000:.3f} mm, {math.degrees(ori_err):.2f} deg",
    )
    print(f"Final end-link error: position {pos_err*1000:.3f} mm, orientation {math.degrees(ori_err):.2f} deg")

    while p.isConnected():
        set_joint_positions(scene, goal_arr)
        time.sleep(dt)


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    default_env = root / "physical_env" / "packaging_station_edit.obj"
    parser = argparse.ArgumentParser(description="RRTConnect + CHOMP plan and PyBullet demo.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--robot", type=str, default="Dummy_ros2")
    parser.add_argument("--env", type=Path, default=default_env)
    parser.add_argument("--targets", type=Path, default=_targets_path())
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--end-link", type=str, default="")
    parser.add_argument("--plan-only", action="store_true", help="Plan and print summary without GUI playback.")
    parser.add_argument("--replay", action="store_true", help="Replay last_plan.npz (skip replanning).")
    parser.add_argument("--seed", type=int, default=42, help="RRTConnect RNG seed (same seed => same path).")
    parser.add_argument(
        "--verify-coordinates",
        action="store_true",
        help="Print frame/goal FK check vs env_robot_control_sim setup, then exit.",
    )
    parser.add_argument("--compare-window", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def verify_coordinate_alignment(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    start_q: Sequence[float],
    goal_joints_deg: Optional[Sequence[float]],
) -> None:
    """Confirm planner uses the same world frame as env_robot_control_sim (base Z+180°, world FK)."""
    print("--- Coordinate verification (trajectory_planner vs env_robot_control_sim) ---")
    print(f"Robot base: world origin, orientation = 180 deg about Z (same as control sim)")
    print(f"Environment: world origin, identity orientation")
    print(f"End link: [{scene.end_link_idx}] {scene.end_link_name}")
    print(f"FK uses: p.getLinkState -> linkWorldPosition / linkWorldOrientation (absolute world frame)")
    set_joint_positions(scene, start_q)
    pos0, rpy0 = get_end_link_pose(scene)
    print(f"HOME FK pos (m): {pos0}")
    print(f"HOME FK rpy (rad): {rpy0}")
    goal_q = solve_goal_configuration(scene, target_pos, target_rpy, start_q, goal_joints_deg)
    if goal_q is None:
        print("FAIL: could not resolve goal configuration")
        return
    pos_err, ori_err = measure_pose_error(scene, target_pos, target_rpy, goal_q)
    print(f"Target pos (m): {target_pos}")
    print(f"Target rpy (rad): {target_rpy}")
    pos, rpy = get_end_link_pose(scene)
    print(f"Goal FK pos (m): {pos}")
    print(f"Goal FK rpy (rad): {rpy}")
    print(f"Error: position {pos_err*1000:.3f} mm | orientation {math.degrees(ori_err):.2f} deg")
    if pos_err > 0.2e-3:
        print("Hint: paste joint angles from env_robot_control_sim into plan_targets.json as goal_joints_deg.")
    print("---")


def main() -> None:
    args = parse_args()
    if args.compare_window:
        run_compare_status_window()
        return

    start_deg, target_pos, target_rpy, goal_joints_deg = load_plan_targets(args.targets.resolve())
    print(f"Targets file: {args.targets}")
    print(f"Start (deg): {start_deg}")
    print(f"Target end pos (m): {target_pos}")
    print(f"Target end rpy (rad): {target_rpy}")
    if goal_joints_deg is not None:
        print(f"Goal joints (deg, from teleop): {goal_joints_deg}")

    root = args.root.resolve()
    candidates = discover_descriptions(root)
    selected = choose_description(candidates, args.robot, root)
    package_map = build_package_map(root)
    urdf_path, temp_file = materialize_urdf(selected, package_map)
    extra = {root, selected.parent, urdf_path.parent, *package_map.values()}

    try:
        cached = load_plan_cache() if args.replay else None
        if args.replay and cached is None:
            raise SystemExit(f"No cached plan at {_plan_cache_path()}. Run once without --replay first.")
        if cached is not None:
            print(f"Replaying cached plan (seed={cached['seed']}) from {_plan_cache_path()}")
            trajectory = cached["trajectory"]
            goal_q = cached["goal_q"]
            start_q = cached["start_q"]
            target_pos = cached["target_pos"]
            target_rpy = cached["target_rpy"]
            start_deg = cached["start_deg"]
        else:
            print("Building planning scene (DIRECT mode)...")
            scene = build_planning_scene(
                urdf_path=urdf_path,
                env_path=args.env,
                mesh_scale=args.mesh_scale,
                fixed_base=True,
                extra_search_paths=extra,
                load_plane=False,
                end_link_hint=args.end_link,
                use_gui=False,
            )
            start_q = joint_array_from_transformed_deg(scene, start_deg)
            set_joint_positions(scene, start_q)
            if not is_state_valid(scene, start_q):
                raise SystemExit("Start configuration is in collision.")

            if args.verify_coordinates:
                verify_coordinate_alignment(scene, target_pos, target_rpy, start_q, goal_joints_deg)
                p.disconnect()
                return

            goal_q = solve_goal_configuration(scene, target_pos, target_rpy, start_q, goal_joints_deg)
            if goal_q is None:
                raise SystemExit("Could not find a valid IK goal configuration.")

            trajectory = plan_trajectory(scene, start_q, goal_q, target_pos, target_rpy, seed=args.seed)
            p.disconnect()
            if trajectory is None:
                raise SystemExit("Planning failed.")

            save_plan_cache(trajectory, goal_q, start_q, target_pos, target_rpy, start_deg, args.seed)
            print(f"Saved plan cache to {_plan_cache_path()}")

            if args.plan_only:
                print(f"Trajectory samples: {len(trajectory)}")
                return

        if args.plan_only and cached is None:
            return

        script_path = Path(__file__).resolve()
        compare_proc = start_compare_window(script_path)
        send_compare_update(
            compare_proc,
            "Planning complete",
            target_pos,
            target_rpy,
            target_pos,
            target_rpy,
            note="Opening simulation...",
        )

        print("Opening simulation (GUI)...")
        scene_gui = build_planning_scene(
            urdf_path=urdf_path,
            env_path=args.env,
            mesh_scale=args.mesh_scale,
            fixed_base=True,
            extra_search_paths=extra,
            load_plane=False,
            end_link_hint=args.end_link,
            use_gui=True,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        frame_camera_on_scene(scene_gui)

        # Draw target frame marker
        target_orn = p.getQuaternionFromEuler(list(target_rpy))
        p.addUserDebugLine(target_pos, [target_pos[0] + 0.05, target_pos[1], target_pos[2]], [1, 0, 0], 3, 0)
        p.addUserDebugLine(target_pos, [target_pos[0], target_pos[1] + 0.05, target_pos[2]], [0, 1, 0], 3, 0)
        p.addUserDebugLine(target_pos, [target_pos[0], target_pos[1], target_pos[2] + 0.05], [0, 0, 1], 3, 0)

        # Re-apply refined goal in GUI scene so FK matches planning (same URDF, zero physics drift).
        if goal_joints_deg is None:
            goal_q = refine_joint_config_to_pose(scene_gui, target_pos, target_rpy, goal_q)
        trajectory = prepare_playback_trajectory(trajectory, start_q, goal_q)

        set_joint_positions(scene_gui, trajectory[0])
        set_joint_positions(scene_gui, goal_q)
        pos_check, rpy_check = get_end_link_pose(scene_gui)
        pos_err, ori_err = pose_error(pos_check, rpy_check, target_pos, target_rpy)
        print(f"GUI goal check: pos_err={pos_err*1000:.3f} mm, ori_err={math.degrees(ori_err):.2f} deg")

        send_compare_update(
            compare_proc,
            "Ready",
            target_pos,
            target_rpy,
            pos_check,
            rpy_check,
            note=f"Goal preview: {pos_err*1000:.3f} mm",
        )
        set_joint_positions(scene_gui, trajectory[0])
        time.sleep(0.5)

        try:
            playback_trajectory(scene_gui, trajectory, goal_q, target_pos, target_rpy, compare_proc)
        finally:
            stop_compare_window(compare_proc)
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
