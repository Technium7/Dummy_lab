"""Plan HOME -> target with discrete iLQR, play back in PyBullet.

Edit targets in plan_targets.json. Run from repo root:

  python software/trajectory_planner/plan_and_demo.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

try:
    import pybullet as p
except ImportError as exc:
    raise SystemExit("Missing dependency: pybullet. Install with: pip install pybullet") from exc

from movement_sim.joint_controller import (  # noqa: E402
    build_package_map,
    choose_description,
    discover_descriptions,
    materialize_urdf,
)
from trajectory_planner.collision import is_state_valid  # noqa: E402
from trajectory_planner.collision_report import run_collision_analysis  # noqa: E402
from trajectory_planner.compare_window import (  # noqa: E402
    run_compare_status_window,
    send_compare_update,
    start_compare_window,
    stop_compare_window,
)
from trajectory_planner.config import (  # noqa: E402
    DEFAULT_CLEARANCE_MARGIN,
    DEFAULT_PLANNER_COLLISION_STRIDE,
    DEFAULT_SCP_CONSTRAINT_WEIGHT,
)
from trajectory_planner.goals import (  # noqa: E402
    solve_goal_configuration,
    solve_goal_from_pose,
)
from trajectory_planner.playback import playback_trajectory  # noqa: E402
from trajectory_planner.planning import (  # noqa: E402
    finalize_trajectory_for_cache,
    plan_joint_trajectory,
)
from trajectory_planner.scene import (  # noqa: E402
    build_planning_scene,
    disconnect_planning_scene,
    frame_camera_on_scene,
    get_end_link_pose,
    joint_array_from_transformed_deg,
    measure_pose_error,
    pose_error_rad,
    set_joint_positions,
)
from trajectory_planner.targets import (  # noqa: E402
    load_plan_cache,
    load_plan_targets,
    plan_cache_path,
    save_plan_cache,
    targets_path,
)
from trajectory_planner.trajectory_utils import (  # noqa: E402
    assert_trajectory_endpoints,
    compress_duplicate_waypoints,
    prepare_playback_trajectory,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    default_env = root / "physical_env" / "packaging_station_edit.obj"
    parser = argparse.ArgumentParser(description="iLQR joint-space plan and PyBullet demo.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--robot", type=str, default="Dummy_ros2")
    parser.add_argument("--env", type=Path, default=default_env)
    parser.add_argument("--targets", type=Path, default=targets_path())
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--end-link", type=str, default="")
    parser.add_argument("--plan-only", action="store_true", help="Plan and print summary without GUI playback.")
    parser.add_argument("--replay", action="store_true", help="Replay last_plan.npz (skip replanning).")
    parser.add_argument("--seed", type=int, default=42, help="Stored in plan cache metadata.")
    parser.add_argument(
        "--goal-pose",
        action="store_true",
        help="IK-resolve target_end_link (ignore goal_joints_deg); plan tracks IK joints at goal.",
    )
    parser.add_argument(
        "--collision-penalty",
        action="store_true",
        help="Enable SCP collision planning (slower; default is off).",
    )
    parser.add_argument(
        "--clearance-margin",
        type=float,
        default=DEFAULT_CLEARANCE_MARGIN,
        metavar="M",
        help="Required env clearance during SCP (m); default 3 mm. Auto-tapers at goal.",
    )
    parser.add_argument(
        "--scp-weight",
        type=float,
        default=DEFAULT_SCP_CONSTRAINT_WEIGHT,
        help="SCP constraint penalty weight (lower = smoother; default 60).",
    )
    parser.add_argument(
        "--planner-collision-stride",
        type=int,
        default=DEFAULT_PLANNER_COLLISION_STRIDE,
        metavar="N",
        help="Evaluate collision every N planner waypoints (default 12).",
    )
    parser.add_argument(
        "--show-collision",
        action="store_true",
        help="Print planner collision metrics before GUI playback.",
    )
    parser.add_argument(
        "--collision-stride",
        type=int,
        default=0,
        help="Print every Nth waypoint during collision analysis (0=auto).",
    )
    parser.add_argument(
        "--collision-exact",
        action="store_true",
        help="Per-link clearance (slow); used for analysis and --collision-penalty.",
    )
    parser.add_argument(
        "--verify-coordinates",
        action="store_true",
        help="Print frame/goal FK check vs env_robot_control_sim, then exit.",
    )
    parser.add_argument("--compare-window", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def verify_coordinate_alignment(
    scene,
    target_pos,
    target_rpy,
    start_q,
    goal_joints_deg,
) -> None:
    print("--- Coordinate verification (trajectory_planner vs env_robot_control_sim) ---")
    print("Robot base: world origin, 180 deg about Z (same as control sim)")
    print("Environment: world origin, identity orientation")
    print(f"End link: [{scene.end_link_idx}] {scene.end_link_name}")
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
        print("Hint: paste joint angles from env_robot_control_sim into plan_targets.json.")
    print("---")


def _draw_target_axes(target_pos) -> None:
    p.addUserDebugLine(
        target_pos,
        [target_pos[0] + 0.05, target_pos[1], target_pos[2]],
        [1, 0, 0],
        3,
        0,
    )
    p.addUserDebugLine(
        target_pos,
        [target_pos[0], target_pos[1] + 0.05, target_pos[2]],
        [0, 1, 0],
        3,
        0,
    )
    p.addUserDebugLine(
        target_pos,
        [target_pos[0], target_pos[1], target_pos[2] + 0.05],
        [0, 0, 1],
        3,
        0,
    )


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
            raise SystemExit(f"No cached plan at {plan_cache_path()}. Run once without --replay first.")

        if cached is not None:
            print(f"Replaying cached plan (seed={cached['seed']}) from {plan_cache_path()}")
            trajectory = cached["trajectory"]
            goal_q = cached["goal_q"]
            start_q = cached["start_q"]
            target_pos = cached["target_pos"]
            target_rpy = cached["target_rpy"]
            start_deg = cached["start_deg"]
            cached_goal_pose = cached.get("goal_pose")
            if cached_goal_pose is not None and cached_goal_pose != args.goal_pose:
                print(
                    "Warning: --goal-pose flag differs from cached plan; "
                    "replay uses cached goal_q and targets."
                )
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
                disconnect_planning_scene(scene)
                return

            if args.goal_pose:
                goal_q = solve_goal_from_pose(scene, target_pos, target_rpy, start_q)
                if goal_q is None:
                    raise SystemExit("Could not IK-resolve goal from target_end_link pose.")
                print("Planner goal: IK joints from target_end_link (terminal joint tracking)")
            else:
                goal_q = solve_goal_configuration(
                    scene, target_pos, target_rpy, start_q, goal_joints_deg
                )
                if goal_q is None:
                    raise SystemExit("Could not find a valid goal configuration.")
                print("Planner goal: teleop goal_joints_deg or IK fallback")

            trajectory = plan_joint_trajectory(
                scene,
                start_q,
                goal_q,
                target_pos,
                target_rpy,
                enable_collision_penalty=args.collision_penalty,
                clearance_margin=args.clearance_margin,
                exact_clearance=args.collision_exact,
                collision_stride=args.planner_collision_stride,
                scp_constraint_weight=args.scp_weight,
            )
            disconnect_planning_scene(scene)
            if trajectory is None:
                raise SystemExit("Planning failed.")

            trajectory = finalize_trajectory_for_cache(trajectory, start_q, goal_q)
            save_plan_cache(
                trajectory,
                goal_q,
                start_q,
                target_pos,
                target_rpy,
                start_deg,
                args.seed,
                goal_pose=args.goal_pose,
                collision_penalty=args.collision_penalty,
            )
            print(f"Saved plan cache to {plan_cache_path()}")

            if args.plan_only:
                print(f"Trajectory samples: {len(trajectory)}")
                return

        if args.plan_only and cached is None:
            return

        trajectory_dense = np.asarray(trajectory, dtype=np.float64)
        goal_q_arr = np.asarray(goal_q, dtype=np.float64)
        from_replay = cached is not None
        try:
            assert_trajectory_endpoints(trajectory_dense, start_q, goal_q_arr)
        except ValueError as exc:
            if from_replay:
                raise SystemExit(str(exc)) from exc
            print(f"Warning: {exc}")

        if args.show_collision:
            trajectory_col = prepare_playback_trajectory(
                trajectory_dense, start_q, goal_q, hold_steps=0
            )
            stride = args.collision_stride
            if stride <= 0:
                stride = max(1, len(trajectory_col) // 40)
            scene_col = build_planning_scene(
                urdf_path=urdf_path,
                env_path=args.env,
                mesh_scale=args.mesh_scale,
                fixed_base=True,
                extra_search_paths=extra,
                load_plane=False,
                end_link_hint=args.end_link,
                use_gui=False,
            )
            try:
                run_collision_analysis(
                    scene_col,
                    trajectory_col,
                    stride=stride,
                    exact_clearance=args.collision_exact,
                )
            finally:
                disconnect_planning_scene(scene_col)

        print("Opening simulation (GUI)...")
        script_path = Path(__file__).resolve()
        compare_proc = start_compare_window(script_path)
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
        _draw_target_axes(target_pos)

        # Faithful playback: no re-IK, no teleop override, no endpoint rewrite on replay.
        trajectory = prepare_playback_trajectory(
            trajectory_dense,
            start_q,
            goal_q_arr,
            pin_endpoints=not from_replay,
        )
        trajectory = compress_duplicate_waypoints(trajectory)

        set_joint_positions(scene_gui, trajectory[0])
        set_joint_positions(scene_gui, trajectory[-1])
        pos_check, rpy_check = get_end_link_pose(scene_gui)
        pos_err, ori_err = pose_error_rad(pos_check, rpy_check, target_pos, target_rpy)
        print(
            f"GUI goal check: pos_err={pos_err*1000:.3f} mm, "
            f"ori_err={math.degrees(ori_err):.2f} deg"
        )

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
            playback_trajectory(
                scene_gui,
                trajectory,
                target_pos,
                target_rpy,
                compare_proc,
            )
        finally:
            stop_compare_window(compare_proc)
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
