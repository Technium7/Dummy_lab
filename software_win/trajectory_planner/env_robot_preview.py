"""PyBullet preview: table-frame world + robot at physical_env/robot_in_table.yaml.

Usage from repo root (Dummy_lab/):
  $env:PYTHONPATH = "$PWD\\software_win"
  python -m trajectory_planner.env_robot_preview
  python -m trajectory_planner.env_robot_preview --env physical_env/packaging_station_edit.obj

Planner / preview world = optical table frame (OBJ origin = calib 0,0).
Robot base is placed at ``robot_in_table_frame`` from
``physical_env/robot_in_table.yaml`` (same as grab_sensor planning).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    raise SystemExit("Missing dependency: pybullet. Install with: pip install pybullet") from exc

from movement_sim.joint_controller import (  # noqa: E402
    build_package_map,
    choose_description,
    discover_descriptions,
    joint_limits_or_defaults,
    materialize_urdf,
)
from robot_config import DEFAULT_ROBOT, repo_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_env = root / "physical_env" / "packaging_station_edit.obj"
    default_frames = root / "physical_env" / "robot_in_table.yaml"
    parser = argparse.ArgumentParser(
        description="Preview table-frame OBJ + robot at physical_env/robot_in_table.yaml."
    )
    parser.add_argument("--root", type=Path, default=root, help="Workspace root for URDF discovery.")
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT, help="URDF/Xacro path or name filter.")
    parser.add_argument("--env", type=Path, default=default_env, help="Wavefront .obj (table calib frame).")
    parser.add_argument(
        "--frames",
        type=Path,
        default=default_frames,
        help="Shared robot_in_table YAML (default: physical_env/robot_in_table.yaml).",
    )
    parser.add_argument(
        "--robot-in-table-in",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override robot base in table frame (inches).",
    )
    parser.add_argument(
        "--robot-in-table-rpy-deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "P", "Y"),
        help="Override robot RPY in table frame (deg).",
    )
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="Uniform scale applied to the OBJ mesh.")
    parser.add_argument("--fixed-base", action="store_true", default=True, help="Load robot with fixed base.")
    parser.add_argument("--floating-base", action="store_true", help="Allow robot base to move.")
    parser.add_argument("--no-plane", action="store_true", help="Do not load PyBullet's ground plane.")
    return parser.parse_args()


def _resolve_env_path(env: Path, root: Path) -> Path:
    path = env.expanduser()
    if not path.is_absolute():
        cwd_candidate = (Path.cwd() / path).resolve()
        root_candidate = (root / path).resolve()
        if cwd_candidate.is_file():
            return cwd_candidate
        if root_candidate.is_file():
            return root_candidate
        return root_candidate
    return path.resolve()


def _robot_in_table_from_args(args: argparse.Namespace):
    from high_level_jobs.grab_sensor.frame_config import resolve_robot_in_table_frame

    override = {}
    if args.robot_in_table_in is not None:
        override["translation_in"] = [float(v) for v in args.robot_in_table_in]
    if args.robot_in_table_rpy_deg is not None:
        override["rpy_deg"] = [float(v) for v in args.robot_in_table_rpy_deg]
    return resolve_robot_in_table_frame(
        shared_yaml=args.frames,
        override=override or None,
    )


def load_static_obj_mesh(
    obj_path: Path,
    mesh_scale: float,
    *,
    base_position: Sequence[float] | None = None,
    base_orientation: Sequence[float] | None = None,
) -> int:
    path = obj_path.resolve()
    if not path.is_file():
        raise SystemExit(f"Environment mesh not found: {path}")
    s = max(1e-9, float(mesh_scale))
    scale = [s, s, s]
    mesh_file = str(path)
    concave = getattr(p, "GEOM_FORCE_CONCAVE_TRIMESH", 0)
    try:
        col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file, meshScale=scale, flags=concave)
    except Exception:
        col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file, meshScale=scale)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=mesh_file, meshScale=scale)
    pos = [float(v) for v in (base_position or (0.0, 0.0, 0.0))[:3]]
    orn = [float(v) for v in (base_orientation or (0.0, 0.0, 0.0, 1.0))[:4]]
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orn,
    )


def sim_to_user(value: float, offset: float, sign: float) -> float:
    return sign * (offset - value)


def user_bounds_from_sim(lower: float, upper: float, offset: float, sign: float) -> Tuple[float, float]:
    lower_user = sim_to_user(lower, offset, sign)
    upper_user = sim_to_user(upper, offset, sign)
    return (min(lower_user, upper_user), max(lower_user, upper_user))


def _aabb_union(
    boxes: Iterable[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    mins: List[float] = []
    maxs: List[float] = []
    for mn, mx in boxes:
        mins.append(mn)
        maxs.append(mx)
    if not mins:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    out_min = (
        min(b[0] for b in mins),
        min(b[1] for b in mins),
        min(b[2] for b in mins),
    )
    out_max = (
        max(b[0] for b in maxs),
        max(b[1] for b in maxs),
        max(b[2] for b in maxs),
    )
    return out_min, out_max


def run_preview(
    urdf_path: Path,
    env_path: Path,
    mesh_scale: float,
    fixed_base: bool,
    extra_search_paths: Iterable[Path],
    load_plane: bool,
    *,
    robot_base_position: Sequence[float],
    robot_base_orientation: Sequence[float],
    robot_in_table_display: str,
) -> None:
    cid = p.connect(p.GUI)
    if cid < 0:
        raise SystemExit("Failed to open PyBullet GUI")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    for path in extra_search_paths:
        if path.exists() and path.is_dir():
            p.setAdditionalSearchPath(str(path))

    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    if load_plane:
        plane_path = Path(pybullet_data.getDataPath()) / "plane.urdf"
        if plane_path.exists():
            try:
                p.loadURDF(str(plane_path))
            except Exception:
                pass

    # Table frame = world: scanned mesh at origin (+90° X export fix).
    from high_level_jobs.grab_sensor.table_position import table_frame_pose_in_planner_world

    env_pos, env_orn = table_frame_pose_in_planner_world()
    env_id = load_static_obj_mesh(
        env_path,
        mesh_scale,
        base_position=env_pos,
        base_orientation=env_orn,
    )

    load_flags = (
        p.URDF_USE_INERTIA_FROM_FILE
        | p.URDF_USE_SELF_COLLISION
        | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    )
    robot_id = p.loadURDF(
        str(urdf_path),
        basePosition=[float(v) for v in robot_base_position[:3]],
        baseOrientation=[float(v) for v in robot_base_orientation[:4]],
        useFixedBase=fixed_base,
        flags=load_flags,
    )

    axis_len = 0.08
    rx, ry, rz = (float(v) for v in robot_base_position[:3])
    p.addUserDebugLine([0, 0, 0], [axis_len, 0, 0], [1, 0, 0], 3, 0)
    p.addUserDebugLine([0, 0, 0], [0, axis_len, 0], [0, 1, 0], 3, 0)
    p.addUserDebugLine([0, 0, 0], [0, 0, axis_len], [0, 0, 1], 3, 0)
    p.addUserDebugText("table 0,0", [0, 0, 0.02], [1, 1, 1], 1.2, 0)
    p.addUserDebugLine([rx, ry, rz], [rx + axis_len, ry, rz], [1, 0.4, 0.4], 2, 0)
    p.addUserDebugLine([rx, ry, rz], [rx, ry + axis_len, rz], [0.4, 1, 0.4], 2, 0)
    p.addUserDebugLine([rx, ry, rz], [rx, ry, rz + axis_len], [0.4, 0.4, 1], 2, 0)
    p.addUserDebugText("robot base", [rx, ry, rz + 0.02], [1, 1, 0], 1.2, 0)

    boxes = [p.getAABB(robot_id, -1), p.getAABB(env_id, -1)]
    aabb_min, aabb_max = _aabb_union(boxes)
    cx = (aabb_min[0] + aabb_max[0]) * 0.5
    cy = (aabb_min[1] + aabb_max[1]) * 0.5
    cz = (aabb_min[2] + aabb_max[2]) * 0.5
    span = max(
        abs(aabb_max[0] - aabb_min[0]),
        abs(aabb_max[1] - aabb_min[1]),
        abs(aabb_max[2] - aabb_min[2]),
    )
    p.resetDebugVisualizerCamera(
        cameraDistance=max(0.5, span) * 2.2,
        cameraYaw=50,
        cameraPitch=-28,
        cameraTargetPosition=[cx, cy, cz],
    )

    print(f"[env_robot_preview] robot_in_table: {robot_in_table_display}")
    print(
        f"[env_robot_preview] robot base at table/world "
        f"pos_m={[round(float(v), 4) for v in robot_base_position]}"
    )
    print("[env_robot_preview] env OBJ at table origin. Close GUI to exit.")

    joint_signs = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    joint_offsets_deg = [0.0, 73.0, 180.0, 0.0, 0.0, 0.0]

    def create_slider_for_joint(joint_idx: int) -> Optional[Tuple[int, int, str]]:
        info = p.getJointInfo(robot_id, joint_idx)
        joint_type = info[2]
        if joint_type not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            return None
        name = info[1].decode("utf-8", errors="replace")
        lower, upper = joint_limits_or_defaults(joint_type, float(info[8]), float(info[9]))
        current = float(p.getJointState(robot_id, joint_idx)[0])
        sign = joint_signs[joint_idx] if joint_idx < len(joint_signs) else 1.0
        if joint_type == p.JOINT_REVOLUTE and joint_idx < len(joint_offsets_deg):
            offset_rad = math.radians(joint_offsets_deg[joint_idx])
            current_user = math.degrees(sim_to_user(current, offset_rad, sign))
            lower_user, upper_user = user_bounds_from_sim(lower, upper, offset_rad, sign)
            lower_user = math.degrees(lower_user)
            upper_user = math.degrees(upper_user)
            slider_id = p.addUserDebugParameter(
                f"{name} (deg)", lower_user, upper_user, max(lower_user, min(current_user, upper_user))
            )
        else:
            current_user = sim_to_user(current, 0.0, sign)
            lower_user, upper_user = user_bounds_from_sim(lower, upper, 0.0, sign)
            slider_id = p.addUserDebugParameter(
                name, lower_user, upper_user, max(lower_user, min(current_user, upper_user))
            )
        return slider_id, joint_idx, name

    sliders: Dict[int, Tuple[int, str]] = {}
    for j in range(p.getNumJoints(robot_id)):
        created = create_slider_for_joint(j)
        if created is None:
            continue
        slider_id, joint_idx, name = created
        sliders[slider_id] = (joint_idx, name)

    while p.isConnected():
        for slider_id, (joint_idx, _name) in sliders.items():
            info = p.getJointInfo(robot_id, joint_idx)
            joint_type = info[2]
            raw = float(p.readUserDebugParameter(slider_id))
            sign = joint_signs[joint_idx] if joint_idx < len(joint_signs) else 1.0
            if joint_type == p.JOINT_REVOLUTE and joint_idx < len(joint_offsets_deg):
                offset_rad = math.radians(joint_offsets_deg[joint_idx])
                target = offset_rad - math.radians(raw) * sign
            else:
                target = -raw * sign if abs(sign + 1.0) < 1e-9 else raw * sign
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=500.0,
            )
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    fixed_base = args.fixed_base and not args.floating_base
    candidates = discover_descriptions(root)
    selected = choose_description(candidates, args.robot, root)
    package_map = build_package_map(root)
    urdf_path, temp_file = materialize_urdf(selected, package_map)
    extra = {root, selected.parent, urdf_path.parent, *package_map.values()}
    env_path = _resolve_env_path(Path(args.env), root)
    robot_in_table = _robot_in_table_from_args(args)
    from high_level_jobs.grab_sensor.table_position import (
        format_robot_in_table_display,
        robot_base_pose_in_table_world,
    )

    robot_pos, robot_orn = robot_base_pose_in_table_world(robot_in_table)
    try:
        run_preview(
            urdf_path=urdf_path,
            env_path=env_path,
            mesh_scale=args.mesh_scale,
            fixed_base=fixed_base,
            extra_search_paths=extra,
            load_plane=not args.no_plane,
            robot_base_position=robot_pos,
            robot_base_orientation=robot_orn,
            robot_in_table_display=format_robot_in_table_display(robot_in_table),
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
