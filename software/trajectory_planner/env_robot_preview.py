"""PyBullet preview: default gripper URDF (or chosen URDF) at world origin with a static OBJ environment.

Usage — path depends on current directory:

  From repo root (Dummy_lab/):
    python software/trajectory_planner/env_robot_preview.py

  From software/ (do not prefix another software/):
    python trajectory_planner/env_robot_preview.py

  Options (same either way): --env ../physical_env/packaging_station_edit.obj

Robot base is placed at (0,0,0) in the same frame as the environment mesh (OBJ vertex coordinates).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    parser = argparse.ArgumentParser(description="Preview robot + static OBJ environment in PyBullet.")
    parser.add_argument("--root", type=Path, default=root, help="Workspace root for URDF discovery.")
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT, help="URDF/Xacro path or name filter.")
    parser.add_argument("--env", type=Path, default=default_env, help="Wavefront .obj for the static scene.")
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="Uniform scale applied to the OBJ mesh.")
    parser.add_argument("--fixed-base", action="store_true", default=True, help="Load robot with fixed base.")
    parser.add_argument("--floating-base", action="store_true", help="Allow robot base to move.")
    parser.add_argument("--no-plane", action="store_true", help="Do not load PyBullet's ground plane.")
    return parser.parse_args()


def load_static_obj_mesh(obj_path: Path, mesh_scale: float) -> int:
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
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
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
        min(a[0] for a in mins),
        min(a[1] for a in mins),
        min(a[2] for a in mins),
    )
    out_max = (
        max(a[0] for a in maxs),
        max(a[1] for a in maxs),
        max(a[2] for a in maxs),
    )
    return out_min, out_max


def run_preview(
    urdf_path: Path,
    env_path: Path,
    mesh_scale: float,
    fixed_base: bool,
    extra_search_paths: Iterable[Path],
    load_plane: bool,
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

    env_id = load_static_obj_mesh(env_path, mesh_scale)

    load_flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    robot_id = p.loadURDF(
        str(urdf_path),
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        useFixedBase=fixed_base,
        flags=load_flags,
    )

    boxes = [p.getAABB(robot_id, -1), p.getAABB(env_id, -1)]
    aabb_min, aabb_max = _aabb_union(boxes)
    cx = (aabb_min[0] + aabb_max[0]) * 0.5
    cy = (aabb_min[1] + aabb_max[1]) * 0.5
    cz = (aabb_min[2] + aabb_max[2]) * 0.5
    span = max(abs(aabb_max[0] - aabb_min[0]), abs(aabb_max[1] - aabb_min[1]), abs(aabb_max[2] - aabb_min[2]))
    p.resetDebugVisualizerCamera(
        cameraDistance=max(0.5, span) * 2.2,
        cameraYaw=50,
        cameraPitch=-28,
        cameraTargetPosition=[cx, cy, cz],
    )

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
            slider_id = p.addUserDebugParameter(name, lower_user, upper_user, max(lower_user, min(current_user, upper_user)))
        return joint_idx, slider_id, name

    joint_slider_map: Dict[int, Tuple[int, str]] = {}
    for joint_idx in range(p.getNumJoints(robot_id)):
        slider = create_slider_for_joint(joint_idx)
        if slider is not None:
            joint_slider_map[joint_idx] = (slider[1], slider[2])

    print("PyBullet preview: robot base and environment mesh both at world origin. Close the GUI to exit.")
    if not joint_slider_map:
        print("(No controllable joints — view only.)")

    while p.isConnected():
        for joint_idx, (slider_id, _) in joint_slider_map.items():
            try:
                target_user = p.readUserDebugParameter(slider_id)
            except Exception:
                continue
            info = p.getJointInfo(robot_id, joint_idx)
            sign = joint_signs[joint_idx] if joint_idx < len(joint_signs) else 1.0
            if info[2] == p.JOINT_REVOLUTE and joint_idx < len(joint_offsets_deg):
                offset_rad = math.radians(joint_offsets_deg[joint_idx])
                target = offset_rad - sign * math.radians(target_user)
            else:
                target = -sign * target_user
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
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
    try:
        run_preview(
            urdf_path=urdf_path,
            env_path=args.env,
            mesh_scale=args.mesh_scale,
            fixed_base=fixed_base,
            extra_search_paths=extra,
            load_plane=not args.no_plane,
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
