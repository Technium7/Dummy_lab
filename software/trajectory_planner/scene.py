"""PyBullet planning scene: robot (Z-rotated 180°) + static OBJ environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from movement_sim.end_controller import get_controllable_joints, resolve_end_link

_BASE_ORIENTATION_Z_180 = p.getQuaternionFromEuler([0.0, 0.0, math.pi])


@dataclass
class PlanningScene:
    robot_id: int
    env_id: int
    joint_indices: List[int]
    lower_limits: List[float]
    upper_limits: List[float]
    joint_names: List[str]
    end_link_idx: int
    end_link_name: str
    client_id: int = 0


def disconnect_planning_scene(scene: PlanningScene) -> None:
    """Disconnect only this scene's PyBullet client (does not close other clients)."""
    try:
        if p.getConnectionInfo(physicsClientId=scene.client_id)["isConnected"]:
            p.disconnect(physicsClientId=scene.client_id)
    except Exception:
        pass


def load_static_obj_mesh(obj_path: Path, mesh_scale: float) -> int:
    path = obj_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Environment mesh not found: {path}")
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


def build_planning_scene(
    urdf_path: Path,
    env_path: Path | None,
    mesh_scale: float,
    fixed_base: bool,
    extra_search_paths: Iterable[Path],
    load_plane: bool,
    end_link_hint: str,
    use_gui: bool,
) -> PlanningScene:
    mode = p.GUI if use_gui else p.DIRECT
    cid = p.connect(mode)
    if cid < 0:
        raise RuntimeError("Failed to connect PyBullet")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    for path in extra_search_paths:
        if path.exists() and path.is_dir():
            p.setAdditionalSearchPath(str(path))

    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    if load_plane:
        plane_path = Path(pybullet_data.getDataPath()) / "plane.urdf"
        if plane_path.exists():
            try:
                p.loadURDF(str(plane_path))
            except Exception:
                pass

    if env_path is not None:
        env_id = load_static_obj_mesh(env_path, mesh_scale)
    else:
        env_id = -1
    robot_id = p.loadURDF(
        str(urdf_path),
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=_BASE_ORIENTATION_Z_180,
        useFixedBase=fixed_base,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    joint_indices, lowers, uppers, _, names = get_controllable_joints(robot_id)
    if not joint_indices:
        raise RuntimeError("No controllable joints on robot")
    end_link_idx = resolve_end_link(robot_id, end_link_hint)
    end_link_name = p.getJointInfo(robot_id, end_link_idx)[12].decode("utf-8", errors="replace")

    return PlanningScene(
        robot_id=robot_id,
        env_id=env_id,
        joint_indices=joint_indices,
        lower_limits=lowers,
        upper_limits=uppers,
        joint_names=names,
        end_link_idx=end_link_idx,
        end_link_name=end_link_name,
        client_id=cid,
    )


def set_joint_positions(scene: PlanningScene, q: Sequence[float]) -> None:
    for j_idx, value in zip(scene.joint_indices, q):
        p.resetJointState(scene.robot_id, j_idx, float(value), targetVelocity=0.0)


def get_end_link_pose(scene: PlanningScene) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    state = p.getLinkState(scene.robot_id, scene.end_link_idx, computeForwardKinematics=True)
    pos = tuple(float(v) for v in state[4])
    rpy = tuple(float(v) for v in p.getEulerFromQuaternion(state[5]))
    return pos, rpy


def frame_camera_on_scene(scene: PlanningScene) -> None:
    boxes = [p.getAABB(scene.robot_id, -1)]
    if scene.env_id >= 0:
        boxes.append(p.getAABB(scene.env_id, -1))
    mins = [b[0] for b in boxes]
    maxs = [b[1] for b in boxes]
    aabb_min = (min(v[0] for v in mins), min(v[1] for v in mins), min(v[2] for v in mins))
    aabb_max = (max(v[0] for v in maxs), max(v[1] for v in maxs), max(v[2] for v in maxs))
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


def _full_rest_poses(scene: PlanningScene, controllable_q: Sequence[float]) -> List[float]:
    """Build per-body rest pose array (all PyBullet joints), not only the arm subset."""
    num_joints = p.getNumJoints(scene.robot_id)
    rest = [0.0] * num_joints
    for j_idx, value in zip(scene.joint_indices, controllable_q):
        rest[j_idx] = float(value)
    return rest


def solve_ik(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_orn_quat: Sequence[float],
    rest_pose: Sequence[float],
) -> List[float]:
    n = len(scene.joint_indices)
    lowers = scene.lower_limits
    uppers = scene.upper_limits
    ranges = [max(1e-6, u - lo) for lo, u in zip(lowers, uppers)]
    sol = p.calculateInverseKinematics(
        bodyUniqueId=scene.robot_id,
        endEffectorLinkIndex=scene.end_link_idx,
        targetPosition=list(target_pos),
        targetOrientation=list(target_orn_quat),
        lowerLimits=lowers,
        upperLimits=uppers,
        jointRanges=ranges,
        restPoses=_full_rest_poses(scene, rest_pose),
        maxNumIterations=500,
        residualThreshold=1e-8,
    )
    return [float(sol[j]) for j in scene.joint_indices[:n]]


def pose_error_rad(
    current_pos: Sequence[float],
    current_rpy: Sequence[float],
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
) -> Tuple[float, float]:
    """Position error (m) and orientation error (rad) between two world-frame poses."""
    pos_err_m = float(np.linalg.norm(np.asarray(current_pos) - np.asarray(target_pos)))
    cq = p.getQuaternionFromEuler(list(current_rpy))
    tq = p.getQuaternionFromEuler(list(target_rpy))
    dot = min(1.0, max(-1.0, abs(sum(a * b for a, b in zip(cq, tq)))))
    ori_err_rad = 2.0 * math.acos(dot)
    return pos_err_m, ori_err_rad


def measure_pose_error(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    q: Sequence[float],
) -> Tuple[float, float]:
    set_joint_positions(scene, q)
    pos, rpy = get_end_link_pose(scene)
    return pose_error_rad(pos, rpy, target_pos, target_rpy)


def refine_joint_config_to_pose(
    scene: PlanningScene,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    q_init: Sequence[float],
    pos_tol_m: float = 1e-4,
    ori_tol_rad: float = 0.02,
) -> List[float]:
    """Coordinate-descent refinement on position + orientation (same world frame as env_robot_control_sim)."""
    low = np.asarray(scene.lower_limits, dtype=np.float64)
    high = np.asarray(scene.upper_limits, dtype=np.float64)
    q = np.clip(np.asarray(q_init, dtype=np.float64), low, high)

    def metric(q_try: np.ndarray) -> float:
        pos_err_m, ori_err_rad = measure_pose_error(scene, target_pos, target_rpy, q_try)
        # Weight orientation so teleop pose is reproduced, not just position.
        return pos_err_m * 1000.0 + ori_err_rad

    steps = [0.02, 0.005, 0.001, 0.0002, 0.00005, 0.00001]
    for step in steps:
        improved = True
        while improved:
            improved = False
            base = metric(q)
            for j in range(len(q)):
                for delta in (-step, step):
                    q_try = q.copy()
                    q_try[j] = float(np.clip(q_try[j] + delta, low[j], high[j]))
                    err = metric(q_try)
                    if err + 1e-15 < base:
                        q = q_try
                        base = err
                        improved = True
            pos_err_m, ori_err_rad = measure_pose_error(scene, target_pos, target_rpy, q)
            if pos_err_m <= pos_tol_m and ori_err_rad <= ori_tol_rad:
                break

    set_joint_positions(scene, q)
    return q.tolist()


def joint_array_from_transformed_deg(scene: PlanningScene, joints_deg: Sequence[float]) -> List[float]:
    from manual_control.mapping import transformed_deg_to_sim_rad

    out: List[float] = []
    for local_idx, joint_idx in enumerate(scene.joint_indices[: len(joints_deg)]):
        out.append(transformed_deg_to_sim_rad(local_idx, float(joints_deg[local_idx])))
    return out
