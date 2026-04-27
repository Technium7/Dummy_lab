from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    raise SystemExit("Missing dependency: pybullet. Install with: pip install pybullet") from exc


@dataclass
class TeleopSample:
    t: float
    joints_deg: List[float]
    joint_vel_deg_s: List[float]
    active_keys: List[str]


def joint_limits_or_defaults(joint_type: int, lower: float, upper: float) -> Tuple[float, float]:
    if lower <= upper and not (lower == 0.0 and upper == -1.0):
        return lower, upper
    if joint_type == p.JOINT_REVOLUTE:
        return -math.pi, math.pi
    if joint_type == p.JOINT_PRISMATIC:
        return -0.2, 0.2
    return -1.0, 1.0


def get_controllable_joints(robot_id: int) -> Tuple[List[int], List[str]]:
    indices: List[int] = []
    names: List[str] = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            continue
        indices.append(i)
        names.append(info[1].decode("utf-8", errors="replace"))
    return indices, names


def resolve_end_link(robot_id: int, hint: str) -> int:
    num_joints = p.getNumJoints(robot_id)
    if num_joints <= 0:
        return -1

    if hint:
        if hint.isdigit() or (hint.startswith("-") and hint[1:].isdigit()):
            idx = int(hint)
            if -1 <= idx < num_joints:
                return idx
            raise SystemExit(f"--end-link index out of range: {idx}")

        lookup = hint.lower()
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            child_name = info[12].decode("utf-8", errors="replace").lower()
            joint_name = info[1].decode("utf-8", errors="replace").lower()
            if lookup == child_name or lookup == joint_name:
                return i
        raise SystemExit(f"--end-link={hint!r} did not match any link/joint name")

    return num_joints - 1


def is_key_down(events: Dict[int, int], char: str) -> bool:
    lower = ord(char.lower())
    upper = ord(char.upper())
    mask = p.KEY_IS_DOWN
    return ((events.get(lower, 0) & mask) != 0) or ((events.get(upper, 0) & mask) != 0)


def solve_linear_system(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(a)
    if n == 0 or len(b) != n:
        return None

    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv

        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]

    return [m[i][n] for i in range(n)]


def compute_dls_joint_velocity(
    robot_id: int,
    end_link_idx: int,
    joint_indices: Sequence[int],
    task_twist: Sequence[float],
    damping: float,
    rotation_weight: float,
) -> Optional[List[float]]:
    num_joints = p.getNumJoints(robot_id)
    if num_joints <= 0:
        return None

    q_all = [float(p.getJointState(robot_id, j)[0]) for j in range(num_joints)]
    zeros = [0.0] * num_joints

    jacobian = p.calculateJacobian(
        bodyUniqueId=robot_id,
        linkIndex=end_link_idx,
        localPosition=[0.0, 0.0, 0.0],
        objPositions=q_all,
        objVelocities=zeros,
        objAccelerations=zeros,
    )
    linear_jac = jacobian[0]
    angular_jac = jacobian[1]

    rows: List[List[float]] = []
    row_weights = (1.0, 1.0, 1.0, float(rotation_weight), float(rotation_weight))
    for src_row, row_weight in zip(
        (linear_jac[0], linear_jac[1], linear_jac[2], angular_jac[0], angular_jac[1]),
        row_weights,
    ):
        rows.append([float(src_row[j]) * row_weight for j in joint_indices])

    n = len(joint_indices)
    if n == 0:
        return None

    jt = [[rows[r][c] for r in range(5)] for c in range(n)]
    h = [[0.0 for _ in range(n)] for _ in range(n)]
    g = [0.0 for _ in range(n)]

    for i in range(n):
        for j in range(n):
            acc = 0.0
            for k in range(5):
                acc += jt[i][k] * rows[k][j]
            h[i][j] = acc
        h[i][i] += damping * damping

        rhs = 0.0
        for k in range(5):
            rhs += jt[i][k] * float(task_twist[k]) * row_weights[k]
        g[i] = rhs

    return solve_linear_system(h, g)


class TeleopIKGenerator:
    """Generates joint trajectories from end-effector keyboard teleop in PyBullet."""

    def __init__(
        self,
        urdf_path: Path,
        fixed_base: bool,
        extra_search_paths: Iterable[Path],
        end_link_hint: str,
        linear_speed: float,
        angular_speed_deg: float,
        joint_vel_limit: float,
        sim_hz: float,
        speed_factor: float,
        rotation_weight: float,
        joint_force: float,
        ik_damping: float,
    ) -> None:
        self.urdf_path = urdf_path
        self.fixed_base = fixed_base
        self.extra_search_paths = list(extra_search_paths)
        self.end_link_hint = end_link_hint

        self.linear_speed = max(0.0, float(linear_speed))
        self.angular_speed = math.radians(max(0.0, float(angular_speed_deg)))
        self.joint_vel_limit = max(0.0, float(joint_vel_limit))
        self.sim_hz = max(1.0, float(sim_hz))
        self.speed_factor = max(0.0, float(speed_factor))
        self.rotation_weight = max(1.0, float(rotation_weight))
        self.joint_force = max(1.0, float(joint_force))
        self.damping = max(1e-6, float(ik_damping))

        self.dt = 1.0 / self.sim_hz
        self.step_sleep = self.dt / self.speed_factor if self.speed_factor > 0.0 else 0.0

        self.robot_id = -1
        self.end_link_idx = -1
        self.joint_indices: List[int] = []
        self.joint_names: List[str] = []
        self._wall_start = 0.0

    def setup(self) -> None:
        cid = p.connect(p.GUI)
        if cid < 0:
            raise SystemExit("Failed to open PyBullet GUI")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        for path in self.extra_search_paths:
            if path.exists() and path.is_dir():
                p.setAdditionalSearchPath(str(path))

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt)

        self.robot_id = p.loadURDF(
            str(self.urdf_path),
            useFixedBase=self.fixed_base,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        self.joint_indices, self.joint_names = get_controllable_joints(self.robot_id)
        if len(self.joint_indices) < 6:
            raise SystemExit("Expected at least 6 controllable joints for this pipeline")

        self.joint_indices = self.joint_indices[:6]
        self.joint_names = self.joint_names[:6]

        self.end_link_idx = resolve_end_link(self.robot_id, self.end_link_hint)
        if self.end_link_idx < 0:
            raise SystemExit("No valid end link found")

        self._wall_start = time.perf_counter()

    def _read_task_twist(self) -> Tuple[List[float], List[str]]:
        events = p.getKeyboardEvents()
        vx = 0.0
        vy = 0.0
        vz = 0.0
        wx = 0.0
        wy = 0.0
        active_keys: List[str] = []

        if is_key_down(events, "w"):
            vx += self.linear_speed
            active_keys.append("W")
        if is_key_down(events, "s"):
            vx -= self.linear_speed
            active_keys.append("S")
        if is_key_down(events, "a"):
            vy += self.linear_speed
            active_keys.append("A")
        if is_key_down(events, "d"):
            vy -= self.linear_speed
            active_keys.append("D")
        if is_key_down(events, "q"):
            vz += self.linear_speed
            active_keys.append("Q")
        if is_key_down(events, "e"):
            vz -= self.linear_speed
            active_keys.append("E")

        if is_key_down(events, "h"):
            wx -= self.angular_speed
            active_keys.append("H")
        if is_key_down(events, "k"):
            wx += self.angular_speed
            active_keys.append("K")
        if is_key_down(events, "u"):
            wy += self.angular_speed
            active_keys.append("U")
        if is_key_down(events, "j"):
            wy -= self.angular_speed
            active_keys.append("J")

        return [vx, vy, vz, wx, wy], active_keys

    def step(self) -> Optional[TeleopSample]:
        if not p.isConnected():
            return None

        task_twist, active_keys = self._read_task_twist()
        current_joints = [float(p.getJointState(self.robot_id, j)[0]) for j in self.joint_indices]

        if sum(abs(v) for v in task_twist) <= 1e-12:
            qdot = [0.0] * len(self.joint_indices)
        else:
            solved = compute_dls_joint_velocity(
                robot_id=self.robot_id,
                end_link_idx=self.end_link_idx,
                joint_indices=self.joint_indices,
                task_twist=task_twist,
                damping=self.damping,
                rotation_weight=self.rotation_weight,
            )
            if solved is None:
                qdot = [0.0] * len(self.joint_indices)
            else:
                qdot = [float(v) for v in solved]

        if self.joint_vel_limit > 0.0:
            qdot = [max(-self.joint_vel_limit, min(self.joint_vel_limit, v)) for v in qdot]

        for joint_idx, vel in zip(self.joint_indices, qdot):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=self.joint_force,
            )

        p.stepSimulation()
        time.sleep(self.step_sleep)

        next_joints = [float(p.getJointState(self.robot_id, j)[0]) for j in self.joint_indices]
        deg = [math.degrees(v) for v in next_joints]
        vel_deg_s = [math.degrees((next_joints[i] - current_joints[i]) / self.dt) for i in range(len(next_joints))]

        return TeleopSample(
            t=time.perf_counter() - self._wall_start,
            joints_deg=deg,
            joint_vel_deg_s=vel_deg_s,
            active_keys=active_keys,
        )

    def close(self) -> None:
        if p.isConnected():
            p.disconnect()
