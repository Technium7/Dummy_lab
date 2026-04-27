from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

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


def parse_initial_joint_angles(urdf_path: Path) -> Dict[str, float]:
    """Parse pybullet_initial_angle attributes from URDF joints.
    
    Returns a dict mapping joint names to initial angles (radians).
    """
    angles: Dict[str, float] = {}
    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        for joint in root.findall(".//joint"):
            joint_name = joint.get("name", "")
            initial_angle_elem = joint.find("pybullet_initial_angle")
            if initial_angle_elem is not None and "value" in initial_angle_elem.attrib:
                try:
                    angle_rad = float(initial_angle_elem.attrib["value"])
                    angles[joint_name] = angle_rad
                except (ValueError, TypeError):
                    pass
        if angles:
            print(f"[parse_initial_joint_angles] Found {len(angles)} calibrated joint angles in URDF")
        else:
            print(f"[parse_initial_joint_angles] WARNING: No pybullet_initial_angle tags found in URDF")
    except Exception as e:
        print(f"[parse_initial_joint_angles] ERROR: Could not parse URDF: {e}")
    return angles


def parse_joint_origin_offsets(urdf_path: Path) -> Dict[str, float]:
    """Parse joint origin rpy x-values from URDF and return per-joint offset (radians).

    The returned offsets are the fixed rotation applied in the joint origin about the
    first rpy axis. Callers should subtract this offset from robot-native angles
    when applying to the simulator, and add it when converting simulator angles
    back to robot-native values.
    """
    offsets: Dict[str, float] = {}
    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        for joint in root.findall('.//joint'):
            name = joint.get('name', '')
            origin = joint.find('origin')
            axis = joint.find('axis')
            if origin is None:
                continue
            rpy = origin.get('rpy', '')
            if not rpy:
                continue
            try:
                rx, ry, rz = (float(v) for v in rpy.split())
            except Exception:
                continue
            # Only record offsets for joints whose axis aligns with the X component
            # convention used in this robot description (common for these joints).
            # If axis is present, confirm it has a dominant x component.
            axis_xyz = (1.0, 0.0, 0.0)
            if axis is not None and 'xyz' in axis.attrib:
                try:
                    ax, ay, az = (float(v) for v in axis.get('xyz').split())
                    axis_xyz = (ax, ay, az)
                except Exception:
                    axis_xyz = (1.0, 0.0, 0.0)

            if abs(axis_xyz[0]) >= max(abs(axis_xyz[1]), abs(axis_xyz[2])):
                offsets[name] = rx
    except Exception as e:
        print(f"[parse_joint_origin_offsets] ERROR: Could not parse URDF: {e}")
    return offsets


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
        enable_twist_sliders: bool = False,
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
        self.enable_twist_sliders = bool(enable_twist_sliders)

        self.dt = 1.0 / self.sim_hz
        self.step_sleep = self.dt / self.speed_factor if self.speed_factor > 0.0 else 0.0

        self.robot_id = -1
        self.end_link_idx = -1
        self.joint_indices: List[int] = []
        self.joint_names: List[str] = []
        self._wall_start = 0.0
        self._slider_ids: Dict[str, int] = {}
        self.joint_index_to_offset: Dict[int, float] = {}

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

        # Parse fixed origin offsets from URDF so we can map simulator <-> robot joint frames.
        origin_offsets_by_name = parse_joint_origin_offsets(self.urdf_path)

        # Load and apply initial calibration angles from URDF (robot-native angles).
        initial_angles = parse_initial_joint_angles(self.urdf_path)
        if initial_angles:
            applied_count = 0
            for i in range(p.getNumJoints(self.robot_id)):
                info = p.getJointInfo(self.robot_id, i)
                joint_name = info[1].decode("utf-8", errors="replace")
                if joint_name in initial_angles:
                    angle_rad = initial_angles[joint_name]
                    # Convert robot-native angle -> simulator joint angle by subtracting origin offset
                    offset = origin_offsets_by_name.get(joint_name, 0.0)
                    sim_angle = float(angle_rad) - float(offset)
                    p.resetJointState(
                        self.robot_id,
                        i,
                        targetValue=sim_angle,
                        targetVelocity=0.0,
                    )
                    applied_count += 1
            print(f"[TeleopIKGenerator] Applied {applied_count} calibrated initial joint angles to simulator.")
        else:
            print("[TeleopIKGenerator] WARNING: No calibrated angles found to apply.")

        self.joint_indices, self.joint_names = get_controllable_joints(self.robot_id)
        if len(self.joint_indices) < 6:
            raise SystemExit("Expected at least 6 controllable joints for this pipeline")

        self.joint_indices = self.joint_indices[:6]
        self.joint_names = self.joint_names[:6]

        # Build per-joint index -> origin offset mapping for the controllable joints
        self.joint_index_to_offset = {}
        origin_offsets_by_name = parse_joint_origin_offsets(self.urdf_path)
        for idx, name in zip(self.joint_indices, self.joint_names):
            self.joint_index_to_offset[idx] = float(origin_offsets_by_name.get(name, 0.0))

        self.end_link_idx = resolve_end_link(self.robot_id, self.end_link_hint)
        if self.end_link_idx < 0:
            raise SystemExit("No valid end link found")

        if self.enable_twist_sliders:
            self._slider_ids = {
                "vx": p.addUserDebugParameter("vx (m/s)", -self.linear_speed, self.linear_speed, 0.0),
                "vy": p.addUserDebugParameter("vy (m/s)", -self.linear_speed, self.linear_speed, 0.0),
                "vz": p.addUserDebugParameter("vz (m/s)", -self.linear_speed, self.linear_speed, 0.0),
                "wx": p.addUserDebugParameter("wx (rad/s)", -self.angular_speed, self.angular_speed, 0.0),
                "wy": p.addUserDebugParameter("wy (rad/s)", -self.angular_speed, self.angular_speed, 0.0),
            }

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

        if self._slider_ids:
            vx += float(p.readUserDebugParameter(self._slider_ids["vx"]))
            vy += float(p.readUserDebugParameter(self._slider_ids["vy"]))
            vz += float(p.readUserDebugParameter(self._slider_ids["vz"]))
            wx += float(p.readUserDebugParameter(self._slider_ids["wx"]))
            wy += float(p.readUserDebugParameter(self._slider_ids["wy"]))
            if abs(vx) + abs(vy) + abs(vz) + abs(wx) + abs(wy) > 1e-6 and "SLIDER" not in active_keys:
                active_keys.append("SLIDER")

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
        # Convert simulator joint angles -> robot-native angles by adding origin offsets
        robot_joints = [next_joints[i] + self.joint_index_to_offset.get(self.joint_indices[i], 0.0) for i in range(len(next_joints))]
        robot_prev = [current_joints[i] + self.joint_index_to_offset.get(self.joint_indices[i], 0.0) for i in range(len(current_joints))]
        deg = [math.degrees(v) for v in robot_joints]
        vel_deg_s = [math.degrees((robot_joints[i] - robot_prev[i]) / self.dt) for i in range(len(robot_joints))]

        return TeleopSample(
            t=time.perf_counter() - self._wall_start,
            joints_deg=deg,
            joint_vel_deg_s=vel_deg_s,
            active_keys=active_keys,
        )

    def close(self) -> None:
        if p.isConnected():
            p.disconnect()

    def set_joint_positions_deg(self, joints_deg: Sequence[float]) -> None:
        if not p.isConnected() or self.robot_id < 0:
            return
        if len(joints_deg) < len(self.joint_indices):
            return

        for joint_idx, joint_deg in zip(self.joint_indices, joints_deg):
            # Convert robot-native angle -> simulator joint angle by subtracting origin offset
            offset = self.joint_index_to_offset.get(joint_idx, 0.0)
            sim_angle = math.radians(float(joint_deg)) - float(offset)
            p.resetJointState(self.robot_id, joint_idx, targetValue=sim_angle, targetVelocity=0.0)
