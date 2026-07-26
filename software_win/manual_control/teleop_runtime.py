from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import pybullet as p
import pybullet_data

from manual_control.ik_core import (
    abc_relative_to_semantic_home,
    end_link_name,
    get_controllable_joints,
    get_end_link_pose_robot_frame,
    measure_home_abc_reference,
    resolve_end_link,
)
from manual_control.mapping import HOME_JOINTS_DEG, sim_rad_to_transformed_deg, transformed_deg_to_sim_rad
from manual_control.robot_control_thread import RobotControlThread, ThreadSafeRobotProxy
from manual_control.end_link_window import EndLinkPoseWindow
from manual_control.sim_setup import configure_pybullet_gui, load_ground_plane, load_robot_urdf, refresh_pybullet_display
from manual_control.status_window import ControlStatusWindow
from manual_control.teleop_tick import KeyInput, PyBulletKeyInput, TeleopConfig, TeleopState, TeleopTickResult, run_teleop_tick

_SOFTWARE_DIR = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))


class TeleopLoopHook(Protocol):
    def before_tick(self, keys: KeyInput) -> None: ...

    def after_tick(
        self,
        tick: TeleopTickResult,
        *,
        robot: ThreadSafeRobotProxy | None,
        now: float,
        events: dict[int, int],
    ) -> bool:
        """Return False to exit the teleop loop early."""
        ...


@dataclass
class TeleopLaunchConfig:
    urdf_path: Path
    fixed_base: bool
    extra_search_paths: Iterable[Path]
    end_link_hint: str
    linear_speed: float
    angular_speed_deg: float
    joint_vel_limit: float
    sim_hz: float
    speed_factor: float
    rotation_weight: float
    joint_force: float
    ik_damping: float
    port: str
    baudrate: int
    timeout: float
    control_dt_ms: float
    home_use_firmware: bool = False
    sim_only: bool = False
    debug_serial: bool = False
    joint_current_limit_a: float = 0.0


def run_teleop_session(
    config: TeleopLaunchConfig,
    *,
    keys: KeyInput | None = None,
    hook: TeleopLoopHook | None = None,
    stop_check: Callable[[], bool] | None = None,
    on_scene_ready: Callable[[int, int], None] | None = None,
    on_shutdown: Callable[[], None] | None = None,
    use_status_gui: bool = True,
) -> None:
    """Shared PyBullet + robot teleop loop used by manual_control and calibration."""
    cid = p.connect(p.GUI)
    if cid < 0:
        raise SystemExit("Failed to open PyBullet GUI")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    for path in config.extra_search_paths:
        if path.exists() and path.is_dir():
            p.setAdditionalSearchPath(str(path))

    p.resetSimulation()
    p.setGravity(0, 0, 0)
    configure_pybullet_gui(low_latency=True)
    load_ground_plane()

    robot_id = load_robot_urdf(config.urdf_path, fixed_base=config.fixed_base)
    aabb_min, aabb_max = p.getAABB(robot_id, -1)
    cx = (aabb_min[0] + aabb_max[0]) * 0.5
    cy = (aabb_min[1] + aabb_max[1]) * 0.5
    cz = (aabb_min[2] + aabb_max[2]) * 0.5
    radius = max(
        0.2,
        abs(aabb_max[0] - aabb_min[0]),
        abs(aabb_max[1] - aabb_min[1]),
        abs(aabb_max[2] - aabb_max[2]),
    ) * 2.5
    p.resetDebugVisualizerCamera(
        cameraDistance=radius,
        cameraYaw=50,
        cameraPitch=-25,
        cameraTargetPosition=[cx, cy, cz],
    )
    refresh_pybullet_display()
    print(f"PyBullet robot loaded (id={robot_id}). Close the GUI window to exit.")

    joint_indices, _, _, _, joint_names = get_controllable_joints(robot_id)
    if not joint_indices:
        raise SystemExit("No controllable revolute/prismatic joints found for IK.")
    end_link_idx = resolve_end_link(robot_id, config.end_link_hint)
    end_link_label = end_link_name(robot_id, end_link_idx)
    all_joint_indices = list(range(p.getNumJoints(robot_id)))
    if on_scene_ready is not None:
        on_scene_ready(robot_id, end_link_idx)

    status_gui = ControlStatusWindow() if use_status_gui else None
    if status_gui is not None:
        status_gui.start()
    end_link_gui = EndLinkPoseWindow() if use_status_gui else None
    if end_link_gui is not None:
        end_link_gui.start()

    control_dt_s = max(0.01, float(config.control_dt_ms) / 1000.0)
    robot_control = RobotControlThread(
        port=config.port,
        baudrate=config.baudrate,
        timeout=config.timeout,
        control_dt_s=control_dt_s,
        home_use_firmware=config.home_use_firmware,
        sim_only=config.sim_only,
        debug_serial=config.debug_serial,
        joint_current_limit_a=config.joint_current_limit_a,
    )
    robot_control.start()
    robot_control.wait_until_ready()
    if not config.sim_only:
        hw_joints = robot_control.get_joint_pos_blocking(timeout=1.0)
        if hw_joints is not None and len(hw_joints) >= 6:
            print(f"[Robot] Seeding sim from #GETJPOS: {[round(v, 2) for v in hw_joints[:6]]}")
            for local_idx, joint_idx in enumerate(joint_indices[:6]):
                p.resetJointState(
                    robot_id,
                    joint_idx,
                    targetValue=transformed_deg_to_sim_rad(local_idx, hw_joints[local_idx]),
                    targetVelocity=0.0,
                )
            teleop_state = TeleopState(hold_target_deg=[float(v) for v in hw_joints[:6]])
        else:
            print("[Robot] Warning: could not read #GETJPOS; sim starts at URDF default.", file=sys.stderr)
            teleop_state = TeleopState()
    else:
        teleop_state = TeleopState()
    if config.debug_serial:
        print("[Serial debug] logging all TX/RX (stream waypoints included)")

    sim_dt = 1.0 / max(1.0, float(config.sim_hz))
    step_sleep = sim_dt / max(1e-9, float(config.speed_factor)) if float(config.speed_factor) > 0.0 else 0.0
    p.setPhysicsEngineParameter(fixedTimeStep=sim_dt)

    status_update_stride = max(1, int(round(float(config.sim_hz) / 25.0)))
    sim_frame = 0
    include_joint_telemetry = hook is not None and getattr(hook, "needs_joint_telemetry", True)
    teleop_config = TeleopConfig(
        linear_speed=max(0.0, float(config.linear_speed)),
        angular_speed_deg=max(0.0, float(config.angular_speed_deg)),
        joint_vel_limit=max(0.0, float(config.joint_vel_limit)),
        sim_dt=sim_dt,
        rotation_weight=float(config.rotation_weight),
        joint_force=max(1.0, float(config.joint_force)),
        ik_damping=max(1e-6, float(config.ik_damping)),
        control_dt_s=control_dt_s,
        home_use_firmware=config.home_use_firmware,
    )
    key_input = keys if keys is not None else PyBulletKeyInput()
    robot_for_hook = robot_control.proxy if hook is not None else None
    home_abc_ref = measure_home_abc_reference(
        robot_id, end_link_idx, joint_indices, HOME_JOINTS_DEG
    )

    try:
        while p.isConnected():
            if stop_check is not None and stop_check():
                break
            if hook is not None:
                hook.before_tick(key_input)
            events = p.getKeyboardEvents()
            tick = run_teleop_tick(
                events=events,
                keys=key_input,
                state=teleop_state,
                config=teleop_config,
                robot_id=robot_id,
                end_link_idx=end_link_idx,
                joint_indices=joint_indices,
                all_joint_indices=all_joint_indices,
                robot_control=robot_control,
                include_joint_telemetry=include_joint_telemetry,
            )
            sim_frame += 1
            if status_gui is not None and sim_frame % status_update_stride == 0:
                if not include_joint_telemetry:
                    all_joint_positions = [float(p.getJointState(robot_id, j)[0]) for j in all_joint_indices]
                    joint_display = [
                        sim_rad_to_transformed_deg(idx, pos) for idx, pos in enumerate(all_joint_positions)
                    ]
                else:
                    joint_display = tick.transformed_joint_positions_deg
                status_gui.update(
                    tick.active_keys,
                    tick.twist,
                    joint_names,
                    joint_display,
                )
                if end_link_gui is not None:
                    pos, abc = get_end_link_pose_robot_frame(robot_id, end_link_idx)
                    abc = abc_relative_to_semantic_home(abc, home_abc_ref)
                    end_link_gui.update(
                        link_name=end_link_label,
                        link_index=end_link_idx,
                        position_m=pos,
                        abc_deg=abc,
                    )
            if hook is not None:
                if not hook.after_tick(tick, robot=robot_for_hook, now=time.perf_counter(), events=events):
                    break
            if step_sleep > 0.0:
                time.sleep(step_sleep)
    finally:
        if on_shutdown is not None:
            try:
                on_shutdown()
            except Exception:
                pass
        robot_control.request_stop()
        robot_control.join(timeout=1.0)
        if status_gui is not None:
            status_gui.close()
        if end_link_gui is not None:
            end_link_gui.close()
