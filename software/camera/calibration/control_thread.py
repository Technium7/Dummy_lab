from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Iterable

_SOFTWARE_DIR = Path(__file__).resolve().parents[2]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from manual_control.teleop_runtime import TeleopLaunchConfig, run_teleop_session
from manual_control.teleop_tick import TeleopTickResult
from manual_control.robot_control_thread import ThreadSafeRobotProxy

from .card_config import CalibrationCardConfig
from .fk import base_to_board_from_card, set_robot_joints_deg
from .keyboard_input import DualWindowKeyboard
from .shared_state import CaptureSnapshot, SharedCalibrationBus


class _CalibrationTeleopHook:
    def __init__(
        self,
        bus: SharedCalibrationBus,
        *,
        card_cfg: CalibrationCardConfig,
        robot_id_holder: list[int | None],
        end_link_idx_holder: list[int | None],
    ) -> None:
        self._bus = bus
        self._card_cfg = card_cfg
        self._robot_id_holder = robot_id_holder
        self._end_link_idx_holder = end_link_idx_holder
        self._keys = DualWindowKeyboard()

    @property
    def keys(self) -> DualWindowKeyboard:
        return self._keys

    def before_tick(self, keys: DualWindowKeyboard) -> None:
        keys.drain_cv2_queue(self._bus.key_codes)

    def after_tick(
        self,
        tick: TeleopTickResult,
        *,
        robot: ThreadSafeRobotProxy | None,
        now: float,
        events: dict[int, int],
    ) -> bool:
        self._bus.publish_joint_snapshot(
            transformed_joint_positions_deg=tick.transformed_joint_positions_deg,
            robot_target_deg=tick.robot_target_deg,
            active_keys=tick.active_keys,
            twist=tick.twist,
            timestamp=now,
        )
        self._service_capture(tick, robot=robot, now=now, events=events)
        return not self._bus.stop.is_set()

    def _service_capture(
        self,
        tick: TeleopTickResult,
        *,
        robot: ThreadSafeRobotProxy | None,
        now: float,
        events: dict[int, int],
    ) -> None:
        capture_signaled = self._keys.was_triggered(events, "c")
        try:
            while True:
                self._bus.capture_signals.get_nowait()
                capture_signaled = True
        except Exception:
            pass

        if not capture_signaled:
            return

        robot_id = self._robot_id_holder[0]
        end_link_idx = self._end_link_idx_holder[0]
        if robot_id is None or end_link_idx is None:
            return

        joints_deg = tick.robot_target_deg[:6]
        if robot is not None:
            physical_joints = robot.get_joint_pos()
            if physical_joints is not None:
                joints_deg = physical_joints

        set_robot_joints_deg(robot_id, joints_deg)
        t_base_target = base_to_board_from_card(robot_id, end_link_idx, self._card_cfg)
        self._bus.capture_snapshots.put(
            CaptureSnapshot(
                joints_deg=[float(v) for v in joints_deg[:6]],
                t_base_target=t_base_target,
                transformed_joint_positions_deg=list(tick.transformed_joint_positions_deg),
                timestamp=now,
            )
        )


class CalibrationControlThread(threading.Thread):
    """
    Dedicated teleop thread via manual_control.teleop_runtime (same loop as manual_control_app).
    """

    def __init__(
        self,
        bus: SharedCalibrationBus,
        *,
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
        port: str,
        baudrate: int,
        timeout: float,
        control_dt_ms: float,
        home_use_firmware: bool,
        sim_only: bool,
        debug_serial: bool,
        joint_current_limit_a: float,
        card_cfg: CalibrationCardConfig,
    ) -> None:
        super().__init__(name="calibration-control", daemon=True)
        self._bus = bus
        self._launch = TeleopLaunchConfig(
            urdf_path=urdf_path,
            fixed_base=fixed_base,
            extra_search_paths=extra_search_paths,
            end_link_hint=end_link_hint,
            linear_speed=linear_speed,
            angular_speed_deg=angular_speed_deg,
            joint_vel_limit=joint_vel_limit,
            sim_hz=sim_hz,
            speed_factor=speed_factor,
            rotation_weight=rotation_weight,
            joint_force=joint_force,
            ik_damping=ik_damping,
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            control_dt_ms=control_dt_ms,
            home_use_firmware=home_use_firmware,
            sim_only=sim_only,
            debug_serial=debug_serial,
            joint_current_limit_a=joint_current_limit_a,
        )
        self._card_cfg = card_cfg
        self._robot_id_holder: list[int | None] = [None]
        self._end_link_idx_holder: list[int | None] = [None]
        self._error: BaseException | None = None

    @property
    def error(self) -> BaseException | None:
        return self._error

    def run(self) -> None:
        hook = _CalibrationTeleopHook(
            self._bus,
            card_cfg=self._card_cfg,
            robot_id_holder=self._robot_id_holder,
            end_link_idx_holder=self._end_link_idx_holder,
        )
        try:
            print(
                "[Control] Teleop thread running (manual_control teleop_runtime). "
                "WASD/QE/HK/UJ/IY in PyBullet or camera window."
            )

            def _scene_ready(robot_id: int, end_link_idx: int) -> None:
                self._robot_id_holder[0] = robot_id
                self._end_link_idx_holder[0] = end_link_idx

            run_teleop_session(
                self._launch,
                keys=hook.keys,
                hook=hook,
                stop_check=lambda: self._bus.stop.is_set(),
                on_scene_ready=_scene_ready,
            )
        except BaseException as exc:
            self._error = exc
            raise
        finally:
            self._bus.stop.set()
