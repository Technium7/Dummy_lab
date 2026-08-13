from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable

_SOFTWARE_DIR = Path(__file__).resolve().parents[1]
_MANUAL_CONTROL_DIR = Path(__file__).resolve().parent
for _path in (_SOFTWARE_DIR, _MANUAL_CONTROL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from teleop_runtime import TeleopLaunchConfig, TeleopLoopHook, run_teleop_session
    from teleop_tick import PyBulletKeyInput
except ImportError:  # imported as package (e.g. session → manual_control.controller_runtime)
    from manual_control.teleop_runtime import TeleopLaunchConfig, TeleopLoopHook, run_teleop_session
    from manual_control.teleop_tick import PyBulletKeyInput


def is_key_triggered(events: dict[int, int], char: str) -> bool:
    return PyBulletKeyInput().was_triggered(events, char)


def launch_controller(
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
    home_use_firmware: bool = False,
    sim_only: bool = False,
    debug_serial: bool = False,
    joint_current_limit_a: float = 0.0,
    creep_enabled: bool = True,
    creep_step_m: float = 0.00012,
    creep_linear_speed: float = 0.00165,
    creep_accel_time: float = 0.20,
    creep_settle_tol: float = 1e-5,
    creep_hold_threshold_s: float = 0.20,
    *,
    teleop_hook: TeleopLoopHook | None = None,
    on_shutdown: Callable[[], None] | None = None,
) -> None:
    run_teleop_session(
        TeleopLaunchConfig(
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
            creep_enabled=creep_enabled,
            creep_step_m=creep_step_m,
            creep_linear_speed=creep_linear_speed,
            creep_accel_time=creep_accel_time,
            creep_settle_tol=creep_settle_tol,
            creep_hold_threshold_s=creep_hold_threshold_s,
        ),
        hook=teleop_hook,
        on_shutdown=on_shutdown,
    )
