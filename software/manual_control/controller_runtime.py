from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from teleop_runtime import TeleopLaunchConfig, run_teleop_session
from teleop_tick import PyBulletKeyInput

_SOFTWARE_DIR = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))


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
        )
    )
