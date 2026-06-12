from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable

_SOFTWARE_DIR = Path(__file__).resolve().parents[2]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from .card_config import CalibrationCardConfig
from .control_thread import CalibrationControlThread
from .cv_thread import CalibrationCvThread
from .hand_eye import CameraMountConstraint
from .shared_state import SharedCalibrationBus


def launch_calibration_session(
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
    teleop_only: bool,
    camera_mount: CameraMountConstraint,
    *,
    card_cfg: CalibrationCardConfig,
    out_dir: Path,
    camera_serial: str,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
) -> None:
    bus = SharedCalibrationBus()

    control = CalibrationControlThread(
        bus,
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
        card_cfg=card_cfg,
    )

    if teleop_only:
        print("[Calibration] Teleop-only mode (manual_control stack, no RealSense).")
        control.start()
        try:
            while control.is_alive() and not bus.stop.is_set():
                if control.error is not None:
                    raise control.error
                time.sleep(0.05)
        except KeyboardInterrupt:
            bus.stop.set()
        finally:
            bus.stop.set()
            control.join(timeout=2.0)
            if control.error is not None:
                raise control.error
        return

    cv = CalibrationCvThread(
        bus,
        card_cfg=card_cfg,
        out_dir=out_dir,
        camera_serial=camera_serial,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        camera_mount=camera_mount,
    )

    print(
        "[Calibration] Starting control thread (manual_control teleop) "
        "and CV thread (camera + checkerboard)."
    )
    control.start()
    cv.start()

    try:
        while control.is_alive() and not bus.stop.is_set():
            if control.error is not None:
                raise control.error
            if cv.error is not None:
                raise cv.error
            time.sleep(0.05)
    except KeyboardInterrupt:
        bus.stop.set()
    finally:
        bus.stop.set()
        control.join(timeout=2.0)
        cv.join(timeout=2.0)
        if control.error is not None:
            raise control.error
        if cv.error is not None:
            raise cv.error
