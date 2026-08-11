"""CLI for Windows power_measure click-to-insert."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[2]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.power_measure.config_loader import (  # noqa: E402
    apply_runtime_calibrations,
    default_config_path,
    load_power_measure_config,
)
from high_level_jobs.power_measure.live_gui import run_live_gui  # noqa: E402
from high_level_jobs.power_measure.pipeline import run_insert_pipeline  # noqa: E402
from high_level_jobs.power_measure.target import build_insert_target_from_uv  # noqa: E402
from robot_config import default_robot_port  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Windows power_measure: live top-camera click to "
            "radial-left TCP insert from above (sensor already held)."
        )
    )
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--execute", action="store_true", help="Stream plan to robot after confirm")
    p.add_argument("--port", default=default_robot_port())
    p.add_argument("--baudrate", type=int, default=int(os.environ.get("ROBOT_BAUDRATE", "115200")))
    p.add_argument("--camera-serial", type=str, default=None)
    p.add_argument(
        "--plan-gui",
        "--gui",
        action="store_true",
        dest="plan_gui",
        help="PyBullet playback of planned trajectory (like grab_sensor --plan-gui)",
    )
    p.add_argument(
        "--ik-gui",
        action="store_true",
        help="PyBullet IK goal debug (like grab_sensor --ik-gui); skips full plan",
    )
    p.add_argument("--joint-current-limit-a", type=float, default=None)
    p.add_argument(
        "--click-uv",
        type=str,
        default=None,
        help="Skip live GUI; plan from U,V pixel (e.g. 320,240)",
    )
    p.add_argument("--no-gui", action="store_true", help="Require --click-uv; no live window")
    p.add_argument(
        "--no-power-scan",
        action="store_true",
        help="Skip PM100 record/search/maximize (insert + lift + WORK only)",
    )
    p.add_argument(
        "--pm-serial",
        type=str,
        default=None,
        help="Thorlabs PM100 serial substring (overrides config power_meter.serial)",
    )
    p.add_argument(
        "--pm-resource",
        type=str,
        default=None,
        help="Full VISA resource string for the power meter",
    )
    args = p.parse_args()

    cfg = apply_runtime_calibrations(load_power_measure_config(args.config), {})
    if args.pm_serial is not None or args.pm_resource is not None:
        from dataclasses import replace

        meter = cfg.power_meter
        cfg = replace(
            cfg,
            power_meter=replace(
                meter,
                serial=str(args.pm_serial).strip() if args.pm_serial is not None else meter.serial,
                visa_resource=(
                    str(args.pm_resource).strip()
                    if args.pm_resource is not None
                    else meter.visa_resource
                ),
            ),
        )
    if cfg.table_calib_path is None:
        print(
            "[power_measure] error: no table calib found under "
            "software_win/camera/calibration_table/runs/ "
            "(or software/camera/calibration_table/runs/)",
            file=sys.stderr,
        )
        return 1

    params: dict = {
        "execute": bool(args.execute),
        "port": args.port,
        "baudrate": args.baudrate,
        "plan_gui": bool(args.plan_gui),
        "ik_gui": bool(args.ik_gui),
        "no_power_scan": bool(args.no_power_scan),
    }
    if args.joint_current_limit_a is not None:
        params["joint_current_limit_a"] = float(args.joint_current_limit_a)
    elif args.execute:
        params.setdefault("joint_current_limit_a", cfg.execute.joint_current_limit_a)
    if args.camera_serial is not None:
        params["camera_serial"] = str(args.camera_serial).strip()

    if args.click_uv:
        parts = [s.strip() for s in str(args.click_uv).replace(" ", "").split(",")]
        if len(parts) != 2:
            print("--click-uv must be U,V", file=sys.stderr)
            return 1
        uv = (int(parts[0]), int(parts[1]))
        insert = build_insert_target_from_uv(cfg, uv)
        print(
            f"[power_measure] --click-uv {uv} -> "
            f"TCP table in={insert.debug.get('tcp_table_xyz_in')}"
        )
        try:
            run_insert_pipeline(cfg, insert, params)
        except KeyboardInterrupt:
            print("\n[power_measure] Interrupted.")
            raise
        return 0

    if args.no_gui:
        print("--no-gui requires --click-uv", file=sys.stderr)
        return 1

    # Live click first; close RealSense/OpenCV before PyBullet (same idea as grab_sensor).
    live = run_live_gui(cfg, params)
    if not live.confirmed or live.insert is None:
        print("[power_measure] quit without confirm.")
        return 0

    try:
        run_insert_pipeline(cfg, live.insert, params)
    except KeyboardInterrupt:
        print("\n[power_measure] Interrupted.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
