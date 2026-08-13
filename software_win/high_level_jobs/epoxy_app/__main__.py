"""CLI for epoxy phases: pre_apply / xy_optimize / z_creep."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[2]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.approach_config import (  # noqa: E402
    default_approach_config_path,
    load_epoxy_approach_config,
)
from high_level_jobs.epoxy_app.models import ALL_PHASES  # noqa: E402
from high_level_jobs.epoxy_app.pipeline import (  # noqa: E402
    normalize_phases,
    run_epoxy_phases,
)
from robot_config import default_robot_port  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Epoxy apply phases (run separately for debug):\n"
            "  pre_apply   — WORK→rough TCP (safety-offset pose; one motion)\n"
            "  xy_optimize — camera XY refine at fixed Z (placeholder)\n"
            "  z_creep     — vision Z creeps (placeholder; use epoxy_apply)\n"
            "Uses grab_sensor planner + env mesh for pre_apply."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=default_approach_config_path())
    p.add_argument(
        "--phase",
        action="append",
        dest="phases",
        metavar="NAME",
        help=(
            "Phase to run (repeatable): pre_apply | xy_optimize | z_creep. "
            "Default: all three in order."
        ),
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Stream / run selected phase(s) on robot (placeholders are no-ops)",
    )
    p.add_argument("--port", default=default_robot_port())
    p.add_argument("--baudrate", type=int, default=int(os.environ.get("ROBOT_BAUDRATE", "115200")))
    p.add_argument(
        "--plan-gui",
        "--gui",
        action="store_true",
        dest="plan_gui",
        help="PyBullet playback (pre_apply only)",
    )
    p.add_argument(
        "--ik-gui",
        action="store_true",
        help="PyBullet IK goal debug (pre_apply only); skips full plan",
    )
    p.add_argument("--joint-current-limit-a", type=float, default=None)
    args = p.parse_args()

    try:
        phases = normalize_phases(args.phases)
    except ValueError as exc:
        print(f"[epoxy_app] error: {exc}", file=sys.stderr)
        return 1

    cfg = load_epoxy_approach_config(args.config)
    tcp_in = cfg.effective_tcp_table_xyz_in()
    print(
        f"[epoxy_app] rough TCP table in={list(tcp_in)}  "
        f"(nominal={list(cfg.nominal_tcp_table_xyz_in)}, "
        f"safety y-{cfg.safety_offset_y_in} z+{cfg.safety_offset_z_in})  "
        f"load_environment={cfg.load_environment}"
    )
    print(f"[epoxy_app] selected phases: {list(phases)}  (all={list(ALL_PHASES)})")

    params: dict = {
        "execute": bool(args.execute),
        "port": args.port,
        "baudrate": args.baudrate,
        "plan_gui": bool(args.plan_gui),
        "ik_gui": bool(args.ik_gui),
        "phases": list(phases),
    }
    if args.joint_current_limit_a is not None:
        params["joint_current_limit_a"] = float(args.joint_current_limit_a)
    elif args.execute:
        params.setdefault("joint_current_limit_a", cfg.execute.joint_current_limit_a)

    try:
        run_epoxy_phases(cfg, params, phases=phases)
    except ValueError as exc:
        print(f"[epoxy_app] error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[epoxy_app] Interrupted.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
