"""Launch Windows manual_control teleop from the session (port already free)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from robot_config import DEFAULT_ROBOT, repo_root


def run_manual_control(
    *,
    port: str,
    baudrate: int = 115200,
    joint_current_limit_a: float | None = None,
    sim_only: bool = False,
    timeout: float = 1.0,
) -> None:
    """Block in PyBullet teleop until the GUI is closed."""
    # Same sys.path setup as manual_control_app (bare teleop_* imports).
    manual_dir = Path(__file__).resolve().parents[2] / "manual_control"
    software_win = manual_dir.parent
    for path in (software_win, manual_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from controller_runtime import launch_controller
    from urdf_tools import (
        build_package_map,
        choose_description,
        discover_descriptions,
        materialize_urdf,
    )

    root = repo_root()
    candidates = discover_descriptions(root)
    selected = choose_description(candidates, DEFAULT_ROBOT, root)
    package_map = build_package_map(root)
    urdf_path, temp_file = materialize_urdf(selected, package_map)
    print(f"[session/manual] robot={selected}")
    print(f"[session/manual] URDF={urdf_path}")
    if selected.suffix.lower() == ".xacro":
        urdf_text = urdf_path.read_text(encoding="utf-8")
        if len(re.findall(r"<joint\b", urdf_text)) == 0:
            print("Warning: resolved URDF has 0 joints. IK control will not work.", file=sys.stderr)

    search_paths = {root, selected.parent, urdf_path.parent, *package_map.values()}
    limit = float(joint_current_limit_a or 0.0)
    try:
        launch_controller(
            urdf_path=urdf_path,
            fixed_base=True,
            extra_search_paths=search_paths,
            end_link_hint="",
            linear_speed=0.015,
            angular_speed_deg=10.0,
            joint_vel_limit=0.0,
            sim_hz=480.0,
            speed_factor=1.0,
            rotation_weight=8.0,
            joint_force=2000.0,
            ik_damping=0.05,
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            control_dt_ms=70.0,
            home_use_firmware=False,
            sim_only=sim_only,
            debug_serial=False,
            joint_current_limit_a=limit,
        )
    finally:
        if temp_file is not None and Path(temp_file).exists():
            Path(temp_file).unlink(missing_ok=True)
