#!/usr/bin/env python3
"""
PyBullet preview: table frame, robot, camera, sensor base, and grasp target.

World frame = optical table frame (origin at calibration hole, Z up from surface).
Robot base is placed at ``robot_in_table_frame`` (default -6, -5, 0.5 inches).

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.view_grasp_table
  python -m high_level_jobs.subtasks.grab_sensor.view_grasp_table --offline
  python -m high_level_jobs.subtasks.grab_sensor.view_grasp_table --robot-z-in 0.5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_PKG_DIR = Path(__file__).resolve().parent
_SOFTWARE_DIR = _PKG_DIR.parents[2]
_TABLE_CALIB_DIR = _SOFTWARE_DIR / "camera" / "calibration_table"
for _path in (_SOFTWARE_DIR, _SOFTWARE_DIR / "manual_control", str(_TABLE_CALIB_DIR.parent)):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pybullet as p
import pybullet_data

from camera.calibration_table.view_table_calibration import (  # noqa: E402
    _draw_camera_gizmo,
    _draw_robot_base_marker,
    _draw_table_grid,
    _latest_calib_json,
    _load_table_cam_transform,
    _resolve_calib_path,
    _robot_base_in_table,
)
from manual_control.pose_conventions import world_pos_to_robot_frame
from manual_control.sim_setup import configure_pybullet_gui, refresh_pybullet_display
from manual_control.urdf_tools import build_package_map, choose_description, discover_descriptions, materialize_urdf
from robot_config import DEFAULT_ROBOT, repo_root

from .config_loader import load_grab_sensor_config
from .models import GrabSensorConfig, RobotInTableFrame
from .table_position import (
    resolve_table_calib_path,
    table_point_to_planner_world_m,
    table_point_to_robot_m,
    table_robot_matrix,
)

_COLOR_BASE = [0.1, 0.95, 0.2]
_COLOR_GRASP = [1.0, 0.15, 0.15]
_COLOR_GRASP_OLD = [1.0, 0.4, 1.0]
_COLOR_NN = [1.0, 0.55, 0.1]


def _draw_point_marker(
    pos: tuple[float, float, float] | list[float],
    color: list[float],
    *,
    size_m: float = 0.018,
    axis_m: float = 0.04,
) -> None:
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    s = float(size_m)
    p.addUserDebugLine([x - s, y, z], [x + s, y, z], color, 3.0, 0.0)
    p.addUserDebugLine([x, y - s, z], [x, y + s, z], color, 3.0, 0.0)
    p.addUserDebugLine([x, y, z], [x, y, z + axis_m], color, 3.0, 0.0)


def _load_detection_debug(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("debug", data)


def _planner_world_to_table_m(
    planner_xyz: np.ndarray,
    robot_in_table: RobotInTableFrame,
) -> np.ndarray:
    """Map planner-world point (robot at origin) back to optical table frame."""
    p_robot = np.array(world_pos_to_robot_frame(planner_xyz), dtype=np.float64)
    p_h = np.ones(4, dtype=np.float64)
    p_h[:3] = p_robot
    return (table_robot_matrix(robot_in_table) @ p_h)[:3]


def _grasp_targets_from_debug(cfg: GrabSensorConfig, debug: dict) -> dict[str, np.ndarray]:
    base_xyz = debug.get("base_table_xyz_m")
    if not base_xyz:
        raise ValueError("detection debug missing base_table_xyz_m — run vision first")

    height_m = float(cfg.grasp_motion.grasp_height_above_table_mm) * 1e-3
    grasp_table = np.array(
        [float(base_xyz[0]), float(base_xyz[1]), height_m],
        dtype=np.float64,
    )
    base_table = np.array([float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])], dtype=np.float64)

    grasp_robot = table_point_to_planner_world_m(tuple(grasp_table), cfg.robot_in_table_frame)

    out: dict[str, np.ndarray] = {
        "base_table": base_table,
        "grasp_table": grasp_table,
        "grasp_robot": grasp_robot,
    }

    table_calib = resolve_table_calib_path(cfg.table_calib_path)
    if table_calib is not None:
        grasp_handeye = table_point_to_robot_m(
            tuple(grasp_table),
            table_calib_path=table_calib,
            camera_to_robot=cfg.camera_to_robot,
        )
        out["grasp_handeye_world"] = grasp_handeye
        out["grasp_handeye_table"] = _planner_world_to_table_m(grasp_handeye, cfg.robot_in_table_frame)

    if debug.get("grasp_cam_m"):
        from .grasp import camera_to_robot_matrix

        cam = np.asarray(debug["grasp_cam_m"], dtype=np.float64)
        t_rc = camera_to_robot_matrix(cfg.camera_to_robot)
        p_h = np.ones(4, dtype=np.float64)
        p_h[:3] = cam
        nn_world = (t_rc @ p_h)[:3]
        out["grasp_nn_handeye_world"] = nn_world
        out["grasp_nn_handeye_table"] = _planner_world_to_table_m(nn_world, cfg.robot_in_table_frame)

    return out


def run_preview(
    *,
    cfg: GrabSensorConfig,
    calib_path: Path,
    urdf_path: Path,
    extra_search_paths: set[Path],
    fixed_base: bool,
    targets: dict[str, np.ndarray],
) -> None:
    T_table_cam, data = _load_table_cam_transform(calib_path)
    cam_pos = T_table_cam[:3, 3]
    rtx, rty, rtz = cfg.robot_in_table_frame.translation_m()
    robot_rpy_deg = tuple(float(v) for v in cfg.robot_in_table_frame.rpy_deg)

    cid = p.connect(p.GUI)
    if cid < 0:
        raise SystemExit("Failed to open PyBullet GUI")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        for path in extra_search_paths:
            if path.exists() and path.is_dir():
                p.setAdditionalSearchPath(str(path))

        p.resetSimulation()
        p.setGravity(0, 0, 0)
        configure_pybullet_gui(low_latency=False)

        plane_path = Path(pybullet_data.getDataPath()) / "plane.urdf"
        if plane_path.is_file():
            p.loadURDF(str(plane_path))

        robot_pos, robot_quat = _robot_base_in_table(rtx, rty, rtz, robot_rpy_deg)
        p.loadURDF(
            str(urdf_path),
            basePosition=robot_pos,
            baseOrientation=robot_quat,
            useFixedBase=fixed_base,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        _draw_table_grid(data)
        _draw_camera_gizmo(T_table_cam)
        _draw_robot_base_marker(robot_pos)

        base = targets["base_table"]
        grasp = targets["grasp_table"]
        _draw_point_marker(base, _COLOR_BASE, size_m=0.012, axis_m=0.03)
        _draw_point_marker(grasp, _COLOR_GRASP, size_m=0.015, axis_m=0.065)
        p.addUserDebugLine(
            [float(base[0]), float(base[1]), float(base[2])],
            [float(grasp[0]), float(grasp[1]), float(grasp[2])],
            _COLOR_GRASP,
            2.0,
            0.0,
        )

        if "grasp_handeye_table" in targets:
            he = targets["grasp_handeye_table"]
            _draw_point_marker(he, _COLOR_GRASP_OLD, size_m=0.012, axis_m=0.05)
        if "grasp_nn_handeye_table" in targets:
            nn = targets["grasp_nn_handeye_table"]
            _draw_point_marker(nn, _COLOR_NN, size_m=0.010, axis_m=0.04)

        print(f"[view_grasp_table] Calibration: {calib_path}")
        print(
            f"[view_grasp_table] Robot in table (in): {list(cfg.robot_in_table_frame.translation_in)}  "
            f"rpy deg {list(cfg.robot_in_table_frame.rpy_deg)}"
        )
        print(
            f"[view_grasp_table] Camera in table (m): "
            f"[{cam_pos[0]:+.4f}, {cam_pos[1]:+.4f}, {cam_pos[2]:+.4f}]"
        )
        print(
            f"[view_grasp_table] Base on table (m): "
            f"[{base[0]:+.4f}, {base[1]:+.4f}, {base[2]:+.4f}]"
        )
        print(
            f"[view_grasp_table] Grasp target table (m): "
            f"[{grasp[0]:+.4f}, {grasp[1]:+.4f}, {grasp[2]:+.4f}]  "
            f"({cfg.grasp_motion.grasp_height_above_table_mm:.0f} mm above table)"
        )
        print(
            f"[view_grasp_table] Grasp robot/planner (m): "
            f"[{targets['grasp_robot'][0]:+.4f}, {targets['grasp_robot'][1]:+.4f}, {targets['grasp_robot'][2]:+.4f}]"
        )
        if "grasp_handeye_world" in targets:
            gh = targets["grasp_handeye_world"]
            ht = targets["grasp_handeye_table"]
            print(
                f"[view_grasp_table] OLD hand-eye planner world (m): "
                f"[{gh[0]:+.4f}, {gh[1]:+.4f}, {gh[2]:+.4f}]"
            )
            print(
                f"[view_grasp_table] OLD hand-eye in table frame (m): "
                f"[{ht[0]:+.4f}, {ht[1]:+.4f}, {ht[2]:+.4f}]"
            )
        if "grasp_nn_handeye_table" in targets:
            nt = targets["grasp_nn_handeye_table"]
            print(
                f"[view_grasp_table] NN+cam hand-eye in table frame (m): "
                f"[{nt[0]:+.4f}, {nt[1]:+.4f}, {nt[2]:+.4f}]"
            )
        print(
            "[view_grasp_table] Green = sensor base  |  Red = grasp (base XY + 65mm Z)  |  "
            "Magenta = old hand-eye grasp  |  Orange = NN+camera hand-eye  |  "
            "Yellow/blue = camera  |  Yellow axes = robot base"
        )

        span = max(float(np.linalg.norm(cam_pos)), float(np.linalg.norm(robot_pos)), float(np.linalg.norm(grasp)), 0.5)
        p.resetDebugVisualizerCamera(
            cameraDistance=span * 2.2,
            cameraYaw=125,
            cameraPitch=-35,
            cameraTargetPosition=[
                float(grasp[0]),
                float(grasp[1]),
                max(0.06, float(grasp[2]) * 0.5),
            ],
        )
        refresh_pybullet_display(steps=3)

        while p.isConnected():
            time.sleep(0.05)
    finally:
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_det = _PKG_DIR / "debug" / "detection.json"
    p = argparse.ArgumentParser(description="PyBullet: table frame, robot, camera, grasp target.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--detection", type=Path, default=default_det, help="detection.json from grab_sensor debug")
    p.add_argument("--offline", action="store_true", help="Run live vision capture before preview")
    p.add_argument("--calib", type=Path, default=None)
    p.add_argument("--robot", type=str, default=DEFAULT_ROBOT)
    p.add_argument("--robot-x-in", type=float, default=None)
    p.add_argument("--robot-y-in", type=float, default=None)
    p.add_argument("--robot-z-in", type=float, default=None)
    p.add_argument("--robot-rpy-deg", type=float, nargs=3, default=None, metavar=("R", "P", "Y"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    cfg = load_grab_sensor_config(args.config)

    if args.robot_x_in is not None or args.robot_y_in is not None or args.robot_z_in is not None:
        trans = list(cfg.robot_in_table_frame.translation_in)
        if args.robot_x_in is not None:
            trans[0] = args.robot_x_in
        if args.robot_y_in is not None:
            trans[1] = args.robot_y_in
        if args.robot_z_in is not None:
            trans[2] = args.robot_z_in
        rpy = (
            tuple(args.robot_rpy_deg)
            if args.robot_rpy_deg is not None
            else cfg.robot_in_table_frame.rpy_deg
        )
        cfg = replace(
            cfg,
            robot_in_table_frame=RobotInTableFrame(
                translation_in=tuple(trans),  # type: ignore[arg-type]
                rpy_deg=rpy,  # type: ignore[arg-type]
            ),
        )

    debug: dict
    if args.offline:
        from .pipeline import run_detection

        params = {
            "color_file": str(_PKG_DIR / "debug" / "grab_sensor.color.png"),
            "depth_file": str(_PKG_DIR / "debug" / "grab_sensor.depth.npy"),
            "visualize": False,
        }
        _, targets, _, _ = run_detection(cfg, params)
        debug = targets.debug
    else:
        det_path = args.detection.resolve()
        if not det_path.is_file():
            print(f"Detection file not found: {det_path}. Run grab_sensor --detect-only first or use --offline.")
            return 1
        debug = _load_detection_debug(det_path)

    targets = _grasp_targets_from_debug(cfg, debug)

    if args.calib is not None:
        calib_path = _resolve_calib_path(args.calib, root)
    else:
        calib_path = resolve_table_calib_path(cfg.table_calib_path)
        if calib_path is None:
            calib_path = _latest_calib_json(_TABLE_CALIB_DIR / "runs")
    if calib_path is None or not calib_path.is_file():
        print("No table_calib.json found.")
        return 1

    selected = choose_description(discover_descriptions(root), args.robot, root)
    urdf_path, temp_file = materialize_urdf(selected, build_package_map(root))
    if selected.suffix.lower() == ".xacro":
        urdf_text = urdf_path.read_text(encoding="utf-8")
        if len(re.findall(r"<joint\b", urdf_text)) == 0:
            print("Warning: resolved URDF has 0 joints.")

    search_paths = {root, selected.parent, urdf_path.parent}
    try:
        run_preview(
            cfg=cfg,
            calib_path=calib_path,
            urdf_path=urdf_path,
            extra_search_paths=search_paths,
            fixed_base=True,
            targets=targets,
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
