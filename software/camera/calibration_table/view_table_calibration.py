#!/usr/bin/env python3
"""PyBullet preview: optical table frame, robot base pose, and calibrated camera."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

_TABLE_CALIB_DIR = Path(__file__).resolve().parent
_SOFTWARE_DIR = _TABLE_CALIB_DIR.parents[1]
_MANUAL_CONTROL_DIR = _SOFTWARE_DIR / "manual_control"
for _path in (_SOFTWARE_DIR, _MANUAL_CONTROL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pybullet as p
import pybullet_data

from manual_control.sim_setup import (
    ROBOT_BASE_ORIENTATION,
    configure_pybullet_gui,
    refresh_pybullet_display,
)
from manual_control.urdf_tools import build_package_map, choose_description, discover_descriptions, materialize_urdf
from robot_config import DEFAULT_ROBOT, repo_root

INCH_TO_M = 0.0254


def _pose_to_matrix(translation_m: tuple[float, float, float], rpy_rad: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.array(
        p.getMatrixFromQuaternion(p.getQuaternionFromEuler(list(rpy_rad))),
        dtype=np.float64,
    ).reshape(3, 3)
    mat[:3, 3] = np.asarray(translation_m, dtype=np.float64)[:3]
    return mat


_COLOR_TABLE = [0.55, 0.55, 0.58]
_COLOR_HOLE = [0.12, 0.12, 0.14]
_COLOR_ORIGIN = [1.0, 0.15, 0.15]
_COLOR_ROBOT_AXES = [1.0, 0.85, 0.2]
_COLOR_CAMERA = [0.25, 0.65, 1.0]
_COLOR_LOOK = [1.0, 0.92, 0.15]


def _resolve_calib_path(path: Path, root: Path) -> Path | None:
    candidates = [
        path,
        root / path,
        Path.cwd() / path,
        _TABLE_CALIB_DIR / path,
        _TABLE_CALIB_DIR / "runs" / path,
        _TABLE_CALIB_DIR / "runs" / path.name,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _latest_calib_json(runs_dir: Path) -> Path | None:
    if not runs_dir.is_dir():
        return None
    runs = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run_dir in runs:
        candidate = run_dir / "table_calib.json"
        if candidate.is_file():
            return candidate
    return None


def _load_table_cam_transform(path: Path) -> tuple[np.ndarray, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("camera_in_table_frame")
    if not isinstance(block, dict):
        raise ValueError(f"Missing camera_in_table_frame in {path}")
    mat = np.asarray(block["transform_table_cam"], dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError("transform_table_cam must be 4x4")
    return mat, data


def _xform_point(T: np.ndarray, local: tuple[float, float, float]) -> list[float]:
    p_local = np.array([*local, 1.0], dtype=np.float64)
    p_world = T @ p_local
    return [float(p_world[0]), float(p_world[1]), float(p_world[2])]


def _draw_camera_gizmo(
    T_table_cam: np.ndarray,
    *,
    half_width_m: float = 0.04,
    half_height_m: float = 0.025,
    optical_axis_m: float = 0.12,
) -> None:
    hw, hh = float(half_width_m), float(half_height_m)
    corners = [(-hw, -hh, 0.0), (hw, -hh, 0.0), (hw, hh, 0.0), (-hw, hh, 0.0)]
    corners_w = [_xform_point(T_table_cam, c) for c in corners]
    center = _xform_point(T_table_cam, (0.0, 0.0, 0.0))
    look_tip = _xform_point(T_table_cam, (0.0, 0.0, optical_axis_m))
    for i in range(4):
        p.addUserDebugLine(corners_w[i], corners_w[(i + 1) % 4], _COLOR_CAMERA, 2.5, 0.0)
    p.addUserDebugLine(center, look_tip, _COLOR_LOOK, 3.0, 0.0)


def _draw_table_grid(data: dict, *, max_holes: int = 400) -> None:
    holes = data.get("holes") or []
    pitch_m = float(data.get("grid_config", {}).get("pitch_m", INCH_TO_M))
    if not holes:
        half = 8 * pitch_m
        p.addUserDebugLine([-half, 0, 0], [half, 0, 0], [0.35, 0.35, 0.35], 1.0, 0.0)
        p.addUserDebugLine([0, -half, 0], [0, half, 0], [0.35, 0.35, 0.35], 1.0, 0.0)
        return

    step = max(1, len(holes) // max_holes)
    for hole in holes[::step]:
        xy = hole.get("table_xy_m")
        if xy is None:
            continue
        x, y = float(xy[0]), float(xy[1])
        p.addUserDebugLine([x, y, 0.0], [x, y, 0.004], _COLOR_HOLE, 2.0, 0.0)

    cols = [float(h["table_xy_m"][0]) for h in holes if h.get("table_xy_m")]
    rows = [float(h["table_xy_m"][1]) for h in holes if h.get("table_xy_m")]
    if cols and rows:
        pad = pitch_m * 0.5
        x0, x1 = min(cols) - pad, max(cols) + pad
        y0, y1 = min(rows) - pad, max(rows) + pad
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            p.addUserDebugLine([a[0], a[1], 0.0], [b[0], b[1], 0.0], _COLOR_TABLE, 1.5, 0.0)

    p.addUserDebugLine([-0.03, 0, 0], [0.03, 0, 0], _COLOR_ORIGIN, 3.0, 0.0)
    p.addUserDebugLine([0, -0.03, 0], [0, 0.03, 0], _COLOR_ORIGIN, 3.0, 0.0)
    p.addUserDebugLine([0, 0, 0], [0, 0, 0.05], _COLOR_ORIGIN, 3.0, 0.0)


def _robot_base_in_table(
    x_m: float,
    y_m: float,
    z_m: float,
    rpy_deg: tuple[float, float, float],
) -> tuple[list[float], list[float]]:
    table_pos = [x_m, y_m, z_m]
    table_quat = p.getQuaternionFromEuler([math.radians(v) for v in rpy_deg])
    pos, quat = p.multiplyTransforms(
        table_pos,
        table_quat,
        [0.0, 0.0, 0.0],
        ROBOT_BASE_ORIENTATION,
    )
    return list(pos), list(quat)


def _draw_robot_base_marker(pos: list[float]) -> None:
    p.addUserDebugLine(
        [pos[0] - 0.04, pos[1], pos[2]],
        [pos[0] + 0.04, pos[1], pos[2]],
        _COLOR_ROBOT_AXES,
        3.0,
        0.0,
    )
    p.addUserDebugLine(
        [pos[0], pos[1] - 0.04, pos[2]],
        [pos[0], pos[1] + 0.04, pos[2]],
        _COLOR_ROBOT_AXES,
        3.0,
        0.0,
    )


def run_preview(
    *,
    calib_path: Path,
    urdf_path: Path,
    extra_search_paths: set[Path],
    fixed_base: bool,
    robot_xyz_m: tuple[float, float, float],
    robot_rpy_deg: tuple[float, float, float],
    half_width_m: float,
    half_height_m: float,
    optical_axis_m: float,
) -> None:
    T_table_cam, data = _load_table_cam_transform(calib_path)
    cam_pos = T_table_cam[:3, 3]

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

        robot_pos, robot_quat = _robot_base_in_table(*robot_xyz_m, robot_rpy_deg)
        robot_id = p.loadURDF(
            str(urdf_path),
            basePosition=robot_pos,
            baseOrientation=robot_quat,
            useFixedBase=fixed_base,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        _draw_table_grid(data)
        _draw_camera_gizmo(
            T_table_cam,
            half_width_m=half_width_m,
            half_height_m=half_height_m,
            optical_axis_m=optical_axis_m,
        )
        _draw_robot_base_marker(robot_pos)

        T_table_robot = _pose_to_matrix(list(robot_xyz_m), np.radians(robot_rpy_deg))
        T_robot_cam = np.linalg.inv(T_table_robot) @ T_table_cam

        span = max(
            float(np.linalg.norm(cam_pos)),
            float(np.linalg.norm(robot_pos)),
            0.5,
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=span * 2.0,
            cameraYaw=125,
            cameraPitch=-35,
            cameraTargetPosition=[
                0.5 * (cam_pos[0] + robot_pos[0]),
                0.5 * (cam_pos[1] + robot_pos[1]),
                max(0.08, 0.35 * (cam_pos[2] + robot_pos[2])),
            ],
        )
        refresh_pybullet_display(steps=3)

        print(f"[view_table_calib] Calibration: {calib_path}")
        print(
            f"[view_table_calib] Table origin (0,0,0) = calibration hole; +Z up from table"
        )
        print(
            f"[view_table_calib] Robot base in table frame (m): "
            f"[{robot_xyz_m[0]:+.4f}, {robot_xyz_m[1]:+.4f}, {robot_xyz_m[2]:+.4f}]"
        )
        print(
            f"[view_table_calib] Camera in table frame (m): "
            f"[{cam_pos[0]:+.4f}, {cam_pos[1]:+.4f}, {cam_pos[2]:+.4f}]"
        )
        t_rc = T_robot_cam[:3, 3]
        print(
            f"[view_table_calib] Camera in robot base frame (m): "
            f"[{t_rc[0]:+.4f}, {t_rc[1]:+.4f}, {t_rc[2]:+.4f}]"
        )
        print(
            "[view_table_calib] Red = table origin  yellow = robot base  "
            "blue/yellow = camera  dark ticks = holes"
        )

        while p.isConnected():
            time.sleep(0.05)
    finally:
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview table calibration: table frame, robot base, and camera in PyBullet.",
    )
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT)
    parser.add_argument("--fixed-base", action="store_true", default=True)
    parser.add_argument("--floating-base", action="store_true")
    parser.add_argument("--calib", type=Path, default=None, help="table_calib.json path")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("software/camera/calibration_table/runs"),
    )
    parser.add_argument(
        "--robot-x-in",
        type=float,
        default=-6.0,
        help="Robot base X in table frame (inches, default: -6)",
    )
    parser.add_argument(
        "--robot-y-in",
        type=float,
        default=-5.0,
        help="Robot base Y in table frame (inches, default: -5)",
    )
    parser.add_argument(
        "--robot-z-in",
        type=float,
        default=1.25,
        help="Robot base Z in table frame (inches, default: 1.25)",
    )
    parser.add_argument(
        "--robot-rpy-deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Robot base orientation in table frame (degrees)",
    )
    parser.add_argument("--camera-width-m", type=float, default=0.08)
    parser.add_argument("--camera-height-m", type=float, default=0.05)
    parser.add_argument("--optical-axis-m", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    fixed_base = args.fixed_base and not args.floating_base
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else root / args.runs_dir

    if args.calib is not None:
        calib_path = _resolve_calib_path(args.calib, root)
        if calib_path is None:
            raise SystemExit(f"Calibration file not found: {args.calib}")
    else:
        calib_path = _latest_calib_json(runs_dir)
        if calib_path is None:
            raise SystemExit(f"No table_calib.json found under {runs_dir}")

    robot_xyz_m = (
        args.robot_x_in * INCH_TO_M,
        args.robot_y_in * INCH_TO_M,
        args.robot_z_in * INCH_TO_M,
    )
    robot_rpy_deg = tuple(float(v) for v in args.robot_rpy_deg)

    selected = choose_description(discover_descriptions(root), args.robot, root)
    urdf_path, temp_file = materialize_urdf(selected, build_package_map(root))
    if selected.suffix.lower() == ".xacro":
        urdf_text = urdf_path.read_text(encoding="utf-8")
        if len(re.findall(r"<joint\b", urdf_text)) == 0:
            print("Warning: resolved URDF has 0 joints.")

    search_paths = {root, selected.parent, urdf_path.parent}
    try:
        run_preview(
            calib_path=calib_path,
            urdf_path=urdf_path,
            extra_search_paths=search_paths,
            fixed_base=fixed_base,
            robot_xyz_m=robot_xyz_m,
            robot_rpy_deg=robot_rpy_deg,
            half_width_m=args.camera_width_m * 0.5,
            half_height_m=args.camera_height_m * 0.5,
            optical_axis_m=args.optical_axis_m,
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
