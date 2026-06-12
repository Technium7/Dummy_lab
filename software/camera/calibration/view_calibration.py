#!/usr/bin/env python3
"""PyBullet preview of a solved camera_to_robot extrinsic with the robot URDF."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_CALIBRATION_DIR = Path(__file__).resolve().parent
_SOFTWARE_DIR = _CALIBRATION_DIR.parents[1]
_MANUAL_CONTROL_DIR = _SOFTWARE_DIR / "manual_control"
for _path in (_SOFTWARE_DIR, _MANUAL_CONTROL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pybullet as p
import pybullet_data

from high_level_jobs.subtasks.grab_sensor.transforms import pose_to_matrix
from manual_control.ik_core import resolve_end_link
from manual_control.sim_setup import configure_pybullet_gui, load_ground_plane, load_robot_urdf, refresh_pybullet_display
from manual_control.urdf_tools import build_package_map, choose_description, discover_descriptions, materialize_urdf
from robot_config import DEFAULT_ROBOT, repo_root

from camera.calibration.card_config import CalibrationCardConfig, DEFAULT_END_LINK_TO_TCP_DISTANCE_M
from camera.calibration.fk import base_to_board_from_card, set_robot_joints_deg
from camera.calibration.hand_eye import _base_cam_from_params

_YELLOW = [1.0, 0.92, 0.15]
_GREEN_BOARD = [0.15, 0.9, 0.35]
_CYAN_PATTERN = [0.2, 0.85, 0.95]
_BLUE_CAMERA = [0.25, 0.65, 1.0]


def _cam_to_world(T_base_cam: np.ndarray, point_cam: Sequence[float]) -> list[float]:
    p_cam = np.asarray(point_cam, dtype=np.float64).reshape(3)
    p_base = T_base_cam[:3, :3] @ p_cam + T_base_cam[:3, 3]
    return [float(v) for v in p_base]


def _resolve_result_path(result_path: Path, root: Path) -> Path | None:
    candidates = [
        result_path,
        root / result_path,
        Path.cwd() / result_path,
        _CALIBRATION_DIR / result_path,
        _CALIBRATION_DIR / "runs" / result_path,
        _CALIBRATION_DIR / "runs" / result_path.name,
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


def _resolve_run_dir(run_dir: Path, root: Path) -> Path | None:
    candidates = [
        run_dir,
        root / run_dir,
        Path.cwd() / run_dir,
        _CALIBRATION_DIR / "runs" / run_dir,
        _CALIBRATION_DIR / "runs" / run_dir.name,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            cam = resolved / "camera_to_robot.json"
            if cam.is_file():
                return cam
            hist = resolved / "convergence_history.json"
            if hist.is_file():
                return hist
    return None


def _latest_result_file(runs_dir: Path) -> Path | None:
    if not runs_dir.is_dir():
        return None
    run_dirs = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        cam = run_dir / "camera_to_robot.json"
        if cam.is_file():
            return cam
        hist = run_dir / "convergence_history.json"
        if hist.is_file():
            return hist
    return None


def _mount_constraint_block(data: dict, path: Path) -> dict | None:
    block = data.get("camera_mount_constraint")
    if isinstance(block, dict):
        return block
    sibling = path.parent / "camera_to_robot.json"
    if sibling.is_file() and sibling != path:
        sibling_data = json.loads(sibling.read_text(encoding="utf-8"))
        block = sibling_data.get("camera_mount_constraint")
        if isinstance(block, dict):
            return block
    return None


def _load_transform_from_json(path: Path) -> tuple[np.ndarray, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    run_dir = path.parent.name

    def _meta(**kwargs) -> dict:
        return {"source": str(path), "run_dir": run_dir, **kwargs}

    if "camera_to_robot" in data and "samples" in data:
        mount = _mount_constraint_block(data, path)
        fitted_yaw = data.get("fitted_yaw_deg")
        origin = data.get("camera_origin_in_base_m") or data["camera_to_robot"]["translation_m"]
        if mount is not None and fitted_yaw is not None and not mount.get("fit_full_rotation", False):
            T = _base_cam_from_params(
                origin,
                roll_deg=float(mount["fixed_roll_deg"]),
                pitch_deg=float(mount["fixed_pitch_deg"]),
                yaw_deg=float(fitted_yaw),
            )
            return T, _meta(
                sample_count=data.get("sample_count"),
                method=data.get("method", ""),
                fitted_yaw_deg=float(fitted_yaw),
            )
        block = data["camera_to_robot"]
        T = pose_to_matrix(block["translation_m"], np.radians(block["rpy_deg"]))
        return T, _meta(sample_count=data.get("sample_count"), method=data.get("method", ""))

    if "history" in data and data["history"]:
        last = data["history"][-1]
        mount = _mount_constraint_block(data, path)
        fitted_yaw = last.get("fitted_yaw_deg")
        origin = last.get("camera_origin_m") or last.get("translation_m")
        if mount is not None and fitted_yaw is not None and origin is not None and not mount.get(
            "fit_full_rotation", False
        ):
            T = _base_cam_from_params(
                origin,
                roll_deg=float(mount["fixed_roll_deg"]),
                pitch_deg=float(mount["fixed_pitch_deg"]),
                yaw_deg=float(fitted_yaw),
            )
            return T, _meta(sample_count=last.get("sample_count"), method="convergence_history")
        T = pose_to_matrix(last["translation_m"], np.radians(last["rpy_deg"]))
        return T, _meta(sample_count=last.get("sample_count"), method="convergence_history")

    raise ValueError(f"Unrecognized calibration JSON format: {path}")


def _card_config_block(path: Path) -> dict | None:
    for json_path in (path, path.parent / "camera_to_robot.json"):
        if not json_path.is_file():
            continue
        block = json.loads(json_path.read_text(encoding="utf-8")).get("card_config")
        if isinstance(block, dict):
            return block
    return None


def _load_card_config(path: Path, args: argparse.Namespace) -> CalibrationCardConfig:
    return CalibrationCardConfig.from_cli_or_json(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_size_m=args.square_size_mm * 1e-3,
        board_mount=args.board_mount,
        grasp_point=args.grasp_point,
        end_link_to_tcp_xyz_m=args.end_link_tcp_xyz_m,
        end_link_to_tcp_distance_m=args.end_link_tcp_distance_m,
        json_block=_card_config_block(path),
    )


def _last_sample_joints_deg(path: Path) -> list[float] | None:
    for json_path in (path, path.parent / "camera_to_robot.json"):
        if not json_path.is_file():
            continue
        samples = json.loads(json_path.read_text(encoding="utf-8")).get("samples")
        if not samples:
            continue
        for sample in reversed(samples):
            if sample.get("board_found") and sample.get("joints_deg"):
                return [float(v) for v in sample["joints_deg"][:6]]
    return None


def _draw_attached_polyline(
    robot_id: int,
    link_idx: int,
    points: Sequence[Sequence[float]],
    color: Sequence[float],
    width: float,
) -> None:
    for i in range(len(points)):
        p.addUserDebugLine(
            list(points[i]),
            list(points[(i + 1) % len(points)]),
            color,
            width,
            0.0,
            parentObjectUniqueId=robot_id,
            parentLinkIndex=link_idx,
        )


def _draw_yellow_cross_attached(
    robot_id: int,
    link_idx: int,
    local_point: Sequence[float],
    *,
    half_len_m: float = 0.018,
) -> None:
    tip = [float(local_point[0]), float(local_point[1]), float(local_point[2])]
    for axis in (0, 1):
        start = tip.copy()
        end = tip.copy()
        start[axis] -= half_len_m
        end[axis] += half_len_m
        p.addUserDebugLine(
            start,
            end,
            _YELLOW,
            3.0,
            0.0,
            parentObjectUniqueId=robot_id,
            parentLinkIndex=link_idx,
        )


def draw_board_gizmo_attached(robot_id: int, link_idx: int, card_cfg: CalibrationCardConfig) -> None:
    """Green = board edge, cyan = OpenCV pattern, yellow = TCP + board origin."""
    outer, inner = card_cfg.board_outline_corners_board_frame()
    outer_link = [list(card_cfg.board_point_in_link_frame_m(c)) for c in outer]
    inner_link = [list(card_cfg.board_point_in_link_frame_m(c)) for c in inner]
    _draw_attached_polyline(robot_id, link_idx, outer_link, _GREEN_BOARD, 2.5)
    _draw_attached_polyline(robot_id, link_idx, inner_link, _CYAN_PATTERN, 2.0)
    _draw_yellow_cross_attached(robot_id, link_idx, card_cfg.end_link_to_tcp_xyz_m)
    _draw_yellow_cross_attached(robot_id, link_idx, card_cfg.board_origin_in_link_frame_m)


def draw_camera_gizmo(
    T_base_cam: np.ndarray,
    *,
    half_width_m: float = 0.04,
    half_height_m: float = 0.025,
    optical_axis_m: float = 0.12,
) -> None:
    hw, hh = float(half_width_m), float(half_height_m)
    corners_cam = [(-hw, -hh, 0.0), (hw, -hh, 0.0), (hw, hh, 0.0), (-hw, hh, 0.0)]
    corners_world = [_cam_to_world(T_base_cam, c) for c in corners_cam]
    center = _cam_to_world(T_base_cam, (0.0, 0.0, 0.0))
    look_tip = _cam_to_world(T_base_cam, (0.0, 0.0, optical_axis_m))
    for i in range(4):
        p.addUserDebugLine(corners_world[i], corners_world[(i + 1) % 4], _BLUE_CAMERA, 2.5, 0.0)
    p.addUserDebugLine(center, look_tip, _YELLOW, 3.0, 0.0)


def run_preview(
    *,
    result_path: Path,
    urdf_path: Path,
    extra_search_paths: set[Path],
    fixed_base: bool,
    joints_deg: list[float] | None,
    end_link_hint: str,
    card_cfg: CalibrationCardConfig,
    half_width_m: float,
    half_height_m: float,
    optical_axis_m: float,
) -> None:
    T_base_cam, meta = _load_transform_from_json(result_path)
    origin = T_base_cam[:3, 3]

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
        load_ground_plane()
        robot_id = load_robot_urdf(urdf_path, fixed_base=fixed_base)

        if joints_deg is not None:
            set_robot_joints_deg(robot_id, joints_deg)

        end_link_idx = resolve_end_link(robot_id, end_link_hint)
        draw_camera_gizmo(
            T_base_cam,
            half_width_m=half_width_m,
            half_height_m=half_height_m,
            optical_axis_m=optical_axis_m,
        )

        if joints_deg is not None:
            draw_board_gizmo_attached(robot_id, end_link_idx, card_cfg)
            t_board = base_to_board_from_card(robot_id, end_link_idx, card_cfg)
            print(
                f"[view_calibration] Board origin (FK): "
                f"[{t_board[0,3]:+.3f}, {t_board[1,3]:+.3f}, {t_board[2,3]:+.3f}] m"
            )

        cx, cy, cz = float(origin[0]), float(origin[1]), float(origin[2])
        p.resetDebugVisualizerCamera(
            cameraDistance=max(0.55, float(np.linalg.norm(origin)) * 1.35),
            cameraYaw=125,
            cameraPitch=-32,
            cameraTargetPosition=[cx * 0.45, cy * 0.45, max(0.08, cz * 0.45 + 0.06)],
        )
        refresh_pybullet_display(steps=3)

        print(f"[view_calibration] Run: {meta.get('run_dir', result_path.parent.name)}")
        print(
            f"[view_calibration] Camera origin: "
            f"[{origin[0]:+.3f}, {origin[1]:+.3f}, {origin[2]:+.3f}] m"
        )
        print(
            "[view_calibration] Blue = camera, green/cyan = board, yellow = TCP + pattern origin. "
            f"TCP offset (link m): {card_cfg.end_link_to_tcp_xyz_m}"
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
        description="Preview a solved camera_to_robot pose in PyBullet with the robot URDF.",
    )
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT)
    parser.add_argument("--end-link", type=str, default="", help="End link name or index (default: last link).")
    parser.add_argument("--fixed-base", action="store_true", default=True)
    parser.add_argument("--floating-base", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("software/camera/calibration/runs"),
    )
    parser.add_argument("--joints-deg", type=float, nargs=6, default=None)
    parser.add_argument("--use-last-sample-pose", action="store_true", default=True)
    parser.add_argument("--home-pose", action="store_true")
    parser.add_argument("--camera-width-m", type=float, default=0.08)
    parser.add_argument("--camera-height-m", type=float, default=0.05)
    parser.add_argument("--optical-axis-m", type=float, default=0.12)
    parser.add_argument("--squares-x", type=int, default=10)
    parser.add_argument("--squares-y", type=int, default=10)
    parser.add_argument("--square-size-mm", type=float, default=10.128)
    parser.add_argument("--board-mount", choices=("gripper_xy", "gripper_xz", "gripper_yz"), default="gripper_yz")
    parser.add_argument(
        "--grasp-point",
        choices=("corner_tip", "origin_corner_tip", "link_side_edge", "edge_center", "far_corner_tip"),
        default="corner_tip",
    )
    parser.add_argument("--end-link-tcp-xyz-m", type=float, nargs=3, default=None)
    parser.add_argument(
        "--end-link-tcp-distance-m",
        type=float,
        default=None,
        help=f"Pinch distance (m) along link6 −Y (default {DEFAULT_END_LINK_TO_TCP_DISTANCE_M}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    fixed_base = args.fixed_base and not args.floating_base

    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else root / args.runs_dir

    if args.run_dir is not None:
        result_path = _resolve_run_dir(args.run_dir, root)
        if result_path is None:
            raise SystemExit(f"No result JSON found in run dir: {args.run_dir}")
    elif args.result is not None:
        result_path = _resolve_result_path(args.result, root)
        if result_path is None:
            raise SystemExit(f"Result file not found: {args.result}")
    else:
        result_path = _latest_result_file(runs_dir)
        if result_path is None:
            raise SystemExit(f"No calibration result found under {runs_dir}")

    joints_deg = list(args.joints_deg) if args.joints_deg is not None else None
    if args.use_last_sample_pose and not args.home_pose and joints_deg is None:
        joints_deg = _last_sample_joints_deg(result_path)

    if args.end_link_tcp_xyz_m is not None:
        args.end_link_tcp_xyz_m = tuple(args.end_link_tcp_xyz_m)

    card_cfg = _load_card_config(result_path, args)

    selected = choose_description(discover_descriptions(root), args.robot, root)
    urdf_path, temp_file = materialize_urdf(selected, build_package_map(root))
    if selected.suffix.lower() == ".xacro":
        urdf_text = urdf_path.read_text(encoding="utf-8")
        if len(re.findall(r"<joint\b", urdf_text)) == 0:
            print("Warning: resolved URDF has 0 joints.")

    search_paths = {root, selected.parent, urdf_path.parent}
    try:
        run_preview(
            result_path=result_path,
            urdf_path=urdf_path,
            extra_search_paths=search_paths,
            fixed_base=fixed_base,
            joints_deg=joints_deg,
            end_link_hint=args.end_link,
            card_cfg=card_cfg,
            half_width_m=args.camera_width_m * 0.5,
            half_height_m=args.camera_height_m * 0.5,
            optical_axis_m=args.optical_axis_m,
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
