"""Map NN-detected sensor base from camera frame to optical-table coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_INCH_TO_M = 0.0254
# Prefer Windows calib runs; fall back to Linux software/ copy.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RUNS_DIRS = (
    _REPO_ROOT / "software_win" / "camera" / "calibration_table" / "runs",
    _REPO_ROOT / "software" / "camera" / "calibration_table" / "runs",
)


def _table_calib_reproj_px(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("metrics", {}).get("reprojection_error_px", float("inf")))
    except (json.JSONDecodeError, TypeError, ValueError, OSError):
        return None


def _iter_table_calib_runs(*, newest_first: bool = True) -> list[Path]:
    run_dirs: list[Path] = []
    for runs_dir in _DEFAULT_RUNS_DIRS:
        if not runs_dir.is_dir():
            continue
        run_dirs.extend(d for d in runs_dir.iterdir() if d.is_dir())
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=newest_first)
    return [run_dir / "table_calib.json" for run_dir in run_dirs]


def resolve_table_calib_path(
    configured: Path | None = None,
    *,
    max_reproj_px: float = 20.0,
    prefer_latest: bool = True,
) -> Path | None:
    """Use explicit path, else newest valid ``table_calib.json`` under calibration_table/runs."""
    if configured is not None:
        path = configured.resolve()
        if path.is_file():
            return path
    for candidate in _iter_table_calib_runs(newest_first=True):
        if not candidate.is_file():
            continue
        err = _table_calib_reproj_px(candidate)
        if err is None or err > max_reproj_px:
            continue
        if prefer_latest:
            return candidate
        break

    if prefer_latest:
        return None

    best: tuple[float, Path] | None = None
    for candidate in _iter_table_calib_runs(newest_first=True):
        if not candidate.is_file():
            continue
        err = _table_calib_reproj_px(candidate)
        if err is None or err > max_reproj_px:
            continue
        if best is None or err < best[0]:
            best = (err, candidate)
    return best[1] if best else None


def load_table_calib(table_calib_path: Path) -> dict[str, Any]:
    table_calib_path = table_calib_path.resolve()
    return json.loads(table_calib_path.read_text(encoding="utf-8"))


_homography_cache: dict[str, tuple[float, np.ndarray | None]] = {}


def build_table_plane_homography(calib_data: dict[str, Any]) -> np.ndarray | None:
    """Fit table-surface (x,y) -> image (u,v) homography from calibrated holes."""
    try:
        import cv2
    except ImportError:
        return None

    holes = calib_data.get("holes") or []
    if len(holes) < 4:
        return None
    src = np.array([[h["table_xy_m"][0], h["table_xy_m"][1]] for h in holes], dtype=np.float64)
    dst = np.array([h["image_px"] for h in holes], dtype=np.float64)
    homography, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    return homography


def _homography_for_calib(table_calib_path: Path) -> np.ndarray | None:
    path = table_calib_path.resolve()
    mtime = path.stat().st_mtime
    cached = _homography_cache.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    data = load_table_calib(path)
    homography = build_table_plane_homography(data)
    _homography_cache[str(path)] = (mtime, homography)
    return homography


def pixel_to_table_surface_xy(
    uv: tuple[int, int],
    table_calib_path: Path,
) -> tuple[float, float] | None:
    """Map image pixel to table-surface XY (m) using calib hole homography."""
    homography = _homography_for_calib(table_calib_path)
    if homography is None:
        return None
    homography_inv = np.linalg.inv(homography)
    u, v = float(uv[0]), float(uv[1])
    projected = homography_inv @ np.array([u, v, 1.0], dtype=np.float64)
    if abs(projected[2]) < 1e-9:
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def infer_pixel_table_pose(
    uv: tuple[int, int],
    table_calib_path: Path,
    *,
    z_m: float = 0.0,
) -> dict[str, Any] | None:
    """Table pose from pixel on the table plane (matches overlay grid labels)."""
    xy = pixel_to_table_surface_xy(uv, table_calib_path)
    if xy is None:
        return None
    data = load_table_calib(table_calib_path)
    pitch_m = float(data.get("grid_config", {}).get("pitch_m", _INCH_TO_M))
    tx, ty = xy
    grid_col, grid_row, col_f, row_f = table_xy_to_grid_indices(tx, ty, pitch_m)
    return {
        "table_xyz_m": [tx, ty, float(z_m)],
        "table_xy_mm": [tx * 1000.0, ty * 1000.0],
        "table_xy_in": [tx / _INCH_TO_M, ty / _INCH_TO_M],
        "grid_col": grid_col,
        "grid_row": grid_row,
        "grid_col_f": col_f,
        "grid_row_f": row_f,
        "pitch_m": pitch_m,
        "table_calib_path": str(table_calib_path),
        "table_pose_source": "calib_homography",
    }


def load_transform_table_cam(table_calib_path: Path) -> np.ndarray:
    """Load 4x4 camera->table transform from table_calib.json."""
    data = load_table_calib(table_calib_path)
    block = data.get("camera_in_table_frame", {})
    mat = block.get("transform_table_cam")
    if mat is None:
        raise KeyError(f"transform_table_cam missing in {table_calib_path}")
    transform = np.asarray(mat, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform_table_cam must be 4x4, got {transform.shape}")
    return transform


def cam_point_to_table_m(point_cam_m: np.ndarray, transform_table_cam: np.ndarray) -> np.ndarray:
    """Apply table_calib transform_table_cam to a 3D point in the camera frame."""
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(point_cam_m, dtype=np.float64)[:3]
    return (transform_table_cam @ p)[:3]


def base_cam_to_table_xy(
    base_cam_m: np.ndarray,
    table_calib_path: Path,
) -> tuple[float, float, float]:
    """
    Convert sensor-base 3D point (camera frame) to table XYZ (meters).

    Table frame matches calibration_table: origin at chosen hole, Z up from surface.
    """
    transform = load_transform_table_cam(table_calib_path)
    table_xyz = cam_point_to_table_m(base_cam_m, transform)
    return float(table_xyz[0]), float(table_xyz[1]), float(table_xyz[2])


def table_xy_to_grid_indices(
    table_x_m: float,
    table_y_m: float,
    pitch_m: float,
) -> tuple[int, int, float, float]:
    """Return nearest grid cell (col, row) and fractional indices.

    Matches saved table_calib holes: +Y table = +row (north on grow overlay).
    """
    pitch = max(float(pitch_m), 1e-9)
    col_f = table_x_m / pitch
    row_f = table_y_m / pitch
    return int(round(col_f)), int(round(row_f)), col_f, row_f


def table_robot_matrix(robot_in_table: Any) -> np.ndarray:
    """4x4 table->robot-base (robot origin in table frame)."""
    from .transforms import pose_to_matrix

    tx, ty, tz = robot_in_table.translation_m()
    rpy_rad = np.radians(np.asarray(robot_in_table.rpy_deg, dtype=np.float64))
    return pose_to_matrix((tx, ty, tz), rpy_rad)


def table_frame_pose_in_planner_world(
    robot_in_table: Any = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Pose of table-calib OBJ in planner world (origin = table 0,0).

    Temporary +90° about X undoes a Blender/export −90° X tilt so the mesh
    matches table +Z up. ``robot_in_table`` unused (kept for call sites).
    """
    import math

    import pybullet as p

    del robot_in_table
    # +90° X corrects apparent −90° X on the scanned OBJ.
    orn = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
    return (0.0, 0.0, 0.0), (float(orn[0]), float(orn[1]), float(orn[2]), float(orn[3]))


def robot_base_pose_in_table_world(
    robot_in_table: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Robot base pose in planner world (= table frame).

    Position/RPY come from ``physical_env/robot_in_table.yaml``. An additional
    180° Z rotation matches the URDF convention used by trajectory_planner.
    """
    import math

    import pybullet as p

    tx, ty, tz = robot_in_table.translation_m()
    rpy_rad = np.radians(np.asarray(robot_in_table.rpy_deg, dtype=np.float64))
    q_rpy = p.getQuaternionFromEuler([float(v) for v in rpy_rad])
    q_z180 = p.getQuaternionFromEuler([0.0, 0.0, math.pi])
    _, q_base = p.multiplyTransforms([0.0, 0.0, 0.0], q_rpy, [0.0, 0.0, 0.0], q_z180)
    return (float(tx), float(ty), float(tz)), (
        float(q_base[0]),
        float(q_base[1]),
        float(q_base[2]),
        float(q_base[3]),
    )


def table_point_to_planner_world_m(
    table_xyz_m: tuple[float, float, float],
    robot_in_table: Any | None = None,
) -> np.ndarray:
    """Table-frame point -> planner world (identity: planner world = table)."""
    del robot_in_table
    return np.asarray(table_xyz_m, dtype=np.float64)[:3].copy()


def table_point_to_robot_gui_m(
    table_xyz_m: tuple[float, float, float],
    robot_in_table: Any,
) -> np.ndarray:
    """Table-frame point -> robot-base coordinates (not planner world).

    Prefer ``table_point_to_planner_world_m`` for IK / planning targets (table =
    world). This remains for diagnostics that need robot-local XYZ.
    """
    from .transforms import invert_transform

    t_table_robot = table_robot_matrix(robot_in_table)
    t_robot_table = invert_transform(t_table_robot)
    p = np.array([*table_xyz_m, 1.0], dtype=np.float64)
    return (t_robot_table @ p)[:3]


def table_grasp_frame_summary(
    table_xyz_m: tuple[float, float, float],
    robot_in_table: Any,
) -> dict[str, Any]:
    """Diagnostic: table/planner point and matching robot-base coordinates."""
    inch = _INCH_TO_M
    robot_in = robot_in_table.translation_in
    planner_m = table_point_to_planner_world_m(table_xyz_m, robot_in_table)
    robot_m = table_point_to_robot_gui_m(table_xyz_m, robot_in_table)

    tx_mm, ty_mm, tz_mm = (float(table_xyz_m[i]) * 1000.0 for i in range(3))
    exp_gui = np.array(
        [
            -robot_in[0] * inch + float(table_xyz_m[0]),
            -robot_in[1] * inch + float(table_xyz_m[1]),
            float(table_xyz_m[2]) - robot_in[2] * inch,
        ],
        dtype=np.float64,
    )

    return {
        "table_xyz_mm": [round(tx_mm, 2), round(ty_mm, 2), round(tz_mm, 2)],
        "planner_world_m": planner_m.round(4).tolist(),
        "robot_base_m": robot_m.round(4).tolist(),
        "robot_base_mm": (robot_m * 1000.0).round(1).tolist(),
        "robot_gui_m": robot_m.round(4).tolist(),  # back-compat alias
        "robot_gui_mm": (robot_m * 1000.0).round(1).tolist(),
        "robot_gui_formula_mm": {
            "x": f"{-robot_in[0]:+.0f}in + {tx_mm:.0f}mm",
            "y": f"{-robot_in[1]:+.0f}in + {ty_mm:.0f}mm",
            "z": f"{tz_mm:.0f}mm - {robot_in[2]:.1f}in",
        },
        "robot_gui_formula_check_mm": (exp_gui * 1000.0).round(1).tolist(),
        "robot_in_table_in": list(robot_in),
    }


def robot_gui_to_table_m(
    robot_xyz_m: Sequence[float],
    robot_in_table: Any,
) -> np.ndarray:
    """Robot-base frame point -> optical table frame (m).

    Note: planner FK positions are already in table/world; use them directly for
    display. This converts robot-local coordinates only.
    """
    p = np.array([*robot_xyz_m[:3], 1.0], dtype=np.float64)
    return (table_robot_matrix(robot_in_table) @ p)[:3]


def format_table_xyz_display(table_xyz_m: Sequence[float]) -> str:
    """Human-readable table pose: inches and mm."""
    tx, ty, tz = (float(table_xyz_m[i]) for i in range(3))
    return (
        f"({tx / _INCH_TO_M:+.2f}, {ty / _INCH_TO_M:+.2f}, {tz / _INCH_TO_M:+.2f}) in  "
        f"({tx * 1000:+.0f}, {ty * 1000:+.0f}, {tz * 1000:+.0f}) mm"
    )


def format_planner_xyz_display(xyz_m: Sequence[float]) -> str:
    """Human-readable PyBullet planner / robot-base TCP target (m and mm)."""
    x, y, z = (float(xyz_m[i]) for i in range(3))
    return f"({x:+.4f}, {y:+.4f}, {z:+.4f}) m  ({x * 1000:+.1f}, {y * 1000:+.1f}, {z * 1000:+.1f}) mm"


def format_robot_in_table_display(robot_in_table: Any) -> str:
    """Robot base origin expressed in the optical table frame."""
    ti = robot_in_table.translation_in
    rpy = [round(float(v), 1) for v in robot_in_table.rpy_deg]
    return f"offset_in=[{ti[0]:+.1f}, {ti[1]:+.1f}, {ti[2]:+.1f}]  rpy_deg={rpy}"


def robot_point_table_display(robot_xyz_m: Sequence[float], robot_in_table: Any) -> str:
    """Format a robot-frame point as table-frame inches/mm."""
    return format_table_xyz_display(robot_gui_to_table_m(robot_xyz_m, robot_in_table))


def table_point_to_cam_m(
    table_xyz_m: tuple[float, float, float],
    table_calib_path: Path,
) -> np.ndarray:
    """Table-frame point -> camera optical frame (m) via table_calib."""
    data = load_table_calib(table_calib_path)
    block = data.get("table_to_camera", {})
    t_cam_table = np.asarray(block["transform_cam_table"], dtype=np.float64)
    if t_cam_table.shape != (4, 4):
        raise ValueError("transform_cam_table must be 4x4")
    p = np.array([*table_xyz_m, 1.0], dtype=np.float64)
    return (t_cam_table @ p)[:3]


def table_point_to_robot_m(
    table_xyz_m: tuple[float, float, float],
    *,
    table_calib_path: Path,
    camera_to_robot: Any,
) -> np.ndarray:
    """Legacy: table point via hand-eye chain (prefer table_point_to_planner_world_m)."""
    from .transforms import pose_to_matrix

    data = load_table_calib(table_calib_path)
    block = data.get("table_to_camera", {})
    t_cam_table = np.asarray(block["transform_cam_table"], dtype=np.float64)
    if t_cam_table.shape != (4, 4):
        raise ValueError("transform_cam_table must be 4x4")
    rpy_rad = np.radians(np.asarray(camera_to_robot.rpy_deg, dtype=np.float64))
    t_robot_cam = pose_to_matrix(camera_to_robot.translation_m, rpy_rad)
    p = np.array([*table_xyz_m, 1.0], dtype=np.float64)
    return (t_robot_cam @ t_cam_table @ p)[:3]


def table_point_to_pixel(
    table_xyz_m: tuple[float, float, float],
    calib_data: dict[str, Any],
    intrinsics: Any,
) -> tuple[int, int] | None:
    """Project a table-frame point to image pixels using saved PnP extrinsics."""
    try:
        import cv2
    except ImportError:
        return None

    block = calib_data.get("table_to_camera", {})
    rvec = np.asarray(block.get("rvec"), dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(block.get("tvec"), dtype=np.float64).reshape(3, 1)
    cam_mat = np.array(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
            [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)
    objp = np.array([[table_xyz_m[0], table_xyz_m[1], table_xyz_m[2]]], dtype=np.float64)
    projected, _ = cv2.projectPoints(objp, rvec, tvec, cam_mat, dist)
    u = int(round(float(projected[0, 0, 0])))
    v = int(round(float(projected[0, 0, 1])))
    return u, v


def infer_cam_point_table_pose(
    point_cam_m: np.ndarray,
    table_calib_path: Path,
) -> dict[str, Any]:
    """Table pose for any 3D point in the camera frame."""
    data = load_table_calib(table_calib_path)
    pitch_m = float(data.get("grid_config", {}).get("pitch_m", _INCH_TO_M))
    transform = load_transform_table_cam(table_calib_path)
    table_xyz = cam_point_to_table_m(point_cam_m, transform)
    tx, ty, tz = float(table_xyz[0]), float(table_xyz[1]), float(table_xyz[2])
    grid_col, grid_row, col_f, row_f = table_xy_to_grid_indices(tx, ty, pitch_m)
    return {
        "table_xyz_m": [tx, ty, tz],
        "table_xy_mm": [tx * 1000.0, ty * 1000.0],
        "table_xy_in": [tx / _INCH_TO_M, ty / _INCH_TO_M],
        "grid_col": grid_col,
        "grid_row": grid_row,
        "grid_col_f": col_f,
        "grid_row_f": row_f,
        "pitch_m": pitch_m,
        "table_calib_path": str(table_calib_path),
    }


def infer_base_table_pose(
    base_cam_m: np.ndarray,
    table_calib_path: Path,
) -> dict[str, Any]:
    """Full table pose for sensor base."""
    return infer_cam_point_table_pose(base_cam_m, table_calib_path)


def table_pose_offset_mm(
    base_pose: dict[str, Any],
    other_pose: dict[str, Any],
) -> dict[str, float]:
    """Offset from base to another point in table frame (mm)."""
    bx, by, bz = base_pose["table_xyz_m"]
    ox, oy, oz = other_pose["table_xyz_m"]
    dx_mm = (ox - bx) * 1000.0
    dy_mm = (oy - by) * 1000.0
    dz_mm = (oz - bz) * 1000.0
    dist_3d_mm = float(np.sqrt(dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm))
    dist_xy_mm = float(np.hypot(dx_mm, dy_mm))
    return {
        "dx_mm": dx_mm,
        "dy_mm": dy_mm,
        "dz_mm": dz_mm,
        "dist_3d_mm": dist_3d_mm,
        "dist_xy_mm": dist_xy_mm,
    }


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for table grid overlay") from exc
    return cv2


EDIT_BASE_BGR = (0, 255, 120)
NN_BASE_BGR = (0, 220, 80)


def draw_nn_base_marker(bgr: np.ndarray, u: int, v: int) -> None:
    """Simple center marker (Windows cascade does not use base ellipses)."""
    cv2 = _require_cv2()
    cv2.drawMarker(bgr, (int(u), int(v)), NN_BASE_BGR, cv2.MARKER_CROSS, 18, 2)


def draw_table_grid_overlay(
    bgr: np.ndarray,
    calib_data: dict[str, Any],
    *,
    near_table_xy: tuple[float, float] | None = None,
    label_radius_m: float = 0.10,
) -> None:
    """Draw saved calibration grid (holes + row/col lines) on a BGR camera image."""
    cv2 = _require_cv2()
    holes = calib_data.get("holes") or []
    if not holes:
        return

    line_color = (70, 70, 70)
    hole_color = (60, 140, 60)
    origin_color = (0, 0, 220)
    label_color = (180, 220, 180)

    by_row: dict[int, list[dict[str, Any]]] = {}
    by_col: dict[int, list[dict[str, Any]]] = {}
    for hole in holes:
        col = int(hole["grid_col"])
        row = int(hole["grid_row"])
        by_row.setdefault(row, []).append(hole)
        by_col.setdefault(col, []).append(hole)

    def _px(hole: dict[str, Any]) -> tuple[int, int]:
        u, v = hole["image_px"]
        return int(round(float(u))), int(round(float(v)))

    for row_holes in by_row.values():
        pts = [_px(h) for h in sorted(row_holes, key=lambda h: int(h["grid_col"]))]
        for i in range(len(pts) - 1):
            cv2.line(bgr, pts[i], pts[i + 1], line_color, 1, cv2.LINE_AA)

    for col_holes in by_col.values():
        pts = [_px(h) for h in sorted(col_holes, key=lambda h: int(h["grid_row"]))]
        for i in range(len(pts) - 1):
            cv2.line(bgr, pts[i], pts[i + 1], line_color, 1, cv2.LINE_AA)

    for hole in holes:
        u, v = _px(hole)
        r = max(2, int(round(float(hole.get("radius_px", 3)))))
        if hole.get("is_origin"):
            cv2.circle(bgr, (u, v), r + 2, origin_color, 2, cv2.LINE_AA)
            cv2.drawMarker(bgr, (u, v), origin_color, cv2.MARKER_CROSS, 14, 2)
        else:
            cv2.circle(bgr, (u, v), r, hole_color, 1, cv2.LINE_AA)

        if near_table_xy is not None:
            hxy = hole.get("table_xy_m")
            if hxy is None:
                continue
            dist = float(np.hypot(float(hxy[0]) - near_table_xy[0], float(hxy[1]) - near_table_xy[1]))
            if dist <= label_radius_m:
                cv2.putText(
                    bgr,
                    f"{hole['grid_col']},{hole['grid_row']}",
                    (u + 4, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    label_color,
                    1,
                    cv2.LINE_AA,
                )


def draw_base_table_annotation(
    bgr: np.ndarray,
    *,
    base_uv: tuple[int, int],
    table_pose: dict[str, Any],
    base_ellipse: Any = None,
    grasp_uv: tuple[int, int] | None = None,
    grasp_pose: dict[str, Any] | None = None,
    grasp_offset_mm: dict[str, float] | None = None,
) -> None:
    """Highlight sensor center and optional grasp point."""
    del table_pose, grasp_pose, base_ellipse
    u, v = int(base_uv[0]), int(base_uv[1])
    h_img, w_img = bgr.shape[:2]
    draw_nn_base_marker(bgr, u, v)

    if grasp_uv is not None and grasp_offset_mm is not None:
        gu, gv = int(grasp_uv[0]), int(grasp_uv[1])
        if 0 <= gu < w_img and 0 <= gv < h_img:
            cv2 = _require_cv2()
            cv2.line(bgr, (u, v), (gu, gv), (255, 120, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(bgr, (gu, gv), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)


def format_overlay_caption(
    table_pose: dict[str, Any],
    *,
    grasp_pose: dict[str, Any] | None = None,
    grasp_offset_mm: dict[str, float] | None = None,
    table_calib_name: str | None = None,
    nn_grasp_uv: tuple[int, int] | None = None,
    nn_base_uv: tuple[int, int] | None = None,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Caption lines for below-image panel (text, BGR color)."""
    tx_mm, ty_mm = table_pose["table_xy_mm"]
    tx_in, ty_in = table_pose.get("table_xy_in") or (tx_mm / (_INCH_TO_M * 1000.0), ty_mm / (_INCH_TO_M * 1000.0))
    grid_col = table_pose.get("grid_col")
    grid_row = table_pose.get("grid_row")
    grid_bit = ""
    if grid_col is not None and grid_row is not None:
        grid_bit = f" | grid ({grid_col}, {grid_row})"
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (
            f"Sensor base on table: ({tx_in:+.2f}, {ty_in:+.2f}) in "
            f"({tx_mm:+.0f}, {ty_mm:+.0f}) mm{grid_bit} | "
            f"Z={table_pose['table_xyz_m'][2] * 1000:.1f} mm",
            (120, 255, 180),
        ),
    ]
    if grasp_pose is not None and grasp_offset_mm is not None:
        gx_mm, gy_mm = grasp_pose["table_xy_mm"]
        gx_in, gy_in = grasp_pose.get("table_xy_in") or (
            gx_mm / (_INCH_TO_M * 1000.0),
            gy_mm / (_INCH_TO_M * 1000.0),
        )
        gcol = grasp_pose.get("grid_col")
        grow = grasp_pose.get("grid_row")
        ggrid = ""
        if gcol is not None and grow is not None:
            ggrid = f" | grid ({gcol}, {grow})"
        lines.append(
            (
                f"Grasp on sensor: ({gx_in:+.2f}, {gy_in:+.2f}) in "
                f"({gx_mm:+.0f}, {gy_mm:+.0f}) mm{ggrid} | "
                f"{grasp_offset_mm['dist_3d_mm']:.0f} mm from base "
                f"(dX={grasp_offset_mm['dx_mm']:+.1f}, dY={grasp_offset_mm['dy_mm']:+.1f}, "
                f"dZ={grasp_offset_mm['dz_mm']:+.1f} mm)",
                (180, 200, 255),
            )
        )
    coord_bits = []
    if nn_grasp_uv is not None:
        coord_bits.append(f"grasp {nn_grasp_uv}")
    if nn_base_uv is not None:
        coord_bits.append(f"base {nn_base_uv}")
    if coord_bits:
        lines.append((f"NN px: {'  |  '.join(coord_bits)}", (180, 220, 180)))
    if table_calib_name:
        lines.append((f"Table calib: {table_calib_name}", (200, 200, 200)))
    lines.append(
        (
            "Green = NN base ellipse | red = NN grasp | magenta = base-to-grasp | gray grid = table holes",
            (180, 200, 180),
        )
    )
    return lines
