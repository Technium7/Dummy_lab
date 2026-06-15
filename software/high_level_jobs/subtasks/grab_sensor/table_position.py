"""Map NN-detected sensor base from camera frame to optical-table coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_INCH_TO_M = 0.0254
_SOFTWARE_DIR = Path(__file__).resolve().parents[3]
_DEFAULT_RUNS_DIR = _SOFTWARE_DIR / "camera" / "calibration_table" / "runs"


def resolve_table_calib_path(
    configured: Path | None = None,
    *,
    max_reproj_px: float = 20.0,
) -> Path | None:
    """Use explicit path, else best recent table_calib.json by reprojection error."""
    if configured is not None:
        path = configured.resolve()
        if path.is_file():
            return path
    if not _DEFAULT_RUNS_DIR.is_dir():
        return None

    best: tuple[float, Path] | None = None
    for run_dir in sorted(_DEFAULT_RUNS_DIR.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        candidate = run_dir / "table_calib.json"
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            err = float(data.get("metrics", {}).get("reprojection_error_px", float("inf")))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if err > max_reproj_px:
            continue
        if best is None or err < best[0]:
            best = (err, candidate)
    return best[1] if best else None


def load_table_calib(table_calib_path: Path) -> dict[str, Any]:
    table_calib_path = table_calib_path.resolve()
    return json.loads(table_calib_path.read_text(encoding="utf-8"))


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
    """Return nearest grid cell (col, row) and fractional indices."""
    pitch = max(float(pitch_m), 1e-9)
    col_f = table_x_m / pitch
    row_f = -table_y_m / pitch
    return int(round(col_f)), int(round(row_f)), col_f, row_f


def table_robot_matrix(robot_in_table: Any) -> np.ndarray:
    """4x4 table->robot-base (robot origin in table frame)."""
    from .transforms import pose_to_matrix

    tx, ty, tz = robot_in_table.translation_m()
    rpy_rad = np.radians(np.asarray(robot_in_table.rpy_deg, dtype=np.float64))
    return pose_to_matrix((tx, ty, tz), rpy_rad)


def table_point_to_robot_gui_m(
    table_xyz_m: tuple[float, float, float],
    robot_in_table: Any,
) -> np.ndarray:
    """Table-frame point -> robot GUI / PyBullet planner world (m)."""
    from .transforms import invert_transform

    t_table_robot = table_robot_matrix(robot_in_table)
    t_robot_table = invert_transform(t_table_robot)
    p = np.array([*table_xyz_m, 1.0], dtype=np.float64)
    return (t_robot_table @ p)[:3]


def table_grasp_frame_summary(
    table_xyz_m: tuple[float, float, float],
    robot_in_table: Any,
) -> dict[str, Any]:
    """Diagnostic: table-frame point and matching robot GUI coordinates."""
    inch = _INCH_TO_M
    robot_in = robot_in_table.translation_in
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
        "robot_gui_m": robot_m.round(4).tolist(),
        "robot_gui_mm": (robot_m * 1000.0).round(1).tolist(),
        "robot_gui_formula_mm": {
            "x": f"{-robot_in[0]:+.0f}in + {tx_mm:.0f}mm",
            "y": f"{-robot_in[1]:+.0f}in + {ty_mm:.0f}mm",
            "z": f"{tz_mm:.0f}mm - {robot_in[2]:.1f}in",
        },
        "robot_gui_formula_check_mm": (exp_gui * 1000.0).round(1).tolist(),
        "robot_in_table_in": list(robot_in),
    }


def table_point_to_planner_world_m(
    table_xyz_m: tuple[float, float, float],
    robot_in_table: Any,
) -> np.ndarray:
    """Table point -> PyBullet planner world (same as robot GUI for this cell)."""
    return table_point_to_robot_gui_m(table_xyz_m, robot_in_table)


def robot_gui_to_table_m(
    robot_xyz_m: Sequence[float],
    robot_in_table: Any,
) -> np.ndarray:
    """Robot GUI / firmware base frame point -> optical table frame (m)."""
    p = np.array([*robot_xyz_m[:3], 1.0], dtype=np.float64)
    return (table_robot_matrix(robot_in_table) @ p)[:3]


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
    table_surface_uv: tuple[int, int] | None = None,
    grasp_uv: tuple[int, int] | None = None,
    grasp_pose: dict[str, Any] | None = None,
    grasp_offset_mm: dict[str, float] | None = None,
) -> None:
    """Highlight sensor base and optional grasp point with table-frame positions."""
    cv2 = _require_cv2()
    u, v = int(base_uv[0]), int(base_uv[1])
    h_img, w_img = bgr.shape[:2]

    cv2.drawMarker(bgr, (u, v), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 24, 2)
    cv2.circle(bgr, (u, v), 14, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(bgr, (u, v), 5, (0, 255, 255), -1, cv2.LINE_AA)

    if table_surface_uv is not None:
        su, sv = table_surface_uv
        if 0 <= su < w_img and 0 <= sv < h_img:
            cv2.drawMarker(bgr, (su, sv), (255, 200, 0), cv2.MARKER_DIAMOND, 16, 2)

    if grasp_uv is not None and grasp_offset_mm is not None:
        gu, gv = int(grasp_uv[0]), int(grasp_uv[1])
        if 0 <= gu < w_img and 0 <= gv < h_img:
            cv2.line(bgr, (u, v), (gu, gv), (255, 120, 255), 2, cv2.LINE_AA)
            mid_u = (u + gu) // 2
            mid_v = (v + gv) // 2
            cv2.putText(
                bgr,
                f"{grasp_offset_mm['dist_3d_mm']:.0f}mm",
                (mid_u + 6, mid_v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 120, 255),
                1,
                cv2.LINE_AA,
            )

    tx_mm, ty_mm = table_pose["table_xy_mm"]
    tx_in, ty_in = table_pose["table_xy_in"]

    lines = [
        f"BASE table: X={tx_mm:+.1f}mm  Y={ty_mm:+.1f}mm",
        f"           X={tx_in:+.2f}in   Y={ty_in:+.2f}in",
    ]
    if grasp_pose is not None and grasp_offset_mm is not None:
        gx_mm, gy_mm = grasp_pose["table_xy_mm"]
        gx_in, gy_in = grasp_pose["table_xy_in"]
        lines.extend(
            [
                f"GRASP table: X={gx_mm:+.1f}mm  Y={gy_mm:+.1f}mm",
                f"            X={gx_in:+.2f}in   Y={gy_in:+.2f}in",
                (
                    f"from base: dX={grasp_offset_mm['dx_mm']:+.1f}mm  "
                    f"dY={grasp_offset_mm['dy_mm']:+.1f}mm  "
                    f"{grasp_offset_mm['dist_3d_mm']:.0f}mm away"
                ),
            ]
        )

    tx0 = min(u + 16, w_img - 380)
    ty0 = max(v + 24, 60)
    for i, line in enumerate(lines):
        color = (0, 255, 120) if i < 2 else (120, 180, 255)
        cv2.putText(
            bgr,
            line,
            (tx0, ty0 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    panel_h = 112 if grasp_pose is not None else 88
    panel_y = h_img - panel_h
    cv2.rectangle(bgr, (0, panel_y), (w_img, h_img), (30, 30, 30), -1)
    summary = (
        f"Sensor base on table: ({tx_mm:+.1f}, {ty_mm:+.1f}) mm | "
        f"Z={table_pose['table_xyz_m'][2]*1000:.1f} mm"
    )
    cv2.putText(
        bgr,
        summary,
        (10, panel_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (120, 255, 180),
        1,
        cv2.LINE_AA,
    )
    if grasp_pose is not None and grasp_offset_mm is not None:
        gx_mm, gy_mm = grasp_pose["table_xy_mm"]
        grasp_summary = (
            f"Grasp on sensor: ({gx_mm:+.1f}, {gy_mm:+.1f}) mm | "
            f"{grasp_offset_mm['dist_3d_mm']:.0f} mm from base "
            f"(dX={grasp_offset_mm['dx_mm']:+.1f}, dY={grasp_offset_mm['dy_mm']:+.1f}, "
            f"dZ={grasp_offset_mm['dz_mm']:+.1f} mm)"
        )
        cv2.putText(
            bgr,
            grasp_summary,
            (10, panel_y + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 200, 255),
            1,
            cv2.LINE_AA,
        )
        legend_y = panel_y + 82
    else:
        legend_y = panel_y + 56
    cv2.putText(
        bgr,
        "Green X = base  |  red grasp = NN grasp  |  magenta line = base-to-grasp  |  gray grid = table holes",
        (10, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 200, 180),
        1,
        cv2.LINE_AA,
    )
