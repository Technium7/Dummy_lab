"""Hole-grid detection, grow, region filter, and full 6-DOF solvePnP pose solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .grid_config import TableGridConfig
from .grid_extend import (
    GridModel,
    anchor_model_to_origin,
    draw_simple_overlay,
    fit_robust_grid_model,
    remap_holes_to_model,
)
from .grid_grow import draw_grown_grid, grow_grid_from_origin
from .hole_detect import RawHoleCandidate, estimate_pitch_px, find_hole_candidates
from .grid_types import DetectedHole, GridSlot
from .region_select import (
    draw_region_filtered_grid,
    partition_holes_by_polygon,
    snap_hole_from_click,
)


@dataclass
class GridObservation:
    found: bool
    holes: list[DetectedHole]
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    reprojection_error_px: float
    pitch_px: float
    origin_hole: DetectedHole | None
    annotated_bgr: np.ndarray | None
    origin_model_cell: tuple[int, int] | None = None
    origin_lock_px: tuple[float, float] | None = None
    message: str = ""
    debug_edges_bgr: np.ndarray | None = None
    candidates_before_filter: int = 0
    grid_slots: list[GridSlot] | None = None
    holes_found: int = 0
    holes_recovered: int = 0
    holes_missing: int = 0
    holes_rejected: int = 0
    all_grown_holes: list[DetectedHole] | None = None
    detection_rejected: list[DetectedHole] | None = None


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for table calibration. "
            "Install with: pip install -r requirements-vision.txt"
        ) from exc
    return cv2


def camera_matrix_from_intrinsics(intr: Any) -> np.ndarray:
    return np.array(
        [
            [float(intr.fx), 0.0, float(intr.ppx)],
            [0.0, float(intr.fy), float(intr.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_from_intrinsics(intr: Any) -> np.ndarray:
    coeffs = getattr(intr, "coeffs", None)
    if coeffs is None:
        return np.zeros(5, dtype=np.float64)
    return np.asarray(list(coeffs), dtype=np.float64)[:5]


def _candidates_to_holes(candidates: list[RawHoleCandidate]) -> list[DetectedHole]:
    return [
        DetectedHole(
            center_px=c.center_px,
            radius_px=c.radius_px,
            circularity=c.circularity,
            nn_score=c.circularity,
            source=c.source,
            status="detected",
        )
        for c in candidates
    ]


def _choose_origin_hole(
    holes: list[DetectedHole],
    *,
    image_center_px: tuple[float, float] | None,
    click_px: tuple[float, float] | None,
    origin_grid_cell: tuple[int, int] | None,
) -> DetectedHole | None:
    if not holes:
        return None
    if origin_grid_cell is not None:
        oc, orow = origin_grid_cell
        exact = [h for h in holes if h.grid_col == oc and h.grid_row == orow]
        if exact:
            return max(exact, key=lambda h: h.circularity)
        return min(holes, key=lambda h: (h.grid_col - oc) ** 2 + (h.grid_row - orow) ** 2)
    if click_px is not None:
        cx, cy = click_px
        return min(holes, key=lambda h: (h.center_px[0] - cx) ** 2 + (h.center_px[1] - cy) ** 2)
    if image_center_px is not None:
        cx, cy = image_center_px
        return min(holes, key=lambda h: (h.center_px[0] - cx) ** 2 + (h.center_px[1] - cy) ** 2)
    return min(holes, key=lambda h: h.grid_col * h.grid_col + h.grid_row * h.grid_row)


def _reindex_relative_to_origin(holes: list[DetectedHole], origin: DetectedHole) -> None:
    oc, orow = origin.grid_col, origin.grid_row
    for hole in holes:
        hole.grid_col -= oc
        hole.grid_row -= orow
        hole.is_origin = hole is origin


def _object_points_for_holes(holes: list[DetectedHole], pitch_m: float) -> np.ndarray:
    """Table frame: +X = grid column, +Y = -grid row (Z-up right-handed), Z = 0 on surface."""
    pts = np.zeros((len(holes), 3), dtype=np.float64)
    for i, hole in enumerate(holes):
        x = hole.grid_col * pitch_m
        y = -hole.grid_row * pitch_m
        hole.table_xy_m = (x, y)
        pts[i, 0] = x
        pts[i, 1] = y
    return pts


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    rot, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = tvec.reshape(3)
    return mat


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def _enforce_table_z_up(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Safety net if planar PnP still returns camera below the table (+Z is up from surface).
    Uses a 180° flip about table X (proper rotation, valid rvec/tvec).
    """
    transform_cam_table = rvec_tvec_to_matrix(rvec, tvec)
    transform_table_cam = invert_transform(transform_cam_table)
    if transform_table_cam[2, 3] >= 0.0:
        return rvec.reshape(3), tvec.reshape(3)

    flip_x = np.diag([1.0, -1.0, -1.0, 1.0])
    transform_table_cam = flip_x @ transform_table_cam
    transform_cam_table = invert_transform(transform_table_cam)
    rvec_out, _ = _require_cv2().Rodrigues(transform_cam_table[:3, :3])
    return rvec_out.reshape(3), transform_cam_table[:3, 3].copy()


def _slots_from_holes(holes: list[DetectedHole]) -> list[GridSlot]:
    return [
        GridSlot(h.grid_col, h.grid_row, h.center_px, h.status, h)
        for h in holes
    ]


def _establish_origin(
    holes: list[DetectedHole],
    *,
    image_center_px: tuple[float, float],
    origin_click_px: tuple[float, float] | None,
    origin_lock_px: tuple[float, float] | None,
    origin_grid_cell: tuple[int, int] | None,
    mean_radius_px: float,
    model: GridModel | None = None,
) -> tuple[DetectedHole | None, tuple[int, int] | None, tuple[float, float] | None]:
    if origin_lock_px is not None:
        lx, ly = origin_lock_px
        if model is not None:
            oc, orow = model.pixel_to_cell(lx, ly)
        elif origin_grid_cell is not None:
            oc, orow = origin_grid_cell
        else:
            return None, None, None
        holes[:] = [h for h in holes if not (h.grid_col == oc and h.grid_row == orow)]
        origin = DetectedHole(
            center_px=origin_lock_px,
            radius_px=mean_radius_px,
            circularity=1.0,
            nn_score=1.0,
            grid_col=oc,
            grid_row=orow,
            source="origin_lock",
            status="locked",
        )
        _reindex_relative_to_origin(holes, origin)
        origin.grid_col = 0
        origin.grid_row = 0
        origin.is_origin = True
        holes.append(origin)
        return origin, (oc, orow), None

    origin = _choose_origin_hole(
        holes,
        image_center_px=image_center_px,
        click_px=origin_click_px,
        origin_grid_cell=origin_grid_cell,
    )
    if origin is None:
        return None, None, None

    if model is not None:
        oc, orow = model.pixel_to_cell(origin.center_px[0], origin.center_px[1])
        origin.grid_col = oc
        origin.grid_row = orow
    origin_model_cell = (origin.grid_col, origin.grid_row)
    new_lock_px = origin.center_px if origin_click_px is not None else None
    _reindex_relative_to_origin(holes, origin)
    return origin, origin_model_cell, new_lock_px


def _observation(
    *,
    found: bool,
    holes: list[DetectedHole],
    outliers: list[DetectedHole],
    slots: list[GridSlot],
    rvec,
    tvec,
    err: float,
    pitch_px: float,
    origin: DetectedHole | None,
    origin_model_cell: tuple[int, int] | None,
    origin_lock_px: tuple[float, float] | None,
    annotated: np.ndarray,
    msg: str,
    edges_bgr,
    n_raw: int,
) -> GridObservation:
    return GridObservation(
        found=found,
        holes=holes,
        rvec=rvec,
        tvec=tvec,
        reprojection_error_px=err,
        pitch_px=pitch_px,
        origin_hole=origin,
        origin_model_cell=origin_model_cell,
        origin_lock_px=origin_lock_px,
        annotated_bgr=annotated,
        message=msg,
        debug_edges_bgr=edges_bgr,
        candidates_before_filter=n_raw,
        grid_slots=slots,
        holes_found=len(holes),
        holes_recovered=0,
        holes_missing=0,
        holes_rejected=len(outliers),
    )


def _solve_pnp_for_holes(
    holes: list[DetectedHole],
    cfg: TableGridConfig,
    intrinsics: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    cv2 = _require_cv2()
    pose_holes = [h for h in holes if h.status != "rejected"]
    if len(pose_holes) < 3:
        return None, None, float("inf")

    objp = np.ascontiguousarray(_object_points_for_holes(pose_holes, cfg.pitch_m), dtype=np.float64)
    imgp = np.ascontiguousarray(
        np.array([h.center_px for h in pose_holes], dtype=np.float64).reshape(-1, 2),
        dtype=np.float64,
    )
    cam_mat = camera_matrix_from_intrinsics(intrinsics)
    dist = distortion_from_intrinsics(intrinsics)

    try:
        flags = cv2.SOLVEPNP_ITERATIVE if len(pose_holes) >= 4 else cv2.SOLVEPNP_SQPNP
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, cam_mat, dist, flags=flags)
        if not ok:
            return None, None, float("inf")
        rvec, tvec = _enforce_table_z_up(rvec.reshape(3), tvec.reshape(3))
    except cv2.error:
        return None, None, float("inf")

    projected, _ = cv2.projectPoints(objp, rvec, tvec, cam_mat, dist)
    err = float(np.linalg.norm(projected.reshape(-1, 2) - imgp, axis=1).mean())
    return rvec, tvec.reshape(3), err


def solve_pose_from_holes(
    holes: list[DetectedHole],
    cfg: TableGridConfig,
    intrinsics: Any,
    pitch_px: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    if len([h for h in holes if h.status != "rejected"]) < cfg.min_holes_for_pose:
        return None, None, float("inf")
    return _solve_pnp_for_holes(holes, cfg, intrinsics)


def detect_table_grid(
    color_bgr: np.ndarray,
    cfg: TableGridConfig,
    intrinsics: Any,
    *,
    origin_click_px: tuple[float, float] | None = None,
    origin_grid_cell: tuple[int, int] | None = None,
    origin_lock_px: tuple[float, float] | None = None,
) -> GridObservation:
    """
    Simple pipeline: Canny edges -> circular contours -> fit 1″ grid -> overlay.
    """
    cv2 = _require_cv2()
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    annotated = color_bgr.copy()
    h_img, w_img = gray.shape[:2]

    candidates, edges = find_hole_candidates(gray, cfg)
    n_raw = len(candidates)
    detected = _candidates_to_holes(candidates)

    pitch_guess = estimate_pitch_px(np.array([c.center_px for c in candidates], dtype=np.float64))
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if edges is not None else None

    if len(detected) < 3:
        msg = f"{n_raw} edge candidates (need 3+ to fit grid)"
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        for hole in detected:
            cx, cy = hole.center_px
            r = max(3, int(round(hole.radius_px)))
            cv2.circle(annotated, (int(round(cx)), int(round(cy))), r, (0, 255, 255), 2)
        return _observation(
            found=False,
            holes=detected,
            outliers=[],
            slots=_slots_from_holes(detected),
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_guess,
            origin=None,
            origin_model_cell=None,
            origin_lock_px=None,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    robust = fit_robust_grid_model(detected, pitch_guess, cfg)
    if robust is None:
        msg = f"{n_raw} candidates — grid fit failed"
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        for hole in detected:
            cx, cy = hole.center_px
            r = max(3, int(round(hole.radius_px)))
            cv2.circle(annotated, (int(round(cx)), int(round(cy))), r, (0, 255, 255), 2)
        return _observation(
            found=False,
            holes=detected,
            outliers=[],
            slots=_slots_from_holes(detected),
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_guess,
            origin=None,
            origin_model_cell=None,
            origin_lock_px=None,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    model, inliers, outliers, _ = robust
    pitch_px = model.pitch_px
    remap_holes_to_model(inliers, model)

    origin, origin_model_cell, new_lock_px = _establish_origin(
        inliers,
        image_center_px=(w_img * 0.5, h_img * 0.5),
        origin_click_px=origin_click_px,
        origin_lock_px=origin_lock_px,
        origin_grid_cell=origin_grid_cell,
        mean_radius_px=model.mean_radius_px,
        model=model,
    )
    if origin is not None and origin_model_cell is not None:
        model = anchor_model_to_origin(model, origin_model_cell)
        remap_holes_to_model(inliers, model)
    resolved_lock_px = origin_lock_px or new_lock_px
    slots = _slots_from_holes(inliers)

    draw_simple_overlay(annotated, model, inliers, outliers, origin, w_img, h_img, cfg=cfg)

    if len(inliers) < cfg.min_holes_for_pose:
        msg = f"{n_raw} raw -> {len(inliers)} on grid ({len(outliers)} rejected)"
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return _observation(
            found=False,
            holes=inliers,
            outliers=outliers,
            slots=slots,
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_px,
            origin=origin,
            origin_model_cell=origin_model_cell,
            origin_lock_px=resolved_lock_px,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    objp = _object_points_for_holes(inliers, cfg.pitch_m)
    cam_mat = camera_matrix_from_intrinsics(intrinsics)
    dist = distortion_from_intrinsics(intrinsics)

    rvec, tvec, err = _solve_pnp_for_holes(inliers, cfg, intrinsics)
    if rvec is None:
        msg = "solvePnP failed"
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return _observation(
            found=False,
            holes=inliers,
            outliers=outliers,
            slots=slots,
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_px,
            origin=origin,
            origin_model_cell=origin_model_cell,
            origin_lock_px=resolved_lock_px,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    projected, _ = cv2.projectPoints(objp, rvec, tvec, cam_mat, dist)
    for pt in projected.reshape(-1, 2):
        cv2.circle(annotated, (int(round(pt[0])), int(round(pt[1]))), 2, (255, 0, 255), -1)
    msg = f"{len(inliers)} holes on grid, pitch {pitch_px:.1f}px, reproj {err:.2f}px"
    cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return _observation(
        found=True,
        holes=inliers,
        outliers=outliers,
        slots=slots,
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
        err=err,
        pitch_px=pitch_px,
        origin=origin,
        origin_model_cell=origin_model_cell,
        origin_lock_px=resolved_lock_px,
        annotated=annotated,
        msg=msg,
        edges_bgr=edges_bgr,
        n_raw=n_raw,
    )


def detect_table_grid_preview(
    color_bgr: np.ndarray,
    cfg: TableGridConfig,
) -> GridObservation:
    """Live preview before origin lock: raw edge detections only."""
    cv2 = _require_cv2()
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    annotated = color_bgr.copy()
    candidates, edges = find_hole_candidates(gray, cfg)
    holes = _candidates_to_holes(candidates)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if edges is not None else None
    for hole in holes:
        cx, cy = hole.center_px
        r = max(3, int(round(hole.radius_px)))
        cv2.circle(annotated, (int(round(cx)), int(round(cy))), r, (0, 255, 255), 2)
    msg = f"{len(holes)} holes detected — click origin, then pick 4 neighbors"
    cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return _observation(
        found=False,
        holes=holes,
        outliers=[],
        slots=_slots_from_holes(holes),
        rvec=None,
        tvec=None,
        err=float("inf"),
        pitch_px=estimate_pitch_px(np.array([c.center_px for c in candidates], dtype=np.float64)),
        origin=None,
        origin_model_cell=None,
        origin_lock_px=None,
        annotated=annotated,
        msg=msg,
        edges_bgr=edges_bgr,
        n_raw=len(candidates),
    )


def detect_table_grid_grow(
    color_bgr: np.ndarray,
    cfg: TableGridConfig,
    intrinsics: Any,
    origin_click_px: tuple[float, float],
    *,
    manual_seed_clicks: list[tuple[float, float]] | None = None,
) -> GridObservation:
    """Frozen-frame grid: grow from origin via 4-way seeds + BFS path extrapolation."""
    cv2 = _require_cv2()
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    annotated = color_bgr.copy()
    h_img, w_img = gray.shape[:2]

    candidates, edges = find_hole_candidates(gray, cfg)
    n_raw = len(candidates)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if edges is not None else None

    if n_raw < 4:
        msg = f"{n_raw} candidates — need 4+ to grow grid"
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return _observation(
            found=False,
            holes=[],
            outliers=[],
            slots=[],
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=float("nan"),
            origin=None,
            origin_model_cell=None,
            origin_lock_px=origin_click_px,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    try:
        known, rejected, origin, pitch_px, n_seeds = grow_grid_from_origin(
            candidates,
            origin_click_px,
            cfg,
            width=w_img,
            height=h_img,
            manual_seed_clicks=manual_seed_clicks,
        )
    except ValueError as exc:
        msg = str(exc)
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return _observation(
            found=False,
            holes=[],
            outliers=[],
            slots=[],
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=float("nan"),
            origin=None,
            origin_model_cell=None,
            origin_lock_px=origin_click_px,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    inliers = list(known.values())
    draw_grown_grid(annotated, known, rejected, origin, pitch_px)

    if len(inliers) < cfg.min_holes_for_pose:
        msg = f"grown {len(inliers)} holes ({n_seeds} seeds), need {cfg.min_holes_for_pose}+"
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return _observation(
            found=False,
            holes=inliers,
            outliers=rejected,
            slots=_slots_from_holes(inliers),
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_px,
            origin=origin,
            origin_model_cell=(0, 0),
            origin_lock_px=origin.center_px,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    if n_seeds < 2:
        msg = f"grown {len(inliers)} holes but only {n_seeds}/4 direction seeds — click a clearer origin"
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        return _observation(
            found=False,
            holes=inliers,
            outliers=rejected,
            slots=_slots_from_holes(inliers),
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_px,
            origin=origin,
            origin_model_cell=(0, 0),
            origin_lock_px=origin.center_px,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    rvec, tvec, err = _solve_pnp_for_holes(inliers, cfg, intrinsics)
    if rvec is None:
        msg = "solvePnP failed on grown grid"
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return _observation(
            found=False,
            holes=inliers,
            outliers=rejected,
            slots=_slots_from_holes(inliers),
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_px,
            origin=origin,
            origin_model_cell=(0, 0),
            origin_lock_px=origin.center_px,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    objp = _object_points_for_holes(inliers, cfg.pitch_m)
    cam_mat = camera_matrix_from_intrinsics(intrinsics)
    dist = distortion_from_intrinsics(intrinsics)
    projected, _ = cv2.projectPoints(objp, rvec, tvec, cam_mat, dist)
    for pt in projected.reshape(-1, 2):
        cv2.circle(annotated, (int(round(pt[0])), int(round(pt[1]))), 2, (255, 0, 255), -1)
    msg = f"grown {len(inliers)} holes ({n_seeds} seeds), pitch {pitch_px:.1f}px, reproj {err:.2f}px"
    cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    obs = _observation(
        found=True,
        holes=inliers,
        outliers=rejected,
        slots=_slots_from_holes(inliers),
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
        err=err,
        pitch_px=pitch_px,
        origin=origin,
        origin_model_cell=(0, 0),
        origin_lock_px=origin.center_px,
        annotated=annotated,
        msg=msg,
        edges_bgr=edges_bgr,
        n_raw=n_raw,
    )
    obs.all_grown_holes = list(inliers)
    obs.detection_rejected = list(rejected)
    return obs


def apply_region_polygon(
    color_bgr: np.ndarray,
    cfg: TableGridConfig,
    intrinsics: Any,
    *,
    all_grown_holes: list[DetectedHole],
    grow_rejected: list[DetectedHole],
    pitch_px: float,
    polygon: list[tuple[float, float]],
    edges_bgr,
    n_raw: int,
) -> GridObservation:
    """Keep only grown grid holes inside polygon; pose solved after final origin click."""
    cv2 = _require_cv2()
    annotated = color_bgr.copy()

    inside, outside = partition_holes_by_polygon(all_grown_holes, polygon)
    draw_region_filtered_grid(
        annotated,
        inside,
        outside,
        inside[0] if inside else DetectedHole(center_px=(0.0, 0.0), radius_px=5.0, circularity=0.0),
        polygon,
        pitch_px,
        detection_rejected=grow_rejected,
        show_origin_marker=False,
    )

    if len(inside) < cfg.min_holes_for_pose:
        msg = f"region has {len(inside)} holes, need {cfg.min_holes_for_pose}+ — draw larger shape"
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return _observation(
            found=False,
            holes=inside,
            outliers=outside + grow_rejected,
            slots=_slots_from_holes(inside),
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_px,
            origin=None,
            origin_model_cell=None,
            origin_lock_px=None,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    msg = f"region {len(inside)}/{len(all_grown_holes)} holes — click final origin"
    cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    obs = _observation(
        found=False,
        holes=inside,
        outliers=outside + grow_rejected,
        slots=_slots_from_holes(inside),
        rvec=None,
        tvec=None,
        err=float("inf"),
        pitch_px=pitch_px,
        origin=None,
        origin_model_cell=None,
        origin_lock_px=None,
        annotated=annotated,
        msg=msg,
        edges_bgr=edges_bgr,
        n_raw=n_raw,
    )
    obs.all_grown_holes = list(all_grown_holes)
    obs.detection_rejected = list(grow_rejected)
    return obs


def set_calibration_origin(
    color_bgr: np.ndarray,
    cfg: TableGridConfig,
    intrinsics: Any,
    *,
    region_holes: list[DetectedHole],
    click_px: tuple[float, float],
    pitch_px: float,
    polygon: list[tuple[float, float]] | None,
    grow_rejected: list[DetectedHole],
    all_grown_holes: list[DetectedHole] | None,
    outside_holes: list[DetectedHole],
    edges_bgr,
    n_raw: int,
) -> GridObservation:
    """Re-index grid to clicked hole as (0,0) and solve calibration pose."""
    from dataclasses import replace

    cv2 = _require_cv2()
    chosen = snap_hole_from_click(region_holes, click_px)
    if chosen is None:
        msg = "click closer to a hole in the region"
        annotated = color_bgr.copy()
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return _observation(
            found=False,
            holes=region_holes,
            outliers=outside_holes + grow_rejected,
            slots=_slots_from_holes(region_holes),
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_px,
            origin=None,
            origin_model_cell=None,
            origin_lock_px=None,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    oc, orow = chosen.grid_col, chosen.grid_row
    holes = [replace(h) for h in region_holes]
    origin = next(
        h for h in holes if h.grid_col == oc and h.grid_row == orow
    )
    _reindex_relative_to_origin(holes, origin)
    origin.is_origin = True
    origin.status = "locked"

    if len(holes) < cfg.min_holes_for_pose:
        msg = f"only {len(holes)} holes — need {cfg.min_holes_for_pose}+"
        annotated = color_bgr.copy()
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return _observation(
            found=False,
            holes=holes,
            outliers=outside_holes + grow_rejected,
            slots=_slots_from_holes(holes),
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_px,
            origin=origin,
            origin_model_cell=(0, 0),
            origin_lock_px=origin.center_px,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    rvec, tvec, err = _solve_pnp_for_holes(holes, cfg, intrinsics)
    annotated = color_bgr.copy()
    draw_region_filtered_grid(
        annotated,
        holes,
        outside_holes,
        origin,
        polygon or [],
        pitch_px,
        detection_rejected=grow_rejected,
        show_origin_marker=True,
    )

    if rvec is None:
        msg = "solvePnP failed after origin pick"
        cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return _observation(
            found=False,
            holes=holes,
            outliers=outside_holes + grow_rejected,
            slots=_slots_from_holes(holes),
            rvec=None,
            tvec=None,
            err=float("inf"),
            pitch_px=pitch_px,
            origin=origin,
            origin_model_cell=(0, 0),
            origin_lock_px=origin.center_px,
            annotated=annotated,
            msg=msg,
            edges_bgr=edges_bgr,
            n_raw=n_raw,
        )

    objp = _object_points_for_holes(holes, cfg.pitch_m)
    cam_mat = camera_matrix_from_intrinsics(intrinsics)
    dist = distortion_from_intrinsics(intrinsics)
    projected, _ = cv2.projectPoints(objp, rvec, tvec, cam_mat, dist)
    for pt in projected.reshape(-1, 2):
        cv2.circle(annotated, (int(round(pt[0])), int(round(pt[1]))), 2, (255, 0, 255), -1)

    msg = (
        f"origin ({origin.grid_col},{origin.grid_row})  {len(holes)} holes, "
        f"reproj {err:.2f}px"
    )
    cv2.putText(annotated, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    obs = _observation(
        found=True,
        holes=holes,
        outliers=outside_holes + grow_rejected,
        slots=_slots_from_holes(holes),
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
        err=err,
        pitch_px=pitch_px,
        origin=origin,
        origin_model_cell=(0, 0),
        origin_lock_px=origin.center_px,
        annotated=annotated,
        msg=msg,
        edges_bgr=edges_bgr,
        n_raw=n_raw,
    )
    obs.all_grown_holes = list(all_grown_holes or region_holes)
    obs.detection_rejected = list(grow_rejected)
    return obs
