from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid_config import TableGridConfig
from .grid_types import DetectedHole, GridSlot
from .hole_detect import RawHoleCandidate, filter_raw_candidates


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for table calibration. "
            "Install with: pip install -r requirements-vision.txt"
        ) from exc
    return cv2


@dataclass
class GridModel:
    origin_px: np.ndarray
    pitch_px: float
    axis_u: np.ndarray
    axis_v: np.ndarray
    mean_radius_px: float
    homography: np.ndarray | None = None  # maps (col, row, 1) -> image pixels

    def cell_to_pixel(self, col: int, row: int) -> np.ndarray:
        if self.homography is not None:
            w = self.homography @ np.array([float(col), float(row), 1.0], dtype=np.float64)
            if abs(w[2]) < 1e-9:
                return self.origin_px.copy()
            return (w[:2] / w[2]).astype(np.float64)
        return (
            self.origin_px
            + col * self.pitch_px * self.axis_u
            + row * self.pitch_px * self.axis_v
        )

    def pixel_to_cell(self, px: float, py: float) -> tuple[int, int]:
        if self.homography is not None:
            h_inv = np.linalg.inv(self.homography)
            w = h_inv @ np.array([px, py, 1.0], dtype=np.float64)
            if abs(w[2]) < 1e-9:
                return 0, 0
            return int(round(w[0] / w[2])), int(round(w[1] / w[2]))
        delta = np.array([px, py], dtype=np.float64) - self.origin_px
        u = float(np.dot(delta, self.axis_u))
        v = float(np.dot(delta, self.axis_v))
        return int(round(u / self.pitch_px)), int(round(v / self.pitch_px))


def local_pitch_px(model: GridModel, col: int, row: int, delta: float = 0.5) -> float:
    """Approximate local grid spacing in pixels (varies under perspective)."""
    p0 = model.cell_to_pixel(col, row)
    pu = model.cell_to_pixel(col + 1, row)
    pv = model.cell_to_pixel(col, row + 1)
    return float(0.5 * (np.linalg.norm(pu - p0) + np.linalg.norm(pv - p0)))


def fit_homography_to_holes(holes: list[DetectedHole]) -> np.ndarray | None:
    """Fit projective grid: integer (col, row) -> image pixel."""
    cv2 = _require_cv2()
    if len(holes) < 4:
        return None
    src = np.array([[h.grid_col, h.grid_row] for h in holes], dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array([h.center_px for h in holes], dtype=np.float32).reshape(-1, 1, 2)
    h_mat, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
    if h_mat is None:
        return None
    return h_mat.astype(np.float64)


def with_homography(model: GridModel, holes: list[DetectedHole]) -> GridModel:
    h_mat = fit_homography_to_holes(holes)
    if h_mat is None:
        return model
    return GridModel(
        origin_px=model.origin_px,
        pitch_px=model.pitch_px,
        axis_u=model.axis_u,
        axis_v=model.axis_v,
        mean_radius_px=model.mean_radius_px,
        homography=h_mat,
    )


def anchor_model_to_origin(model: GridModel, origin_cell: tuple[int, int]) -> GridModel:
    """Re-express the fitted grid so origin_cell becomes index (0, 0)."""
    oc, orow = origin_cell
    if oc == 0 and orow == 0:
        return model

    if model.homography is not None:
        shift = np.array(
            [[1.0, 0.0, float(oc)], [0.0, 1.0, float(orow)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        h_new = model.homography @ shift
        w = h_new @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        origin_px = (w[:2] / w[2]).astype(np.float64) if abs(w[2]) > 1e-9 else model.origin_px.copy()
        return GridModel(
            origin_px=origin_px,
            pitch_px=model.pitch_px,
            axis_u=model.axis_u,
            axis_v=model.axis_v,
            mean_radius_px=model.mean_radius_px,
            homography=h_new,
        )

    origin_px = model.cell_to_pixel(oc, orow)
    return GridModel(
        origin_px=origin_px,
        pitch_px=model.pitch_px,
        axis_u=model.axis_u,
        axis_v=model.axis_v,
        mean_radius_px=model.mean_radius_px,
        homography=None,
    )


def remap_holes_to_model(holes: list[DetectedHole], model: GridModel) -> None:
    """Assign grid indices from the fitted lattice (stable across frames)."""
    for hole in holes:
        col, row = model.pixel_to_cell(hole.center_px[0], hole.center_px[1])
        hole.grid_col = col
        hole.grid_row = row
        expected = model.cell_to_pixel(col, row)
        hole.grid_residual_px = float(np.linalg.norm(np.array(hole.center_px) - expected))


def _grid_axes_from_centers(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(centers) < 3:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    centered = centers - centers.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0] / (np.linalg.norm(vh[0]) + 1e-12)
    axis_v = vh[1] / (np.linalg.norm(vh[1]) + 1e-12)
    return axis_u, axis_v


def assign_grid_indices(
    holes: list[DetectedHole],
    pitch_px: float,
    cfg: TableGridConfig,
) -> None:
    """Seed col/row labels before least-squares grid fit."""
    if len(holes) < 2 or not np.isfinite(pitch_px) or pitch_px < cfg.min_pitch_px:
        return
    centers = np.array([h.center_px for h in holes], dtype=np.float64)
    axis_u, axis_v = _grid_axes_from_centers(centers)
    origin = centers.mean(axis=0)
    for hole in holes:
        delta = np.array(hole.center_px) - origin
        u = float(np.dot(delta, axis_u))
        v = float(np.dot(delta, axis_v))
        hole.grid_col = int(round(u / pitch_px))
        hole.grid_row = int(round(v / pitch_px))


def grid_inlier_threshold_px(model: GridModel, cfg: TableGridConfig, residuals: np.ndarray) -> float:
    """Max distance to fitted grid cell center (perspective-aware when homography is set)."""
    pitch = model.pitch_px
    fixed = cfg.grid_snap_max_frac * pitch
    floor = 0.35 * pitch
    if len(residuals) == 0:
        return max(fixed, floor)
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    if mad < 1e-6:
        return max(fixed, floor)
    adaptive = med + 4.0 * 1.4826 * mad
    return float(max(floor, fixed, adaptive))


def _median_detected_radius(holes: list[DetectedHole], model: GridModel) -> float:
    detected = [h for h in holes if h.status in ("detected", "locked")]
    if detected:
        return float(np.median([h.radius_px for h in detected]))
    if holes:
        return float(np.median([h.radius_px for h in holes]))
    return model.mean_radius_px


def holes_for_model_fit(holes: list[DetectedHole], cfg: TableGridConfig) -> list[DetectedHole]:
    """Use direct edge detections for lattice fitting."""
    pool = [h for h in holes if h.status not in ("rejected", "recovered")]
    if len(pool) >= cfg.min_holes_for_pose:
        return pool
    return [h for h in holes if h.status != "rejected"]


def _hole_inlier_sort_key(
    hole: DetectedHole,
    model: GridModel,
    median_radius_px: float,
) -> tuple:
    """Lower sort key = better inlier. Prefer direct detections over grid-snapped recoveries."""
    if hole.status == "locked":
        return (0, 0.0, 0.0, 0.0)
    status_pri = 0 if hole.status == "detected" else (1 if hole.status == "recovered" else 2)
    pitch = max(model.pitch_px, 1.0)
    res = hole.grid_residual_px / (0.28 * pitch)
    qual = 1.0 - float(np.clip(hole.circularity, 0.0, 1.0))
    return (status_pri, res + 0.35 * qual, hole.grid_residual_px, -hole.circularity)


def _effective_inlier_threshold(
    hole: DetectedHole,
    base_threshold: float,
    model: GridModel,
) -> float:
    """Allow slack proportional to local pitch (smaller/farther under perspective)."""
    local = local_pitch_px(model, hole.grid_col, hole.grid_row)
    if hole.status == "detected":
        return max(base_threshold, 0.55 * local, 8.0)
    if hole.status == "recovered":
        return max(0.35 * local, 6.0)
    return max(base_threshold, 0.45 * local)


def partition_grid_inliers(
    holes: list[DetectedHole],
    model: GridModel,
    cfg: TableGridConfig,
) -> tuple[list[DetectedHole], list[DetectedHole], float]:
    """
    Snap holes to the fitted grid; keep best detection per cell (not just lowest residual).
    Returns (inliers, outliers, threshold_px).
    """
    if not holes:
        return [], [], cfg.grid_snap_max_frac * model.pitch_px

    remap_holes_to_model(holes, model)
    median_r = _median_detected_radius(holes, model)
    detected_residuals = np.array(
        [h.grid_residual_px for h in holes if h.status == "detected"],
        dtype=np.float64,
    )
    all_residuals = np.array([h.grid_residual_px for h in holes], dtype=np.float64)
    threshold = grid_inlier_threshold_px(
        model,
        cfg,
        detected_residuals if len(detected_residuals) >= 4 else all_residuals,
    )

    inlier_map: dict[tuple[int, int], DetectedHole] = {}
    outliers: list[DetectedHole] = []

    ranked = sorted(
        holes,
        key=lambda h: _hole_inlier_sort_key(h, model, median_r),
    )
    for hole in ranked:
        if hole.status == "locked":
            inlier_map[(hole.grid_col, hole.grid_row)] = hole
            continue
        limit = _effective_inlier_threshold(hole, threshold, model)
        if hole.grid_residual_px > limit:
            hole.status = "rejected"
            outliers.append(hole)
            continue
        key = (hole.grid_col, hole.grid_row)
        if key in inlier_map:
            prev = inlier_map[key]
            prev.status = "rejected"
            outliers.append(prev)
        inlier_map[key] = hole

    inliers = list(inlier_map.values())
    return inliers, outliers, threshold


def fit_robust_grid_model(
    holes: list[DetectedHole],
    pitch_px: float,
    cfg: TableGridConfig,
    *,
    max_iterations: int = 3,
) -> tuple[GridModel, list[DetectedHole], list[DetectedHole], float] | None:
    """
    Iteratively fit a grid and drop points far from the lattice.
    Returns (model, inliers, outliers, threshold_px) or None.
    """
    if len(holes) < 3 or not np.isfinite(pitch_px) or pitch_px < 1.0:
        return None

    original = list(holes)
    working = holes_for_model_fit(holes, cfg)
    model: GridModel | None = None

    for _ in range(max_iterations):
        assign_grid_indices(working, pitch_px, cfg)
        model = fit_grid_model(working, pitch_px)
        if model is None:
            return None
        pitch_px = model.pitch_px
        inliers, _, _ = partition_grid_inliers(working, model, cfg)
        fit_pool = holes_for_model_fit(inliers, cfg)
        if len(fit_pool) >= len(working) or len(fit_pool) < 3:
            break
        working = fit_pool

    assert model is not None
    fit_pool = holes_for_model_fit(working, cfg)
    model = with_homography(model, fit_pool)
    remap_holes_to_model(original, model)
    model = with_homography(model, original)
    remap_holes_to_model(original, model)
    inliers, outliers, threshold = partition_grid_inliers(original, model, cfg)
    return model, inliers, outliers, threshold


def refit_and_filter_holes(
    holes: list[DetectedHole],
    model: GridModel,
    cfg: TableGridConfig,
    *,
    max_iterations: int = 2,
) -> tuple[GridModel, list[DetectedHole], list[DetectedHole], float]:
    """Re-fit lattice on extended holes and reject off-grid points (keeps model indexing)."""
    original = list(holes)
    working = holes_for_model_fit(holes, cfg)
    current = model

    for _ in range(max_iterations):
        remap_holes_to_model(working, current)
        inliers, _, _ = partition_grid_inliers(working, current, cfg)
        fit_pool = holes_for_model_fit(inliers, cfg)
        if len(fit_pool) >= len(working) or len(fit_pool) < cfg.min_holes_for_pose:
            break
        working = fit_pool
        updated = fit_grid_model(working, current.pitch_px)
        if updated is not None:
            current = updated

    inliers, outliers, threshold = partition_grid_inliers(original, current, cfg)
    return current, inliers, outliers, threshold


def fit_grid_model(holes: list[DetectedHole], pitch_px: float) -> GridModel | None:
    if len(holes) < 3 or not np.isfinite(pitch_px) or pitch_px < 1.0:
        return None
    centers = np.array([h.center_px for h in holes], dtype=np.float64)
    cols = np.array([h.grid_col for h in holes], dtype=np.float64)
    rows = np.array([h.grid_row for h in holes], dtype=np.float64)

    centered = centers - centers.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0] / (np.linalg.norm(vh[0]) + 1e-12)
    axis_v = vh[1] / (np.linalg.norm(vh[1]) + 1e-12)

    design = np.zeros((len(holes) * 2, 4), dtype=np.float64)
    target = np.zeros(len(holes) * 2, dtype=np.float64)
    for i, hole in enumerate(holes):
        cu = cols[i] * pitch_px
        rv = rows[i] * pitch_px
        design[2 * i, 0] = 1.0
        design[2 * i, 1] = cu * axis_u[0] + rv * axis_v[0]
        design[2 * i + 1, 2] = 1.0
        design[2 * i + 1, 3] = cu * axis_u[1] + rv * axis_v[1]
        target[2 * i] = hole.center_px[0]
        target[2 * i + 1] = hole.center_px[1]

    sol, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    origin_px = np.array([sol[0], sol[2]], dtype=np.float64)
    mean_r = float(np.mean([h.radius_px for h in holes]))
    return GridModel(
        origin_px=origin_px,
        pitch_px=float(pitch_px),
        axis_u=axis_u,
        axis_v=axis_v,
        mean_radius_px=mean_r,
    )


def _image_cell_bounds(model: GridModel, width: int, height: int, margin_cells: int) -> tuple[int, int, int, int]:
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    cols: list[float] = []
    rows: list[float] = []
    for px, py in corners:
        c, r = model.pixel_to_cell(px, py)
        cols.append(c)
        rows.append(r)
    c0 = int(np.floor(min(cols))) - margin_cells
    c1 = int(np.ceil(max(cols))) + margin_cells
    r0 = int(np.floor(min(rows))) - margin_cells
    r1 = int(np.ceil(max(rows))) + margin_cells
    return c0, c1, r0, r1


def image_cell_bounds(
    model: GridModel,
    width: int,
    height: int,
    holes: list[DetectedHole],
    *,
    margin_cells: int,
) -> tuple[int, int, int, int]:
    if holes:
        cols = [h.grid_col for h in holes]
        rows = [h.grid_row for h in holes]
        c0 = int(min(cols)) - margin_cells
        c1 = int(max(cols)) + margin_cells
        r0 = int(min(rows)) - margin_cells
        r1 = int(max(rows)) + margin_cells
    else:
        corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
        cols = []
        rows = []
        for px, py in corners:
            c, r = model.pixel_to_cell(px, py)
            cols.append(c)
            rows.append(r)
        c0 = int(np.floor(min(cols))) - margin_cells
        c1 = int(np.ceil(max(cols))) + margin_cells
        r0 = int(np.floor(min(rows))) - margin_cells
        r1 = int(np.ceil(max(rows))) + margin_cells
    return c0, c1, r0, r1


def local_hole_search(
    gray: np.ndarray,
    preprocessed: np.ndarray,
    edges: np.ndarray | None,
    expected_px: tuple[float, float],
    radius_guess: float,
    pitch_px: float,
    cfg: TableGridConfig,
    *,
    search_frac: float = 0.38,
) -> DetectedHole | None:
    from .hole_classifier import compute_ring_score

    cv2 = _require_cv2()
    ex, ey = expected_px
    h, w = gray.shape[:2]
    span = max(8.0, radius_guess * 2.8, cfg.expected_radius_px * 1.5)
    search_r = max(4.0, pitch_px * search_frac)

    x0 = max(0, int(round(ex - span)))
    y0 = max(0, int(round(ey - span)))
    x1 = min(w, int(round(ex + span)))
    y1 = min(h, int(round(ey + span)))
    roi = preprocessed[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best: DetectedHole | None = None
    best_score = -1.0
    lex, ley = ex - x0, ey - y0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < cfg.min_area_px * 0.5 or area > cfg.max_area_px:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < cfg.min_circularity * 0.85:
            continue
        if len(contour) >= 5:
            (cx, cy), (ma, mi), _ = cv2.fitEllipse(contour)
            radius = 0.25 * (ma + mi)
        else:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
        dist = (cx - lex) ** 2 + (cy - ley) ** 2
        if dist > search_r * search_r:
            continue
        gx, gy = float(cx + x0), float(cy + y0)
        ring = compute_ring_score(gray, edges, gx, gy, radius)
        score = 0.5 * circularity + 0.5 * ring - 0.0005 * dist
        if score > best_score and ring >= cfg.ring_confidence * 0.65:
            best_score = score
            best = DetectedHole(
                center_px=(gx, gy),
                radius_px=float(radius),
                circularity=float(circularity),
                nn_score=float(score),
                source="local_search",
            )
    return best


def extend_and_fill_grid(
    gray: np.ndarray,
    preprocessed: np.ndarray,
    edges: np.ndarray | None,
    model: GridModel,
    initial_holes: list[DetectedHole],
    loose_candidates: list[RawHoleCandidate],
    cfg: TableGridConfig,
    width: int,
    height: int,
) -> tuple[list[GridSlot], list[DetectedHole]]:
    """Build full lattice; match, recover, or mark missing at each cell."""
    remap_holes_to_model(initial_holes, model)
    loose_candidates = filter_raw_candidates(
        loose_candidates,
        cfg,
        width,
        height,
        model.pitch_px,
    )
    min_r, max_r = cfg.hole_radius_limits(model.pitch_px)
    c0, c1, r0, r1 = image_cell_bounds(model, width, height, initial_holes, margin_cells=cfg.grid_margin_cells)
    snap_px = cfg.grid_snap_max_frac * model.pitch_px

    cell_map: dict[tuple[int, int], DetectedHole] = {}
    for hole in initial_holes:
        key = (hole.grid_col, hole.grid_row)
        if key not in cell_map:
            cell_map[key] = hole
            continue
        prev = cell_map[key]
        if _hole_inlier_sort_key(hole, model, model.mean_radius_px) < _hole_inlier_sort_key(
            prev, model, model.mean_radius_px
        ):
            cell_map[key] = hole

    for cand in loose_candidates:
        col, row = model.pixel_to_cell(cand.center_px[0], cand.center_px[1])
        if col < c0 or col > c1 or row < r0 or row > r1:
            continue
        expected = model.cell_to_pixel(col, row)
        dist = np.linalg.norm(np.array(cand.center_px) - expected)
        if dist > snap_px:
            continue
        det = DetectedHole(
            center_px=cand.center_px,
            radius_px=cand.radius_px,
            circularity=cand.circularity,
            nn_score=getattr(cand, "nn_score", cand.score if hasattr(cand, "score") else 0.5),
            grid_col=col,
            grid_row=row,
            grid_residual_px=float(dist),
            source=getattr(cand, "source", "remap"),
        )
        key = (col, row)
        if key not in cell_map or det.nn_score > cell_map[key].nn_score:
            existing = cell_map.get(key)
            if existing is not None and existing.status == "detected":
                if _hole_inlier_sort_key(det, model, model.mean_radius_px) >= _hole_inlier_sort_key(
                    existing, model, model.mean_radius_px
                ):
                    continue
            cell_map[key] = det

    slots: list[GridSlot] = []
    all_holes: list[DetectedHole] = []

    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            expected = model.cell_to_pixel(col, row)
            ex, ey = float(expected[0]), float(expected[1])
            if ex < -model.pitch_px or ey < -model.pitch_px or ex > width + model.pitch_px or ey > height + model.pitch_px:
                continue

            key = (col, row)
            hole = cell_map.get(key)
            if hole is not None:
                hole.grid_col = col
                hole.grid_row = row
                hole.status = "detected"
                slots.append(GridSlot(col, row, (ex, ey), "detected", hole))
                all_holes.append(hole)
                continue

            if not cfg.local_search:
                slots.append(GridSlot(col, row, (ex, ey), "missing", None))
                continue

            recovered = local_hole_search(
                gray,
                preprocessed,
                edges,
                (ex, ey),
                model.mean_radius_px,
                model.pitch_px,
                cfg,
            )
            if recovered is not None:
                if recovered.radius_px < min_r or recovered.radius_px > max_r:
                    recovered = None
            if recovered is not None:
                recovered.grid_col = col
                recovered.grid_row = row
                recovered.grid_residual_px = float(
                    np.linalg.norm(np.array(recovered.center_px) - expected)
                )
                recovered.status = "recovered"
                slots.append(GridSlot(col, row, (ex, ey), "recovered", recovered))
                all_holes.append(recovered)
            else:
                slots.append(GridSlot(col, row, (ex, ey), "missing", None))

    return slots, all_holes


# OpenCV BGR colors
_COLOR_CONFIRMED = (0, 255, 255)  # yellow — NN-confirmed on grid
_COLOR_DETECTED = (0, 255, 255)   # yellow — directly detected
_COLOR_RECOVERED = (0, 165, 255)  # orange — found by local search at grid cell
_COLOR_ORIGIN = (0, 0, 255)       # red — table origin (0,0)
_COLOR_MISSING = (120, 120, 120)  # gray — expected cell, no hole
_COLOR_REJECTED = (255, 0, 255)     # magenta — off-grid false detection


def mark_rejected_slots(slots: list[GridSlot], inlier_holes: list[DetectedHole]) -> None:
    inlier_ids = {id(h) for h in inlier_holes}
    for slot in slots:
        if slot.hole is None:
            continue
        if id(slot.hole) not in inlier_ids:
            slot.hole.status = "rejected"
            slot.status = "rejected"


def draw_simple_overlay(
    annotated: np.ndarray,
    model: GridModel,
    inliers: list[DetectedHole],
    outliers: list[DetectedHole],
    origin_hole: DetectedHole | None,
    width: int,
    height: int,
    *,
    cfg: TableGridConfig | None = None,
) -> None:
    """Draw edge-detected holes and the fitted 1″ lattice."""
    cv2 = _require_cv2()
    margin = cfg.grid_margin_cells if cfg is not None else 0
    c0, c1, r0, r1 = image_cell_bounds(model, width, height, inliers, margin_cells=margin)
    line_color = (90, 90, 90)

    for col in range(c0, c1 + 1):
        pts: list[tuple[int, int]] = []
        for row in range(r0, r1 + 1):
            p = model.cell_to_pixel(col, row)
            pts.append((int(round(p[0])), int(round(p[1]))))
        for i in range(len(pts) - 1):
            cv2.line(annotated, pts[i], pts[i + 1], line_color, 1)

    for row in range(r0, r1 + 1):
        pts = []
        for col in range(c0, c1 + 1):
            p = model.cell_to_pixel(col, row)
            pts.append((int(round(p[0])), int(round(p[1]))))
        for i in range(len(pts) - 1):
            cv2.line(annotated, pts[i], pts[i + 1], line_color, 1)

    for hole in inliers:
        cx, cy = hole.center_px
        ix, iy = int(round(cx)), int(round(cy))
        r = max(3, int(round(hole.radius_px)))
        color = _COLOR_ORIGIN if hole.is_origin else (0, 255, 0)
        thickness = 3 if hole.is_origin else 2
        cv2.circle(annotated, (ix, iy), r, color, thickness)
        cv2.putText(
            annotated,
            f"{hole.grid_col},{hole.grid_row}",
            (ix + 5, iy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )

    if outliers:
        draw_rejected_holes(annotated, outliers)

    if origin_hole is not None:
        ox, oy = origin_hole.center_px
        cv2.drawMarker(
            annotated,
            (int(round(ox)), int(round(oy))),
            _COLOR_ORIGIN,
            cv2.MARKER_CROSS,
            22,
            2,
        )

    h = annotated.shape[0]
    cv2.putText(
        annotated,
        f"{len(inliers)} on-grid  {len(outliers)} rejected  pitch {model.pitch_px:.1f}px",
        (12, h - 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 220, 255),
        1,
    )
    cv2.putText(
        annotated,
        "green=detected hole  gray lines=grid (perspective)  magenta X=duplicate/outlier  red cross=origin",
        (12, h - 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 200, 180),
        1,
    )


def draw_rejected_holes(annotated: np.ndarray, outliers: list[DetectedHole]) -> None:
    cv2 = _require_cv2()
    for hole in outliers:
        cx, cy = hole.center_px
        r = max(4, int(round(hole.radius_px)))
        ix, iy = int(round(cx)), int(round(cy))
        cv2.circle(annotated, (ix, iy), r, _COLOR_REJECTED, 1)
        cv2.line(annotated, (ix - r, iy - r), (ix + r, iy + r), _COLOR_REJECTED, 1)
        cv2.line(annotated, (ix - r, iy + r), (ix + r, iy - r), _COLOR_REJECTED, 1)


def draw_extended_grid(
    annotated: np.ndarray,
    slots: list[GridSlot],
    origin_hole: DetectedHole | None,
    *,
    rejected_holes: list[DetectedHole] | None = None,
) -> None:
    cv2 = _require_cv2()
    n_det = sum(1 for s in slots if s.status in ("detected", "confirmed"))
    n_rec = sum(1 for s in slots if s.status == "recovered")
    n_conf = sum(1 for s in slots if s.status == "confirmed")
    n_miss = sum(1 for s in slots if s.status == "missing")
    n_rej = sum(1 for s in slots if s.status == "rejected")

    for slot in slots:
        ex, ey = slot.expected_px
        ix, iy = int(round(ex)), int(round(ey))
        if slot.status == "rejected" and slot.hole is not None:
            hole = slot.hole
            cx, cy = hole.center_px
            r = max(4, int(round(hole.radius_px)))
            cv2.circle(annotated, (int(round(cx)), int(round(cy))), r, _COLOR_REJECTED, 1)
            cv2.line(annotated, (int(round(cx)) - r, int(round(cy)) - r), (int(round(cx)) + r, int(round(cy)) + r), _COLOR_REJECTED, 1)
            cv2.line(annotated, (int(round(cx)) - r, int(round(cy)) + r), (int(round(cx)) + r, int(round(cy)) - r), _COLOR_REJECTED, 1)
            continue
        if slot.status == "missing":
            cv2.circle(annotated, (ix, iy), 6, _COLOR_MISSING, 1)
            cv2.line(annotated, (ix - 4, iy - 4), (ix + 4, iy + 4), (80, 80, 80), 1)
            cv2.line(annotated, (ix - 4, iy + 4), (ix + 4, iy - 4), (80, 80, 80), 1)
            label_color = (140, 140, 140)
        elif slot.hole is not None:
            hole = slot.hole
            if hole.status == "rejected":
                continue
            cx, cy = hole.center_px
            if hole.is_origin or hole.status == "locked":
                color = _COLOR_ORIGIN
            elif slot.status == "recovered":
                color = _COLOR_RECOVERED
            else:
                color = _COLOR_CONFIRMED
            thickness = 3 if hole.is_origin else 2
            draw_r = max(3, int(round(hole.radius_px)))
            cv2.circle(annotated, (int(round(cx)), int(round(cy))), draw_r, color, thickness)
            label_color = color
        else:
            continue

        label = f"{slot.grid_col},{slot.grid_row}"
        cv2.putText(
            annotated,
            label,
            (ix + 5, iy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            label_color,
            1,
        )

    if origin_hole is not None:
        ox, oy = origin_hole.center_px
        cv2.drawMarker(
            annotated,
            (int(round(ox)), int(round(oy))),
            _COLOR_ORIGIN,
            cv2.MARKER_CROSS,
            20,
            2,
        )

    if rejected_holes:
        draw_rejected_holes(annotated, rejected_holes)

    h = annotated.shape[0]
    cv2.putText(
        annotated,
        f"grid {len(slots)} cells: {n_conf} confirmed + {n_det - n_conf} other, {n_miss} empty",
        (12, h - 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 220, 255),
        1,
    )
    cv2.putText(
        annotated,
        "yellow=NN-confirmed  gray X=empty cell  red cross=origin",
        (12, h - 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 200, 180),
        1,
    )
