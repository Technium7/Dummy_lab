"""Walk a fitted hole grid and confirm each cell with local search + NN."""

from __future__ import annotations

import numpy as np

from .grid_config import TableGridConfig
from .grid_extend import GridModel, image_cell_bounds
from .grid_types import DetectedHole, GridSlot
from .hole_detect import preprocess_gray


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for table calibration. "
            "Install with: pip install -r software_win/requirements.txt"
        ) from exc
    return cv2


def _contour_candidates_near(
    gray: np.ndarray,
    preprocessed: np.ndarray,
    expected_px: tuple[float, float],
    radius_guess: float,
    pitch_px: float,
    cfg: TableGridConfig,
    *,
    search_frac: float = 0.38,
) -> list[tuple[float, float, float, float, float]]:
    """Return (cx, cy, radius, circularity, dist_to_expected) for local contours."""
    cv2 = _require_cv2()
    ex, ey = expected_px
    h, w = gray.shape[:2]
    span = max(8.0, radius_guess * 2.8, cfg.expected_radius_px * 1.2, 12.0)
    search_r = max(4.0, pitch_px * search_frac)

    x0 = max(0, int(round(ex - span)))
    y0 = max(0, int(round(ey - span)))
    x1 = min(w, int(round(ex + span)))
    y1 = min(h, int(round(ey + span)))
    roi = preprocessed[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area = cfg.resolved_max_area_px()
    out: list[tuple[float, float, float, float, float]] = []
    lex, ley = ex - x0, ey - y0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < cfg.min_area_px * 0.5 or area > cfg.resolved_max_area_px():
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
        if radius < 1.0:
            continue
        if cfg.max_radius_px is not None and radius > cfg.max_radius_px:
            continue
        dist = float(np.hypot(cx - lex, cy - ley))
        if dist > search_r:
            continue
        gx, gy = float(cx + x0), float(cy + y0)
        out.append((gx, gy, float(radius), float(circularity), dist))

    return out


def nn_confirm_at_cell(
    gray: np.ndarray,
    preprocessed: np.ndarray,
    edges: np.ndarray | None,
    expected_px: tuple[float, float],
    radius_guess: float,
    pitch_px: float,
    cfg: TableGridConfig,
) -> DetectedHole | None:
    """Find and NN-verify a hole near the lattice prediction; return refined center."""
    from .hole_classifier import compute_ring_score, get_classifier

    candidates = _contour_candidates_near(
        gray,
        preprocessed,
        expected_px,
        radius_guess,
        pitch_px,
        cfg,
    )
    if not candidates:
        return None

    classifier = get_classifier(cfg.resolved_model_path()) if cfg.use_nn_filter else None
    threshold = cfg.nn_confidence if cfg.use_nn_filter else cfg.ring_confidence

    best: DetectedHole | None = None
    best_score = -1.0

    for gx, gy, radius, circularity, dist in candidates:
        ring = compute_ring_score(gray, edges, gx, gy, radius)
        nn_score = classifier.score(gray, gx, gy, radius) if classifier else 0.55
        combined = 0.45 * nn_score + 0.40 * ring + 0.15 * circularity - 0.0004 * dist
        if combined < threshold or combined <= best_score:
            continue
        best_score = combined
        best = DetectedHole(
            center_px=(gx, gy),
            radius_px=radius,
            circularity=circularity,
            nn_score=float(combined),
            grid_residual_px=float(dist),
            source="grid_nn",
            status="confirmed",
        )

    return best


def build_confirmed_grid(
    gray: np.ndarray,
    model: GridModel,
    cfg: TableGridConfig,
    width: int,
    height: int,
    *,
    seed_holes: list[DetectedHole],
    edges: np.ndarray | None = None,
) -> tuple[list[GridSlot], list[DetectedHole]]:
    """
    Fit is already done — walk each lattice cell, NN-confirm hole presence + center.
    Only confirmed holes are returned for calibration.
    """
    preprocessed = preprocess_gray(gray)
    c0, c1, r0, r1 = image_cell_bounds(model, width, height, seed_holes, margin_cells=cfg.grid_margin_cells)

    slots: list[GridSlot] = []
    confirmed: list[DetectedHole] = []

    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            expected = model.cell_to_pixel(col, row)
            ex, ey = float(expected[0]), float(expected[1])
            if ex < -model.pitch_px or ey < -model.pitch_px or ex > width + model.pitch_px or ey > height + model.pitch_px:
                continue

            hole = nn_confirm_at_cell(
                gray,
                preprocessed,
                edges,
                (ex, ey),
                model.mean_radius_px,
                model.pitch_px,
                cfg,
            )
            if hole is not None:
                hole.grid_col = col
                hole.grid_row = row
                hole.grid_residual_px = float(
                    np.linalg.norm(np.array(hole.center_px) - expected)
                )
                slots.append(GridSlot(col, row, (ex, ey), "confirmed", hole))
                confirmed.append(hole)
            else:
                slots.append(GridSlot(col, row, (ex, ey), "missing", None))

    return slots, confirmed
