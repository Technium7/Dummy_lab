from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid_config import TableGridConfig


@dataclass
class RawHoleCandidate:
    center_px: tuple[float, float]
    radius_px: float
    circularity: float
    score: float = 0.0
    source: str = "unknown"


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for table calibration. "
            "Install with: pip install -r requirements-vision.txt"
        ) from exc
    return cv2


def _preprocess_gray(gray: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.bilateralFilter(enhanced, 7, 50, 50)


def preprocess_gray(gray: np.ndarray) -> np.ndarray:
    return _preprocess_gray(gray)


def _canny_edges(preprocessed: np.ndarray) -> tuple[np.ndarray, int, int]:
    cv2 = _require_cv2()
    v = float(np.median(preprocessed))
    lower = int(max(0, 0.5 * v))
    upper = int(min(255, 1.5 * v + 1))
    if upper <= lower:
        upper = min(255, lower + 40)
    edges = cv2.Canny(preprocessed, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return edges, lower, upper


def _contours_to_candidates(
    binary_or_edges: np.ndarray,
    cfg: TableGridConfig,
    *,
    source: str,
    pitch_px: float | None = None,
) -> list[RawHoleCandidate]:
    cv2 = _require_cv2()
    max_area = cfg.resolved_max_area_px()
    contours, _ = cv2.findContours(binary_or_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out: list[RawHoleCandidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < cfg.min_area_px or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < cfg.min_circularity:
            continue
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (ma, mi), _ = ellipse
            radius = 0.25 * (ma + mi)
        else:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 1.0:
            continue
        if cfg.max_radius_px is not None and radius > cfg.max_radius_px:
            continue
        out.append(
            RawHoleCandidate(
                center_px=(float(cx), float(cy)),
                radius_px=float(radius),
                circularity=float(circularity),
                score=float(circularity),
                source=source,
            )
        )
    return out


def _candidates_from_threshold(preprocessed: np.ndarray, cfg: TableGridConfig) -> list[RawHoleCandidate]:
    cv2 = _require_cv2()
    blurred = cv2.GaussianBlur(preprocessed, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _contours_to_candidates(binary, cfg, source="threshold")


def _candidates_from_edges(edges: np.ndarray, cfg: TableGridConfig) -> list[RawHoleCandidate]:
    return _contours_to_candidates(edges, cfg, source="edge")


def _candidates_from_hough(
    preprocessed: np.ndarray,
    cfg: TableGridConfig,
    *,
    min_dist_px: float,
    pitch_px: float | None = None,
) -> list[RawHoleCandidate]:
    cv2 = _require_cv2()
    blurred = cv2.GaussianBlur(preprocessed, (9, 9), 2)
    _, _, upper = _canny_edges(preprocessed)
    _, max_r = cfg.hole_radius_limits(pitch_px)
    min_r_h = 2
    max_r_h = max(min_r_h + 1, int(max_r))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(4.0, min_dist_px * 0.55),
        param1=max(40, upper),
        param2=cfg.hough_param2,
        minRadius=min_r_h,
        maxRadius=max_r_h,
    )
    if circles is None:
        return []
    out: list[RawHoleCandidate] = []
    for cx, cy, radius in circles[0]:
        out.append(
            RawHoleCandidate(
                center_px=(float(cx), float(cy)),
                radius_px=float(radius),
                circularity=1.0,
                score=0.9,
                source="hough",
            )
        )
    return out


def _refine_center_from_edges(
    preprocessed: np.ndarray,
    edges: np.ndarray,
    candidate: RawHoleCandidate,
) -> RawHoleCandidate:
    cv2 = _require_cv2()
    cx, cy = candidate.center_px
    r = candidate.radius_px
    h, w = preprocessed.shape[:2]
    pad = int(max(8, r * 2.5))
    x0 = max(0, int(cx - pad))
    y0 = max(0, int(cy - pad))
    x1 = min(w, int(cx + pad))
    y1 = min(h, int(cy + pad))
    roi_edges = edges[y0:y1, x0:x1]
    contours, _ = cv2.findContours(roi_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return candidate

    best = candidate
    best_score = -1.0
    for contour in contours:
        if len(contour) < 8:
            continue
        area = cv2.contourArea(contour)
        if area < 10:
            continue
        (ex, ey), (ma, mi), _ = cv2.fitEllipse(contour)
        ecc = abs(ma - mi) / (ma + 1e-6)
        if ecc > 0.65:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        dist = (ex + x0 - cx) ** 2 + (ey + y0 - cy) ** 2
        if dist > (r * 1.5) ** 2:
            continue
        score = circularity - 0.001 * dist
        if score > best_score:
            best_score = score
            best = RawHoleCandidate(
                center_px=(float(ex + x0), float(ey + y0)),
                radius_px=float(0.25 * (ma + mi)),
                circularity=float(circularity),
                score=float(max(candidate.score, circularity)),
                source=f"{candidate.source}+ellipse",
            )
    return best


def _merge_candidates(
    groups: list[list[RawHoleCandidate]],
    merge_px: float,
) -> list[RawHoleCandidate]:
    flat = [c for g in groups for c in g]
    if not flat:
        return []
    ranked = sorted(flat, key=lambda c: c.score, reverse=True)
    merged: list[RawHoleCandidate] = []
    for cand in ranked:
        if all(
            (cand.center_px[0] - k.center_px[0]) ** 2 + (cand.center_px[1] - k.center_px[1]) ** 2
            >= merge_px * merge_px
            for k in merged
        ):
            merged.append(cand)
    return merged


def find_hole_candidates(
    gray: np.ndarray,
    cfg: TableGridConfig,
    *,
    pitch_guess_px: float | None = None,
) -> tuple[list[RawHoleCandidate], np.ndarray | None]:
    """Find dark circular holes via Canny edges (+ optional Otsu backup)."""
    preprocessed = _preprocess_gray(gray)
    edges, _, _ = _canny_edges(preprocessed)
    h, w = gray.shape[:2]

    edge_cands = _candidates_from_edges(edges, cfg)
    otsu_cands = _candidates_from_threshold(preprocessed, cfg)
    merge_px = max(4.0, (pitch_guess_px or cfg.min_pitch_px) * 0.35)
    merged = _merge_candidates([edge_cands, otsu_cands], merge_px)
    refined = [_refine_center_from_edges(preprocessed, edges, c) for c in merged]
    return filter_raw_candidates(refined, cfg, w, h, pitch_guess_px), edges


def estimate_pitch_px(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return float("nan")
    dists: list[float] = []
    for i, pt in enumerate(centers):
        others = np.delete(centers, i, axis=0)
        diff = others - pt
        d = np.linalg.norm(diff, axis=1)
        d = d[d > 1.0]
        if d.size:
            dists.append(float(np.min(d)))
    if not dists:
        return float("nan")
    return float(np.median(dists))


def filter_raw_candidates(
    candidates: list[RawHoleCandidate],
    cfg: TableGridConfig,
    width: int,
    height: int,
    pitch_px: float | None = None,
) -> list[RawHoleCandidate]:
    """Drop border candidates; optional --max-radius-px only."""
    if not candidates:
        return []
    margin = cfg.edge_margin_px
    out: list[RawHoleCandidate] = []
    for cand in candidates:
        cx, cy = cand.center_px
        if cx < margin or cy < margin or cx > width - margin or cy > height - margin:
            continue
        if cfg.max_radius_px is not None and cand.radius_px > cfg.max_radius_px:
            continue
        out.append(cand)
    return out


def filter_detected_holes_by_radius(
    holes: list,
    cfg: TableGridConfig,
    pitch_px: float | None = None,
) -> list:
    """No-op unless --max-radius-px set (hole size varies with distance)."""
    if cfg.max_radius_px is None:
        return holes
    return [h for h in holes if h.radius_px <= cfg.max_radius_px]
