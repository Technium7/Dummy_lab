"""Polygon region selection for filtering calibration holes."""

from __future__ import annotations

import numpy as np

from .grid_types import DetectedHole


def snap_hole_from_click(
    holes: list[DetectedHole],
    click_px: tuple[float, float],
    *,
    max_dist_px: float | None = None,
) -> DetectedHole | None:
    if not holes:
        return None
    best = min(
        holes,
        key=lambda h: (h.center_px[0] - click_px[0]) ** 2 + (h.center_px[1] - click_px[1]) ** 2,
    )
    dist = float(
        np.hypot(best.center_px[0] - click_px[0], best.center_px[1] - click_px[1])
    )
    limit = max_dist_px
    if limit is None:
        limit = max(20.0, best.radius_px * 4.0)
    if dist > limit:
        return None
    return best


def hole_in_polygon(hole: DetectedHole, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return True
    import cv2

    pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    cx, cy = hole.center_px
    return cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0


def partition_holes_by_polygon(
    holes: list[DetectedHole],
    polygon: list[tuple[float, float]],
    *,
    origin: DetectedHole | None = None,
) -> tuple[list[DetectedHole], list[DetectedHole]]:
    inside: list[DetectedHole] = []
    outside: list[DetectedHole] = []
    for hole in holes:
        if origin is not None and hole is origin:
            inside.append(hole)
            continue
        if hole_in_polygon(hole, polygon):
            inside.append(hole)
        else:
            outside.append(hole)
    return inside, outside


def draw_polygon_draft(
    annotated: np.ndarray,
    polygon: list[tuple[float, float]],
    *,
    cursor_px: tuple[float, float] | None = None,
) -> None:
    import cv2

    if not polygon:
        return
    pts_i = [(int(round(x)), int(round(y))) for x, y in polygon]
    if len(pts_i) >= 2:
        cv2.polylines(annotated, [np.array(pts_i, dtype=np.int32).reshape(-1, 1, 2)], False, (0, 200, 255), 2)
    for px, py in pts_i:
        cv2.circle(annotated, (px, py), 5, (0, 200, 255), -1)
    if cursor_px is not None and pts_i:
        cv2.line(
            annotated,
            pts_i[-1],
            (int(round(cursor_px[0])), int(round(cursor_px[1]))),
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )


def draw_region_filtered_grid(
    annotated: np.ndarray,
    inside: list[DetectedHole],
    outside: list[DetectedHole],
    origin: DetectedHole,
    polygon: list[tuple[float, float]],
    pitch_px: float,
    *,
    detection_rejected: list[DetectedHole] | None = None,
    show_origin_marker: bool = True,
) -> None:
    import cv2

    line_color = (90, 90, 90)
    if len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        tint = annotated.copy()
        cv2.fillPoly(tint, [pts], (255, 200, 0))
        cv2.addWeighted(tint, 0.10, annotated, 0.90, 0, annotated)
        cv2.polylines(annotated, [pts], True, (0, 200, 255), 2)

    by_row: dict[int, list[int]] = {}
    for hole in inside:
        by_row.setdefault(hole.grid_row, []).append(hole.grid_col)
    inside_map = {(h.grid_col, h.grid_row): h for h in inside}
    for row, cols in by_row.items():
        cols = sorted(cols)
        pts = [
            (
                int(round(inside_map[(c, row)].center_px[0])),
                int(round(inside_map[(c, row)].center_px[1])),
            )
            for c in cols
        ]
        for i in range(len(pts) - 1):
            cv2.line(annotated, pts[i], pts[i + 1], line_color, 1)

    for hole in outside:
        cx, cy = hole.center_px
        ix, iy = int(round(cx)), int(round(cy))
        r = max(3, int(round(hole.radius_px)))
        cv2.circle(annotated, (ix, iy), r, (100, 100, 100), 1)
        cv2.putText(
            annotated,
            f"{hole.grid_col},{hole.grid_row}",
            (ix + 4, iy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (100, 100, 100),
            1,
        )

    for hole in inside:
        cx, cy = hole.center_px
        ix, iy = int(round(cx)), int(round(cy))
        r = max(3, int(round(hole.radius_px)))
        color = (0, 0, 255) if hole.is_origin and show_origin_marker else (0, 255, 0)
        thickness = 3 if hole.is_origin and show_origin_marker else 2
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

    if detection_rejected:
        for hole in detection_rejected:
            cx, cy = hole.center_px
            ix, iy = int(round(cx)), int(round(cy))
            rad = max(4, int(round(hole.radius_px)))
            cv2.circle(annotated, (ix, iy), rad, (255, 0, 255), 1)
            cv2.line(annotated, (ix - rad, iy - rad), (ix + rad, iy + rad), (255, 0, 255), 1)
            cv2.line(annotated, (ix - rad, iy + rad), (ix + rad, iy - rad), (255, 0, 255), 1)

    if show_origin_marker:
        ox, oy = origin.center_px
        cv2.drawMarker(
            annotated,
            (int(round(ox)), int(round(oy))),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            22,
            2,
        )

    h = annotated.shape[0]
    cv2.putText(
        annotated,
        f"{len(inside)} in region  {len(outside)} excluded  pitch ~{pitch_px:.1f}px",
        (12, h - 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 220, 255),
        1,
    )
    cv2.putText(
        annotated,
        "green=in region  gray=excluded  cyan=boundary  magenta X=not on grid",
        (12, h - 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 200, 180),
        1,
    )


def draw_final_origin_prompt(
    annotated: np.ndarray,
    holes: list[DetectedHole],
    polygon: list[tuple[float, float]] | None,
    pitch_px: float,
) -> None:
    import cv2

    outside: list[DetectedHole] = []
    inside = list(holes)
    placeholder = holes[0] if holes else DetectedHole(
        center_px=(0.0, 0.0), radius_px=5.0, circularity=1.0
    )
    draw_region_filtered_grid(
        annotated,
        inside,
        outside,
        placeholder,
        polygon or [],
        pitch_px,
        detection_rejected=None,
        show_origin_marker=False,
    )
    h = annotated.shape[0]
    cv2.putText(
        annotated,
        "click the calibration origin hole (table 0,0)",
        (12, h - 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255),
        2,
    )
