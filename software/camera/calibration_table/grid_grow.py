"""Grow a hole grid from origin by walking rays with multi-point fit and local search."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .grid_config import TableGridConfig
from .grid_types import DetectedHole
from .hole_detect import RawHoleCandidate, estimate_pitch_px


_CARDINAL: list[tuple[tuple[int, int], tuple[float, float]]] = [
    ((1, 0), (1.0, 0.0)),
    ((-1, 0), (-1.0, 0.0)),
    ((0, 1), (0.0, 1.0)),
    ((0, -1), (0.0, -1.0)),
]


@dataclass
class _CandidatePool:
    holes: list[DetectedHole]
    used: list[bool]
    centers: np.ndarray

    @classmethod
    def from_holes(cls, holes: list[DetectedHole]) -> _CandidatePool:
        return cls(
            holes=holes,
            used=[False] * len(holes),
            centers=np.array([h.center_px for h in holes], dtype=np.float64),
        )

    def mark(self, index: int) -> None:
        self.used[index] = True


def _hole_from_candidate(c: RawHoleCandidate) -> DetectedHole:
    return DetectedHole(
        center_px=c.center_px,
        radius_px=c.radius_px,
        circularity=c.circularity,
        nn_score=c.circularity,
        source=c.source,
        status="detected",
    )


def _dedupe_candidates(candidates: list[RawHoleCandidate], merge_px: float) -> list[RawHoleCandidate]:
    ranked = sorted(candidates, key=lambda c: c.circularity, reverse=True)
    kept: list[RawHoleCandidate] = []
    for cand in ranked:
        if all(
            (cand.center_px[0] - k.center_px[0]) ** 2 + (cand.center_px[1] - k.center_px[1]) ** 2
            >= merge_px * merge_px
            for k in kept
        ):
            kept.append(cand)
    return kept


def _extrapolate_1d(indices: list[int], positions: list[np.ndarray], k: int) -> np.ndarray | None:
    """Fit/extrapolate pixel position along a grid index axis from confirmed holes."""
    if not indices:
        return None
    pairs = sorted(zip(indices, positions), key=lambda x: x[0])
    idx = [p[0] for p in pairs]
    pos = [np.asarray(p[1], dtype=np.float64) for p in pairs]

    if k in idx:
        return pos[idx.index(k)]

    if len(idx) == 1:
        return None

    if k > idx[-1]:
        i1, i2 = idx[-2], idx[-1]
        p1, p2 = pos[-2], pos[-1]
        di = i2 - i1
        if di == 0:
            return p2.copy()
        return p2 + (p2 - p1) * ((k - i2) / di)

    if k < idx[0]:
        i1, i2 = idx[0], idx[1]
        p1, p2 = pos[0], pos[1]
        di = i2 - i1
        if di == 0:
            return p1.copy()
        return p1 - (p2 - p1) * ((i1 - k) / di)

    for j in range(len(idx) - 1):
        if idx[j] <= k <= idx[j + 1]:
            t = (k - idx[j]) / (idx[j + 1] - idx[j])
            return pos[j] * (1.0 - t) + pos[j + 1] * t
    return None


def _ray_step_magnitude(known: dict[tuple[int, int], DetectedHole], *, row: int | None, col: int | None) -> float:
    steps: list[float] = []
    if row is not None:
        cols = sorted(c for c, r in known if r == row)
        for a, b in zip(cols, cols[1:]):
            pa = np.array(known[(a, row)].center_px, dtype=np.float64)
            pb = np.array(known[(b, row)].center_px, dtype=np.float64)
            steps.append(float(np.linalg.norm(pb - pa) / max(abs(b - a), 1)))
    if col is not None:
        rows = sorted(r for c, r in known if c == col)
        for a, b in zip(rows, rows[1:]):
            pa = np.array(known[(col, a)].center_px, dtype=np.float64)
            pb = np.array(known[(col, b)].center_px, dtype=np.float64)
            steps.append(float(np.linalg.norm(pb - pa) / max(abs(b - a), 1)))
    if steps:
        return float(np.median(steps))
    return 20.0


def _predict_cell_px(
    known: dict[tuple[int, int], DetectedHole],
    col: int,
    row: int,
    origin_px: tuple[float, float],
    step_u: np.ndarray,
    step_v: np.ndarray,
) -> np.ndarray:
    """Predict pixel from multi-hole fits on the row and column through (col, row)."""
    row_pred, col_pred = _predict_axes_px(known, col, row, origin_px, step_u, step_v)
    if row_pred is not None and col_pred is not None:
        return np.mean(np.stack([row_pred, col_pred], axis=0), axis=0)
    if row_pred is not None:
        return row_pred
    if col_pred is not None:
        return col_pred

    o = np.array(origin_px, dtype=np.float64)
    return o + col * step_u + row * step_v


def _predict_axes_px(
    known: dict[tuple[int, int], DetectedHole],
    col: int,
    row: int,
    origin_px: tuple[float, float],
    step_u: np.ndarray,
    step_v: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    row_pred: np.ndarray | None = None
    col_pred: np.ndarray | None = None

    cols_on_row = sorted(c for c, r in known if r == row)
    if cols_on_row:
        pos_row = [np.array(known[(c, row)].center_px, dtype=np.float64) for c in cols_on_row]
        row_pred = _extrapolate_1d(cols_on_row, pos_row, col)

    rows_on_col = sorted(r for c, r in known if c == col)
    if rows_on_col:
        pos_col = [np.array(known[(col, r)].center_px, dtype=np.float64) for r in rows_on_col]
        col_pred = _extrapolate_1d(rows_on_col, pos_col, row)

    if row_pred is None and col_pred is None:
        return None, None
    return row_pred, col_pred


def _ray_tangent(
    known: dict[tuple[int, int], DetectedHole],
    col: int,
    row: int,
    dc: int,
    dr: int,
    step_u: np.ndarray,
    step_v: np.ndarray,
) -> np.ndarray:
    if dc != 0:
        cols = sorted(c for c, r in known if r == row)
        if len(cols) >= 2:
            c1, c2 = cols[-2], cols[-1]
            if c2 != c1:
                p1 = np.array(known[(c1, row)].center_px, dtype=np.float64)
                p2 = np.array(known[(c2, row)].center_px, dtype=np.float64)
                return p2 - p1
        return step_u * dc
    rows = sorted(r for c, r in known if c == col)
    if len(rows) >= 2:
        r1, r2 = rows[-2], rows[-1]
        if r2 != r1:
            p1 = np.array(known[(col, r1)].center_px, dtype=np.float64)
            p2 = np.array(known[(col, r2)].center_px, dtype=np.float64)
            return p2 - p1
    return step_v * dr


def _search_radius(
    known: dict[tuple[int, int], DetectedHole],
    col: int,
    row: int,
    step_u: np.ndarray,
    step_v: np.ndarray,
) -> float:
    step = _ray_step_magnitude(known, row=row, col=col)
    fallback = float(max(np.linalg.norm(step_u), np.linalg.norm(step_v), 8.0))
    return max(6.0, min(step, fallback) * 0.44)


def _match_at_target(
    pool: _CandidatePool,
    pred: np.ndarray,
    search_r: float,
    *,
    ray_tangent: np.ndarray | None = None,
) -> int:
    """Pick best unused candidate near predicted target (not global nearest)."""
    best_i = -1
    best_cost = float("inf")
    r2 = search_r * search_r
    tangent: np.ndarray | None = None
    if ray_tangent is not None:
        n = float(np.linalg.norm(ray_tangent))
        if n > 1e-6:
            tangent = ray_tangent / n

    for i, h in enumerate(pool.holes):
        if pool.used[i]:
            continue
        d = pool.centers[i] - pred
        dist2 = float(d @ d)
        if dist2 > r2:
            continue
        dist = float(np.sqrt(dist2))
        cost = dist
        if tangent is not None:
            lateral = abs(float(d[0] * tangent[1] - d[1] * tangent[0]))
            cost += 1.8 * lateral
        cost += 0.15 * (1.0 - float(np.clip(h.circularity, 0.0, 1.0)))
        if cost < best_cost:
            best_cost = cost
            best_i = i
    return best_i


def _cardinal_cell_from_vector(vx: float, vy: float) -> tuple[int, int]:
    best_cell = (1, 0)
    best_align = -2.0
    norm = float(np.hypot(vx, vy))
    if norm < 1e-6:
        return (0, 0)
    for cell, (dx, dy) in _CARDINAL:
        align = float(vx * dx + vy * dy) / norm
        if align > best_align:
            best_align = align
            best_cell = cell
    return best_cell


def _snap_click_to_hole(
    click_px: tuple[float, float],
    pool: _CandidatePool,
    *,
    skip: set[int],
) -> int:
    best_i = -1
    best_d2 = float("inf")
    for i, h in enumerate(pool.holes):
        if i in skip or pool.used[i]:
            continue
        d2 = float(
            (h.center_px[0] - click_px[0]) ** 2 + (h.center_px[1] - click_px[1]) ** 2
        )
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def _seeds_from_manual_clicks(
    origin_px: tuple[float, float],
    pool: _CandidatePool,
    seed_clicks: list[tuple[float, float]],
    *,
    origin_i: int,
) -> dict[tuple[int, int], int]:
    seeds: dict[tuple[int, int], int] = {}
    skip = {origin_i}
    for click in seed_clicks:
        idx = _snap_click_to_hole(click, pool, skip=skip)
        if idx < 0:
            continue
        vx = float(pool.centers[idx][0] - origin_px[0])
        vy = float(pool.centers[idx][1] - origin_px[1])
        if float(np.hypot(vx, vy)) < 6.0:
            continue
        cell = _cardinal_cell_from_vector(vx, vy)
        if cell == (0, 0) or cell in seeds:
            continue
        seeds[cell] = idx
        skip.add(idx)
    return seeds


def prepare_origin_pool(
    candidates: list[RawHoleCandidate],
    origin_click_px: tuple[float, float],
    cfg: TableGridConfig,
) -> tuple[DetectedHole, _CandidatePool, float, int]:
    """Pick origin hole and candidate pool for manual seed selection."""
    if not candidates:
        raise ValueError("no hole candidates")

    pitch_guess = estimate_pitch_px(np.array([c.center_px for c in candidates], dtype=np.float64))
    if not np.isfinite(pitch_guess):
        pitch_guess = cfg.min_pitch_px * 2.5
    merge_px = max(4.0, pitch_guess * 0.35)
    candidates = _dedupe_candidates(candidates, merge_px)

    holes = [_hole_from_candidate(c) for c in candidates]
    pool = _CandidatePool.from_holes(holes)

    near_r = max(12.0, pitch_guess * 0.5)
    dists = np.linalg.norm(pool.centers - np.array(origin_click_px), axis=1)
    near = np.where(dists <= near_r)[0]
    if len(near):
        origin_i = int(near[np.argmax([pool.holes[i].circularity for i in near])])
    else:
        origin_i = int(np.argmin(dists))

    origin = pool.holes[origin_i]
    origin.grid_col = 0
    origin.grid_row = 0
    origin.is_origin = True
    origin.status = "locked"
    return origin, pool, pitch_guess, origin_i


def assign_seed_click(
    origin_px: tuple[float, float],
    pool: _CandidatePool,
    click_px: tuple[float, float],
    *,
    origin_i: int,
    assigned: dict[tuple[int, int], int],
) -> tuple[tuple[int, int], int] | None:
    """Snap one click to a hole and assign the nearest cardinal direction."""
    skip = {origin_i, *assigned.values()}
    idx = _snap_click_to_hole(click_px, pool, skip=skip)
    if idx < 0:
        return None
    vx = float(pool.centers[idx][0] - origin_px[0])
    vy = float(pool.centers[idx][1] - origin_px[1])
    if float(np.hypot(vx, vy)) < 6.0:
        return None
    cell = _cardinal_cell_from_vector(vx, vy)
    if cell == (0, 0) or cell in assigned:
        return None
    return cell, idx


_CARDINAL_LABEL = {
    (1, 0): "E",
    (-1, 0): "W",
    (0, 1): "S",
    (0, -1): "N",
}


def draw_seed_picking_overlay(
    annotated: np.ndarray,
    pool: _CandidatePool,
    origin: DetectedHole,
    assigned: dict[tuple[int, int], DetectedHole],
) -> None:
    import cv2

    for i, h in enumerate(pool.holes):
        cx, cy = h.center_px
        ix, iy = int(round(cx)), int(round(cy))
        cv2.circle(annotated, (ix, iy), max(3, int(round(h.radius_px))), (255, 255, 0), 1)

    for cell, hole in assigned.items():
        cx, cy = hole.center_px
        ix, iy = int(round(cx)), int(round(cy))
        label = _CARDINAL_LABEL.get(cell, "?")
        cv2.circle(annotated, (ix, iy), max(5, int(round(hole.radius_px)) + 2), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            label,
            (ix + 6, iy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        ox, oy = origin.center_px
        cv2.line(
            annotated,
            (int(round(ox)), int(round(oy))),
            (ix, iy),
            (0, 200, 200),
            1,
        )

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
    remaining = 4 - len(assigned)
    dirs = [lbl for cell, lbl in _CARDINAL_LABEL.items() if cell not in assigned]
    need = f" ({'/'.join(dirs)})" if dirs else ""
    cv2.putText(
        annotated,
        f"click {remaining} neighbor hole(s){need}  ({len(assigned)}/4)",
        (12, h - 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 220, 255),
        1,
    )
    cv2.putText(
        annotated,
        "cyan=seed  yellow circles=candidates  auto E/W/N/S from click angle",
        (12, h - 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 200, 180),
        1,
    )


def _find_cardinal_seeds(
    origin_px: tuple[float, float],
    pool: _CandidatePool,
    pitch_px: float,
) -> dict[tuple[int, int], int]:
    ox, oy = origin_px
    min_p = max(6.0, pitch_px * 0.55)
    max_p = pitch_px * 1.65
    seeds: dict[tuple[int, int], int] = {}
    taken: set[int] = set()
    search_r = max(6.0, pitch_px * 0.38)

    for cell_delta, dir_vec in _CARDINAL:
        dx, dy = dir_vec
        pred = np.array([ox + dx * pitch_px, oy + dy * pitch_px], dtype=np.float64)
        best_i = -1
        best_score = -1.0
        for i, h in enumerate(pool.holes):
            if pool.used[i] or i in taken:
                continue
            vx = h.center_px[0] - ox
            vy = h.center_px[1] - oy
            dist = float(np.hypot(vx, vy))
            if dist < min_p or dist > max_p:
                continue
            norm = dist + 1e-9
            align = float(vx * dx + vy * dy) / norm
            if align < 0.78:
                continue
            off = float(np.linalg.norm(pool.centers[i] - pred))
            if off > search_r:
                continue
            score = align - 2.5 * off / max(pitch_px, 1.0)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i >= 0:
            seeds[cell_delta] = best_i
            taken.add(best_i)
    return seeds


def _axis_step(
    known: dict[tuple[int, int], DetectedHole],
    origin_px: tuple[float, float],
    positive: tuple[int, int],
    negative: tuple[int, int],
) -> np.ndarray | None:
    o = np.array(origin_px, dtype=np.float64)
    if positive in known:
        return np.array(known[positive].center_px, dtype=np.float64) - o
    if negative in known:
        return o - np.array(known[negative].center_px, dtype=np.float64)
    return None


def _refine_step_from_ray(
    known: dict[tuple[int, int], DetectedHole],
    *,
    axis: str,
) -> np.ndarray | None:
    steps: list[np.ndarray] = []
    if axis == "col":
        by_row: dict[int, list[int]] = {}
        for c, r in known:
            by_row.setdefault(r, []).append(c)
        for row, cols in by_row.items():
            cols = sorted(cols)
            for a, b in zip(cols, cols[1:]):
                pa = np.array(known[(a, row)].center_px, dtype=np.float64)
                pb = np.array(known[(b, row)].center_px, dtype=np.float64)
                steps.append((pb - pa) / max(abs(b - a), 1))
    else:
        by_col: dict[int, list[int]] = {}
        for c, r in known:
            by_col.setdefault(c, []).append(r)
        for col, rows in by_col.items():
            rows = sorted(rows)
            for a, b in zip(rows, rows[1:]):
                pa = np.array(known[(col, a)].center_px, dtype=np.float64)
                pb = np.array(known[(col, b)].center_px, dtype=np.float64)
                steps.append((pb - pa) / max(abs(b - a), 1))
    if not steps:
        return None
    return np.median(np.stack(steps, axis=0), axis=0)


def _try_claim_cell(
    known: dict[tuple[int, int], DetectedHole],
    pool: _CandidatePool,
    col: int,
    row: int,
    origin_px: tuple[float, float],
    step_u: np.ndarray,
    step_v: np.ndarray,
    *,
    dc: int = 0,
    dr: int = 0,
) -> bool:
    if (col, row) in known:
        return True

    row_pred, col_pred = _predict_axes_px(known, col, row, origin_px, step_u, step_v)
    pred = _predict_cell_px(known, col, row, origin_px, step_u, step_v)
    search_r = _search_radius(known, col, row, step_u, step_v)
    tangent = _ray_tangent(known, col, row, dc, dr, step_u, step_v) if (dc or dr) else None
    idx = _match_at_target(pool, pred, search_r, ray_tangent=tangent)
    if idx < 0:
        return False

    center = pool.centers[idx]
    residual = float(np.linalg.norm(center - pred))
    if residual > search_r:
        return False

    axis_r = search_r * 0.85
    n_row = sum(1 for c, r in known if r == row)
    n_col = sum(1 for c, r in known if c == col)
    if row_pred is not None and col_pred is not None and n_row >= 2 and n_col >= 2:
        row_err = float(np.linalg.norm(center - row_pred))
        col_err = float(np.linalg.norm(center - col_pred))
        if row_err > axis_r or col_err > axis_r:
            return False

    hole = pool.holes[idx]
    hole.grid_col = col
    hole.grid_row = row
    hole.grid_residual_px = residual
    known[(col, row)] = hole
    pool.mark(idx)
    return True


def _walk_ray(
    known: dict[tuple[int, int], DetectedHole],
    pool: _CandidatePool,
    origin_px: tuple[float, float],
    step_u: np.ndarray,
    step_v: np.ndarray,
    dc: int,
    dr: int,
    *,
    width: int,
    height: int,
    margin: float,
    max_gap: int = 4,
) -> None:
    """Walk along one axis through missing cells using extrapolated targets."""
    if (dc, dr) not in known:
        return

    k = 1
    while (dc * k, dr * k) in known:
        k += 1
    gap = 0
    while gap < max_gap:
        nc, nr = dc * k, dr * k
        pred = _predict_cell_px(known, nc, nr, origin_px, step_u, step_v)
        if pred[0] < margin or pred[1] < margin or pred[0] > width - margin or pred[1] > height - margin:
            break

        if _try_claim_cell(
            known, pool, nc, nr, origin_px, step_u, step_v, dc=dc, dr=dr
        ):
            k += 1
            gap = 0
            ru = _refine_step_from_ray(known, axis="col")
            rv = _refine_step_from_ray(known, axis="row")
            if ru is not None:
                step_u[:] = ru
            if rv is not None:
                step_v[:] = rv
        else:
            gap += 1
            k += 1


def grow_grid_from_origin(
    candidates: list[RawHoleCandidate],
    origin_click_px: tuple[float, float],
    cfg: TableGridConfig,
    *,
    width: int,
    height: int,
    manual_seed_clicks: list[tuple[float, float]] | None = None,
) -> tuple[dict[tuple[int, int], DetectedHole], list[DetectedHole], DetectedHole, float, int]:
    """
    Returns (known_cells, rejected, origin_hole, pitch_px, n_seeds).
    Origin is always grid cell (0, 0).
    """
    origin, pool, pitch_guess, origin_i = prepare_origin_pool(candidates, origin_click_px, cfg)
    origin_px = origin.center_px
    pool.mark(origin_i)

    known: dict[tuple[int, int], DetectedHole] = {(0, 0): origin}

    if manual_seed_clicks:
        seed_map = _seeds_from_manual_clicks(
            origin_px, pool, manual_seed_clicks, origin_i=origin_i
        )
    else:
        seed_map = _find_cardinal_seeds(origin_px, pool, pitch_guess)

    for cell, idx in seed_map.items():
        h = pool.holes[idx]
        h.grid_col, h.grid_row = cell
        known[cell] = h
        pool.mark(idx)

    step_u = _axis_step(known, origin_px, (1, 0), (-1, 0))
    step_v = _axis_step(known, origin_px, (0, 1), (0, -1))
    if step_u is None and step_v is None:
        step_u = np.array([pitch_guess, 0.0], dtype=np.float64)
        step_v = np.array([0.0, pitch_guess], dtype=np.float64)
    elif step_u is None:
        step_u = np.array([step_v[1], -step_v[0]], dtype=np.float64)
    elif step_v is None:
        step_v = np.array([-step_u[1], step_u[0]], dtype=np.float64)

    refined_u = _refine_step_from_ray(known, axis="col")
    refined_v = _refine_step_from_ray(known, axis="row")
    if refined_u is not None:
        step_u = refined_u.copy()
    if refined_v is not None:
        step_v = refined_v.copy()

    pitch_px = float(0.5 * (np.linalg.norm(step_u) + np.linalg.norm(step_v)))
    margin = cfg.edge_margin_px

    for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _walk_ray(
            known,
            pool,
            origin_px,
            step_u,
            step_v,
            dc,
            dr,
            width=width,
            height=height,
            margin=margin,
        )

    queue: deque[tuple[int, int]] = deque(sorted(known.keys(), key=lambda c: abs(c[0]) + abs(c[1])))
    in_queue = set(known.keys())

    while queue:
        col, row = queue.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = col + dc, row + dr
            if (nc, nr) in known:
                continue
            pred = _predict_cell_px(known, nc, nr, origin_px, step_u, step_v)
            if (
                pred[0] < margin
                or pred[1] < margin
                or pred[0] > width - margin
                or pred[1] > height - margin
            ):
                continue
            if not _try_claim_cell(
                known, pool, nc, nr, origin_px, step_u, step_v, dc=dc, dr=dr
            ):
                continue
            if (nc, nr) not in in_queue:
                queue.append((nc, nr))
                in_queue.add((nc, nr))

        if len(known) % 10 == 0:
            ru = _refine_step_from_ray(known, axis="col")
            rv = _refine_step_from_ray(known, axis="row")
            if ru is not None:
                step_u[:] = ru
            if rv is not None:
                step_v[:] = rv
            pitch_px = float(0.5 * (np.linalg.norm(step_u) + np.linalg.norm(step_v)))

    for _ in range(10):
        added = 0
        cols = [c for c, _ in known]
        rows = [r for _, r in known]
        for col in range(min(cols) - 1, max(cols) + 2):
            for row in range(min(rows) - 1, max(rows) + 2):
                if (col, row) in known:
                    continue
                pred = _predict_cell_px(known, col, row, origin_px, step_u, step_v)
                if (
                    pred[0] < margin
                    or pred[1] < margin
                    or pred[0] > width - margin
                    or pred[1] > height - margin
                ):
                    continue
                if _try_claim_cell(known, pool, col, row, origin_px, step_u, step_v):
                    added += 1
        ru = _refine_step_from_ray(known, axis="col")
        rv = _refine_step_from_ray(known, axis="row")
        if ru is not None:
            step_u[:] = ru
        if rv is not None:
            step_v[:] = rv
        if added == 0:
            break
        pitch_px = float(0.5 * (np.linalg.norm(step_u) + np.linalg.norm(step_v)))

    rejected: list[DetectedHole] = []
    for i, h in enumerate(pool.holes):
        if not pool.used[i]:
            h.status = "rejected"
            rejected.append(h)

    return known, rejected, origin, pitch_px, len(seed_map)


def draw_grown_grid(
    annotated: np.ndarray,
    known: dict[tuple[int, int], DetectedHole],
    rejected: list[DetectedHole],
    origin: DetectedHole,
    pitch_px: float,
) -> None:
    import cv2

    line_color = (90, 90, 90)
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for col, row in known:
        by_row.setdefault(row, []).append(col)
        by_col.setdefault(col, []).append(row)

    for row, cols in by_row.items():
        cols = sorted(cols)
        pts = [
            (int(round(known[(c, row)].center_px[0])), int(round(known[(c, row)].center_px[1])))
            for c in cols
        ]
        for i in range(len(pts) - 1):
            cv2.line(annotated, pts[i], pts[i + 1], line_color, 1)

    for col, rows in by_col.items():
        rows = sorted(rows)
        pts = [
            (int(round(known[(col, r)].center_px[0])), int(round(known[(col, r)].center_px[1])))
            for r in rows
        ]
        for i in range(len(pts) - 1):
            cv2.line(annotated, pts[i], pts[i + 1], line_color, 1)

    for hole in known.values():
        cx, cy = hole.center_px
        ix, iy = int(round(cx)), int(round(cy))
        r = max(3, int(round(hole.radius_px)))
        color = (0, 0, 255) if hole.is_origin else (0, 255, 0)
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

    for hole in rejected:
        cx, cy = hole.center_px
        ix, iy = int(round(cx)), int(round(cy))
        rad = max(4, int(round(hole.radius_px)))
        cv2.circle(annotated, (ix, iy), rad, (255, 0, 255), 1)
        cv2.line(annotated, (ix - rad, iy - rad), (ix + rad, iy + rad), (255, 0, 255), 1)
        cv2.line(annotated, (ix - rad, iy + rad), (ix + rad, iy - rad), (255, 0, 255), 1)

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
        f"{len(known)} grown  {len(rejected)} rejected  pitch ~{pitch_px:.1f}px",
        (12, h - 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 220, 255),
        1,
    )
    cv2.putText(
        annotated,
        "green=grown grid  gray=path fit  magenta X=rejected  red cross=origin",
        (12, h - 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 200, 180),
        1,
    )
