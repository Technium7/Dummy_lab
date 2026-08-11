"""RGB + depth + edge feature crops for sensor keypoint head."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .models import CROP_SIZE

# SensorHead input: R, G, B, depth_norm, edge
SENSOR_IN_CHANNELS = 5


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Normalized Sobel-ish magnitude via np.gradient (no OpenCV required)."""
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError("gray must be HxW")
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx * gx + gy * gy)
    m = float(mag.max())
    if m < 1e-8:
        return np.zeros_like(mag, dtype=np.float32)
    return (mag / m).astype(np.float32)


def _resize_hw(arr: np.ndarray, height: int, width: int, *, nearest: bool = True) -> np.ndarray:
    """Resize HxW float map to (height, width)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    try:
        import cv2

        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(a, (width, height), interpolation=interp).astype(np.float32)
    except Exception:
        im = Image.fromarray(a, mode="F")
        resample = Image.NEAREST if nearest else Image.BILINEAR
        return np.asarray(im.resize((width, height), resample), dtype=np.float32)


def _resize_float2d(arr: np.ndarray, size: int, *, nearest: bool = True) -> np.ndarray:
    return _resize_hw(arr, size, size, nearest=nearest)


def depth_to_norm(depth_m: np.ndarray | None, *, shape: tuple[int, int]) -> np.ndarray:
    """Per-crop robust depth normalization to [0, 1]; zeros if missing/invalid."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    if depth_m is None:
        return out
    d = np.asarray(depth_m, dtype=np.float32)
    if d.shape[:2] != (h, w):
        d = _resize_hw(d, h, w, nearest=True)
    valid = np.isfinite(d) & (d > 0.05) & (d < 5.0)
    if not np.any(valid):
        return out
    vals = d[valid]
    lo, hi = np.percentile(vals, [5.0, 95.0])
    if hi <= lo + 1e-6:
        out[valid] = 0.5
        return out
    norm = (d - lo) / (hi - lo)
    out[valid] = np.clip(norm[valid], 0.0, 1.0)
    return out


def edge_map_from_rgb_depth(
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
) -> np.ndarray:
    """Fused edge layer from color and depth (highlights sensor rectangle sides)."""
    rgb_u8 = np.asarray(rgb)
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = np.clip(rgb_u8, 0, 255).astype(np.uint8)
    gray = (
        0.299 * rgb_u8[..., 0].astype(np.float32)
        + 0.587 * rgb_u8[..., 1].astype(np.float32)
        + 0.114 * rgb_u8[..., 2].astype(np.float32)
    ) / 255.0
    color_edge = _sobel_magnitude(gray)

    if depth_m is None:
        return color_edge

    d_norm = depth_to_norm(depth_m, shape=gray.shape)
    depth_edge = _sobel_magnitude(d_norm)
    valid = d_norm > 0
    fused = color_edge.copy()
    fused[valid] = np.maximum(color_edge[valid], depth_edge[valid])
    return fused.astype(np.float32)


def crop_window(
    arr: np.ndarray,
    bbox: list[float],
    *,
    pad: int = 8,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Crop ``arr`` (HxW or HxWxC) with the same pad/clip as RGB training crops."""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(w, int(x1) + pad)
    y1 = min(h, int(y1) + pad)
    crop = arr[y0:y1, x0:x1]
    return crop, (float(x0), float(y0), float(x1), float(y1))


def resize_rgb(crop: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


def resize_map(crop: np.ndarray, size: int = CROP_SIZE, *, nearest: bool = False) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    return _resize_float2d(crop, size, nearest=nearest)


def _square_from_extent(
    x0: int,
    y0: int,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    h_img: int,
    w_img: int,
    min_side_px: float,
    max_side_px: float,
    pad_frac: float,
    pct: tuple[float, float] = (10.0, 90.0),
) -> list[float]:
    x_lo, x_hi = np.percentile(xs.astype(np.float64), list(pct))
    y_lo, y_hi = np.percentile(ys.astype(np.float64), list(pct))
    cx = float(x0) + 0.5 * (x_lo + x_hi)
    cy = float(y0) + 0.5 * (y_lo + y_hi)
    side = float(max(x_hi - x_lo, y_hi - y_lo)) * (1.0 + float(pad_frac))
    # Never expand to the full loose YOLO box — head was trained on ~80–110px crops
    side = float(np.clip(side, min_side_px, max_side_px))
    return [
        float(np.clip(cx - 0.5 * side, 0, w_img - 1)),
        float(np.clip(cy - 0.5 * side, 0, h_img - 1)),
        float(np.clip(cx + 0.5 * side, 0, w_img - 1)),
        float(np.clip(cy + 0.5 * side, 0, h_img - 1)),
    ]


def refine_sensor_bbox(
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    bbox: list[float],
    *,
    edge_percentile: float = 75.0,
    min_side_px: float = 72.0,
    max_side_px: float = 110.0,
    pad_frac: float = 0.45,
    ignore_top_frac: float = 0.25,
    depth_closer_m: float = 0.003,
) -> list[float]:
    """Tighten a loose YOLO sensor box to match training crops (~square, ~sensor body).

    Prefer a depth "raised blob" inside the YOLO box; fall back to edges. Tall/wide
    YOLO boxes make the keypoint head collapse both points toward the crop center.
    """
    h_img, w_img = rgb.shape[:2]
    x0 = int(np.clip(bbox[0], 0, w_img - 1))
    y0 = int(np.clip(bbox[1], 0, h_img - 1))
    x1 = int(np.clip(bbox[2], 0, w_img - 1))
    y1 = int(np.clip(bbox[3], 0, h_img - 1))
    bw = float(x1 - x0)
    bh = float(y1 - y0)
    if bw < 8 or bh < 8:
        return [float(v) for v in bbox]

    # 1) Depth blob: closer-than-median pixels ≈ raised sensor / holder vs table
    if depth_m is not None:
        d_full = np.asarray(depth_m, dtype=np.float32)
        if d_full.shape[:2] == rgb.shape[:2]:
            d_roi = d_full[y0:y1, x0:x1]
            valid = np.isfinite(d_roi) & (d_roi > 0.05) & (d_roi < 5.0)
            if int(valid.sum()) >= 50:
                vals = d_roi[valid]
                med = float(np.median(vals))
                close = valid & (d_roi < (med - float(depth_closer_m)))
                if int(close.sum()) < 30:
                    thr = float(np.percentile(vals, 35.0))
                    close = valid & (d_roi < thr)
                ys_d, xs_d = np.where(close)
                if len(xs_d) >= 20:
                    return _square_from_extent(
                        x0,
                        y0,
                        xs_d,
                        ys_d,
                        h_img=h_img,
                        w_img=w_img,
                        min_side_px=min_side_px,
                        max_side_px=max_side_px,
                        pad_frac=pad_frac,
                    )

    # 2) Edge fallback (ignore empty top of tall YOLO boxes)
    rgb_roi = rgb[y0:y1, x0:x1]
    d_roi = None
    if depth_m is not None:
        d = np.asarray(depth_m, dtype=np.float32)
        if d.shape[:2] == rgb.shape[:2]:
            d_roi = d[y0:y1, x0:x1]
    edge = edge_map_from_rgb_depth(rgb_roi, d_roi)
    if float(edge.max()) >= 1e-6:
        thr = float(np.percentile(edge, edge_percentile))
        ys, xs = np.where(edge >= max(thr, 1e-3))
        y_cut = ignore_top_frac * bh
        keep = ys >= y_cut
        if int(keep.sum()) >= 20:
            xs, ys = xs[keep], ys[keep]
        if len(xs) >= 20:
            return _square_from_extent(
                x0,
                y0,
                xs,
                ys,
                h_img=h_img,
                w_img=w_img,
                min_side_px=min_side_px,
                max_side_px=max_side_px,
                pad_frac=0.35,
                pct=(8.0, 92.0),
            )

    # 3) Geometric fallback: square ~min(w,h) biased toward lower half of YOLO box
    side = float(np.clip(min(bw, bh) * 1.05, min_side_px, max_side_px))
    cx = 0.5 * (x0 + x1)
    cy = y0 + 0.65 * bh
    return [
        float(np.clip(cx - 0.5 * side, 0, w_img - 1)),
        float(np.clip(cy - 0.5 * side, 0, h_img - 1)),
        float(np.clip(cx + 0.5 * side, 0, w_img - 1)),
        float(np.clip(cy + 0.5 * side, 0, h_img - 1)),
    ]


def build_sensor_crop_chw(
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    bbox: list[float],
    *,
    pad: int = 8,
    size: int = CROP_SIZE,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return float32 CHW (5, size, size) = RGB/255 + depth_norm + edge, and crop origin."""
    rgb_crop, origin = crop_window(rgb, bbox, pad=pad)
    rgb_r = resize_rgb(rgb_crop, size)

    if depth_m is not None:
        d_crop, _ = crop_window(np.asarray(depth_m, dtype=np.float32), bbox, pad=pad)
        d_r = (
            _resize_float2d(d_crop, size, nearest=True)
            if d_crop.size
            else np.zeros((size, size), dtype=np.float32)
        )
        d_norm = depth_to_norm(d_r, shape=(size, size))
        d_full_crop = d_crop if d_crop.size else None
    else:
        d_norm = np.zeros((size, size), dtype=np.float32)
        d_full_crop = None

    edge_src_rgb = rgb_crop if rgb_crop.size else rgb_r
    edge_full = edge_map_from_rgb_depth(edge_src_rgb, d_full_crop)
    edge_r = resize_map(edge_full, size, nearest=False)
    emax = float(edge_r.max())
    if emax > 1e-8:
        edge_r = (edge_r / emax).astype(np.float32)
    else:
        edge_r = np.zeros((size, size), dtype=np.float32)

    rgb_f = rgb_r.astype(np.float32) / 255.0
    chw = np.concatenate(
        [
            rgb_f.transpose(2, 0, 1),
            d_norm[None, ...],
            edge_r[None, ...],
        ],
        axis=0,
    ).astype(np.float32)
    if chw.shape != (SENSOR_IN_CHANNELS, size, size):
        raise RuntimeError(f"bad sensor crop shape {chw.shape}")
    return chw, origin
