"""Full-frame Zelux mono preprocess + light fixed-camera augmentations."""

from __future__ import annotations

import cv2
import numpy as np
import torch

# Approximate ImageNet luminance stats (for 1-ch stem adapted from RGB weights)
GRAY_MEAN = 0.449
GRAY_STD = 0.226
DIFF_MEAN = 0.05
DIFF_STD = 0.20


def bgr_to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """BGR or gray → HxWx1 uint8/float gray."""
    arr = np.asarray(image_bgr)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 1:
        gray = arr[:, :, 0]
    elif arr.ndim == 3:
        # Zelux mono often stored as identical BGR channels — take channel 0 (fast)
        # or true convert if color ever appears.
        b, g, r = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        if np.array_equal(b, g) and np.array_equal(g, r):
            gray = b
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Expected HxW or HxWxC image, got shape {arr.shape}")
    return gray[:, :, None]


# Back-compat alias used by older call sites
def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return bgr_to_gray(image_bgr)


def abs_frame_diff(curr: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    if prev is None or prev.shape != curr.shape:
        return np.zeros_like(curr)
    return cv2.absdiff(curr, prev)


def _as_hwc1(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[:, :, None]
    if img.ndim == 3 and img.shape[2] == 1:
        return img
    if img.ndim == 3:
        return img[:, :, :1]
    raise ValueError(f"Expected HxW or HxWx1, got shape {img.shape}")


def build_temporal_mono(curr_gray: np.ndarray, prev_gray: np.ndarray | None) -> np.ndarray:
    """Stack gray + |diff| → HxWx2."""
    g = _as_hwc1(curr_gray)
    prev = _as_hwc1(prev_gray) if prev_gray is not None else None
    d = _as_hwc1(abs_frame_diff(g, prev))
    return np.concatenate([g, d], axis=2)


def build_six_channel(curr_rgb: np.ndarray, prev_rgb: np.ndarray | None) -> np.ndarray:
    """Deprecated name: returns mono temporal (HxWx2)."""
    curr = bgr_to_gray(curr_rgb) if curr_rgb.ndim == 3 and curr_rgb.shape[2] == 3 else (
        curr_rgb if curr_rgb.ndim == 3 else curr_rgb[:, :, None]
    )
    prev = None
    if prev_rgb is not None:
        if prev_rgb.ndim == 3 and prev_rgb.shape[2] == 3:
            prev = bgr_to_gray(prev_rgb)
        else:
            prev = prev_rgb if prev_rgb.ndim == 3 else prev_rgb[:, :, None]
    return build_temporal_mono(curr, prev)


def crop_around_tip(
    image_hwc: np.ndarray,
    x: float,
    y: float,
    crop_size: int,
) -> np.ndarray:
    """Square crop centered on (x, y) in full-frame pixels; replicate pad at edges."""
    img = np.asarray(image_hwc)
    if img.ndim == 2:
        img = img[:, :, None]
    h, w = img.shape[:2]
    half = crop_size // 2
    cx, cy = int(round(x)), int(round(y))
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + crop_size, y0 + crop_size
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    crop = img[y0c:y1c, x0c:x1c]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REPLICATE
        )
        if crop.ndim == 2:
            crop = crop[:, :, None]
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        if crop.ndim == 2:
            crop = crop[:, :, None]
    return crop


def gaussian_heatmap(
    height: int,
    width: int,
    x: float,
    y: float,
    sigma: float = 2.0,
) -> np.ndarray:
    """Normalized Gaussian peak at (x, y) in heatmap pixel coords → HxW float32."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    heat = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
    m = float(heat.max())
    if m > 0:
        heat /= m
    return heat.astype(np.float32)


def resize_hwc(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


class EpoxyAugment:
    def __init__(
        self,
        brightness: float = 0.15,
        contrast: float = 0.15,
        exposure: float = 0.10,
        noise_std: float = 0.02,
        translate_frac: float = 0.02,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.exposure = exposure
        self.noise_std = noise_std
        self.translate_frac = translate_frac

    def __call__(self, image_hwc: np.ndarray) -> np.ndarray:
        was_uint8 = image_hwc.dtype == np.uint8
        x = image_hwc.astype(np.float32)
        if was_uint8:
            x /= 255.0
        b = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        c = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        x = np.clip((x - 0.5) * c + 0.5, 0.0, 1.0)
        x = np.clip(x * b, 0.0, 1.0)
        e = 1.0 + np.random.uniform(-self.exposure, self.exposure)
        x = np.clip(x * e, 0.0, 1.0)
        if self.translate_frac > 0:
            h, w = x.shape[:2]
            max_dx = int(self.translate_frac * w)
            max_dy = int(self.translate_frac * h)
            dx = np.random.randint(-max_dx, max_dx + 1) if max_dx else 0
            dy = np.random.randint(-max_dy, max_dy + 1) if max_dy else 0
            if dx or dy:
                m = np.float32([[1, 0, dx], [0, 1, dy]])
                x = cv2.warpAffine(
                    x, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                if x.ndim == 2:
                    x = x[..., None]
        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, x.shape).astype(np.float32)
            x = np.clip(x + noise, 0.0, 1.0)
        if was_uint8:
            return (x * 255.0).astype(np.uint8)
        return x


def to_model_tensor(image_hwc: np.ndarray, image_size: int = 224) -> torch.Tensor:
    img = resize_hwc(image_hwc, image_size).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.max() > 1.5:
        img /= 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    if t.shape[0] == 1:
        t[0] = (t[0] - GRAY_MEAN) / GRAY_STD
    elif t.shape[0] == 2:
        t[0] = (t[0] - GRAY_MEAN) / GRAY_STD
        t[1] = (t[1] - DIFF_MEAN) / DIFF_STD
    elif t.shape[0] >= 3:
        # Legacy RGB path
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t[:3] = (t[:3] - mean) / std
        if t.shape[0] >= 6:
            t[3:6] = (t[3:6] - DIFF_MEAN) / DIFF_STD
    return t
