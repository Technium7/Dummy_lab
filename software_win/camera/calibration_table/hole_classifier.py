from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "hole_patch.pt"
DEFAULT_HOLE_MODEL = _DEFAULT_MODEL
_PATCH_SIZE = 64


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required") from exc
    return cv2


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "torch is required for hole NN filtering. "
            "Install with: pip install -r software_win/requirements.txt"
        ) from exc
    return torch, nn


def extract_hole_patch(gray: np.ndarray, cx: float, cy: float, radius_px: float, patch_size: int = _PATCH_SIZE) -> np.ndarray:
    """Crop a square ROI around the candidate and resize to a fixed patch."""
    cv2 = _require_cv2()
    h, w = gray.shape[:2]
    span = max(radius_px * 3.0, patch_size * 0.35)
    x0 = max(0, int(round(cx - span)))
    y0 = max(0, int(round(cy - span)))
    x1 = min(w, int(round(cx + span)))
    y1 = min(h, int(round(cy + span)))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.zeros((patch_size, patch_size), dtype=np.float32)
    patch = cv2.resize(roi, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    return patch


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    return ((patch.astype(np.float32) / 255.0) - 0.5) / 0.5


def compute_ring_score(gray: np.ndarray, edges: np.ndarray | None, cx: float, cy: float, radius_px: float) -> float:
    """
    Classical donut score: dark interior + edge energy in the rim annulus.
    Returns 0..1 (higher = more hole-like).
    """
    cv2 = _require_cv2()
    h, w = gray.shape[:2]
    span = int(max(6, radius_px * 2.5))
    x0 = max(0, int(round(cx - span)))
    y0 = max(0, int(round(cy - span)))
    x1 = min(w, int(round(cx + span)))
    y1 = min(h, int(round(cy + span)))
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return 0.0
    lcx, lcy = cx - x0, cy - y0
    yy, xx = np.ogrid[: roi.shape[0], : roi.shape[1]]
    dist = np.sqrt((xx - lcx) ** 2 + (yy - lcy) ** 2)
    r = max(radius_px, 2.0)
    inner = dist <= r * 0.85
    ring = (dist > r * 0.85) & (dist <= r * 1.45)
    if not np.any(inner):
        return 0.0
    inner_mean = float(np.mean(roi[inner]))
    outer_mean = float(np.mean(roi))
    darkness = np.clip((outer_mean - inner_mean) / 80.0, 0.0, 1.0)
    edge_score = 0.0
    if edges is not None:
        edge_roi = edges[y0:y1, x0:x1]
        if np.any(ring):
            edge_score = float(np.mean(edge_roi[ring] > 0))
    return float(np.clip(0.55 * darkness + 0.45 * edge_score, 0.0, 1.0))


def _build_model(nn):
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.15),
        nn.Linear(64, 1),
    )


def _synthetic_patch(is_hole: bool, rng: np.random.Generator) -> np.ndarray:
    cv2 = _require_cv2()
    bg = float(rng.uniform(150, 230))
    patch = np.full((_PATCH_SIZE, _PATCH_SIZE), bg, dtype=np.float32)
    if is_hole:
        cx = float(rng.uniform(18, _PATCH_SIZE - 18))
        cy = float(rng.uniform(18, _PATCH_SIZE - 18))
        radius = float(rng.uniform(5, 14))
        yy, xx = np.ogrid[:_PATCH_SIZE, :_PATCH_SIZE]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inner = dist <= radius
        rim = (dist > radius) & (dist <= radius + 2.5)
        patch[inner] = float(rng.uniform(15, 55))
        patch[rim] = np.minimum(patch[rim], float(rng.uniform(70, 120)))
    else:
        kind = int(rng.integers(0, 4))
        if kind == 0:
            patch += rng.normal(0, 12, patch.shape).astype(np.float32)
        elif kind == 1:
            x0, y0 = rng.integers(5, _PATCH_SIZE - 10, size=2)
            patch[y0 : y0 + 8, x0 : x0 + 8] = float(rng.uniform(40, 100))
        elif kind == 2:
            for _ in range(int(rng.integers(2, 5))):
                p1 = rng.integers(0, _PATCH_SIZE, size=2)
                p2 = rng.integers(0, _PATCH_SIZE, size=2)
                cv2.line(patch, tuple(p1), tuple(p2), float(rng.uniform(30, 90)), 1)
        else:
            cx, cy = rng.integers(10, _PATCH_SIZE - 10, size=2)
            cv2.circle(patch, (int(cx), int(cy)), int(rng.integers(3, 8)), float(rng.uniform(60, 180)), -1)
    patch += rng.normal(0, 4, patch.shape).astype(np.float32)
    return np.clip(patch, 0, 255)


def _synthetic_batch(n: int, rng: np.random.Generator, torch) -> tuple[Any, Any]:
    patches = np.zeros((n, 1, _PATCH_SIZE, _PATCH_SIZE), dtype=np.float32)
    labels = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        is_hole = i < n // 2
        labels[i, 0] = 1.0 if is_hole else 0.0
        patch = _synthetic_patch(is_hole, rng)
        patches[i, 0] = normalize_patch(patch)
    return torch.from_numpy(patches), torch.from_numpy(labels)


def train_and_save_synthetic(model_path: Path, *, epochs: int = 32, batch_size: int = 64) -> Path:
    torch, nn = _require_torch()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = _build_model(nn)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    model.train()
    for _ in range(epochs):
        x, y = _synthetic_batch(batch_size, rng, torch)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    torch.save({"state_dict": model.state_dict(), "patch_size": _PATCH_SIZE}, model_path)
    return model_path


@dataclass
class HolePatchClassifier:
    model: Any
    patch_size: int = _PATCH_SIZE

    @classmethod
    def load(cls, model_path: Path | None = None, *, auto_bootstrap: bool = True) -> HolePatchClassifier:
        torch, nn = _require_torch()
        path = model_path or _DEFAULT_MODEL
        if not path.is_file():
            if not auto_bootstrap:
                raise FileNotFoundError(f"Hole classifier weights not found: {path}")
            train_and_save_synthetic(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = _build_model(nn)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(model=model, patch_size=int(ckpt.get("patch_size", _PATCH_SIZE)))

    def score(self, gray: np.ndarray, cx: float, cy: float, radius_px: float) -> float:
        torch, _ = _require_torch()
        patch = normalize_patch(extract_hole_patch(gray, cx, cy, radius_px, self.patch_size))
        x = torch.from_numpy(patch[None, None, ...])
        with torch.no_grad():
            return float(torch.sigmoid(self.model(x)).item())


_classifier_cache: HolePatchClassifier | None = None


def get_classifier(model_path: Path | None = None) -> HolePatchClassifier:
    global _classifier_cache
    path = model_path or _DEFAULT_MODEL
    if _classifier_cache is None:
        _classifier_cache = HolePatchClassifier.load(path)
    return _classifier_cache
