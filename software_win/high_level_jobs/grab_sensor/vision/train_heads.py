"""Train ClassNet + SensorHead + CenterHead from crop labels.

SensorHead: grasp UV + (cos θ, sin θ) holder axis from RGB + depth + edge.
ClassNet / CenterHead remain RGB-only.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.grab_sensor.paths import DATASET_DIR, MODEL_DIR  # noqa: E402
from high_level_jobs.grab_sensor.vision.crop_features import build_sensor_crop_chw  # noqa: E402
from high_level_jobs.grab_sensor.vision.dataset_io import (  # noqa: E402
    CLASS_TO_IDX,
    iter_labels,
    load_label,
)
from high_level_jobs.grab_sensor.vision.models import (  # noqa: E402
    CROP_SIZE,
    CenterHead,
    ClassNet,
    SensorHead,
)
from high_level_jobs.grab_sensor.vision.sensor_geometry import (  # noqa: E402
    SENSOR_NUM_KPTS,
    cos_sin_from_kpts,
)


def _crop_resize(
    rgb: np.ndarray,
    bbox: list[float],
    pad: int = 12,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(w, int(x1) + pad)
    y1 = min(h, int(y1) + pad)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    else:
        crop = np.asarray(Image.fromarray(crop).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR))
    return crop, (float(x0), float(y0), float(x1), float(y1))


def _to_crop_uv(uv: list[float], origin: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = origin
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    return ((uv[0] - x0) / bw, (uv[1] - y0) / bh)


def _load_depth(data_dir: Path, data: dict) -> np.ndarray | None:
    rel = (data.get("files") or {}).get("depth")
    if not rel:
        return None
    path = data_dir / rel
    if not path.is_file():
        return None
    return np.load(path).astype(np.float32)


def _jitter_bbox(
    bbox: list[float],
    *,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> list[float]:
    """Randomly expand/shift GT box so the head learns loose YOLO-like crops."""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw = max(8.0, x1 - x0)
    bh = max(8.0, y1 - y0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    scale = float(rng.uniform(1.0, 1.8))
    sx0 = float(rng.uniform(0.3, 0.7))
    sy0 = float(rng.uniform(0.2, 0.8))
    nw, nh = bw * scale, bh * scale
    nx0 = cx - sx0 * nw
    ny0 = cy - sy0 * nh
    nx1 = nx0 + nw
    ny1 = ny0 + nh
    nx0 = float(np.clip(nx0, 0, width - 1))
    ny0 = float(np.clip(ny0, 0, height - 1))
    nx1 = float(np.clip(nx1, 0, width - 1))
    ny1 = float(np.clip(ny1, 0, height - 1))
    if nx1 - nx0 < 8 or ny1 - ny0 < 8:
        return [x0, y0, x1, y1]
    return [nx0, ny0, nx1, ny1]


def _augment_sensor_chw(
    chw: np.ndarray,
    target: list[float],
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[float]]:
    """Light aug: brightness + occasional depth/edge dropout."""
    x = chw.copy()
    y = list(target)
    if rng.random() < 0.8:
        scale = float(rng.uniform(0.75, 1.25))
        x[:3] = np.clip(x[:3] * scale, 0.0, 1.0)
    if rng.random() < 0.15:
        x[3] = 0.0
    if rng.random() < 0.1:
        x[4] = 0.0
    return x, y


class CropDataset(Dataset):
    def __init__(self, data_dir: Path, kind: str, *, augment: bool = False) -> None:
        self.kind = kind
        self.augment = bool(augment) and kind == "sensor"
        self.rng = np.random.default_rng(0)
        self.sensor_raw: list[tuple[np.ndarray, np.ndarray | None, list[float], list]] = []
        self.items: list[tuple[np.ndarray, dict]] = []
        for lp in iter_labels(data_dir):
            data = load_label(lp)
            img_rel = (data.get("files") or {}).get("image")
            if not img_rel:
                continue
            rgb = np.asarray(Image.open(data_dir / img_rel).convert("RGB"))
            depth = _load_depth(data_dir, data) if kind == "sensor" else None
            for obj in data.get("objects") or []:
                cls = str(obj.get("class", "other")).lower()
                if kind == "class":
                    crop, _ = _crop_resize(rgb, obj["bbox_xyxy"])
                    self.items.append((crop, {"class": cls}))
                elif kind == "sensor" and cls == "sensor":
                    kpts = obj.get("kpts") or []
                    if len(kpts) != SENSOR_NUM_KPTS:
                        continue
                    dx = float(kpts[1][0]) - float(kpts[0][0])
                    dy = float(kpts[1][1]) - float(kpts[0][1])
                    if math.hypot(dx, dy) < 4.0:
                        continue
                    self.sensor_raw.append(
                        (
                            rgb,
                            depth,
                            list(obj["bbox_xyxy"]),
                            [[float(p[0]), float(p[1])] for p in kpts[:2]],
                        )
                    )
                elif kind == "center" and cls in ("post", "other"):
                    crop, origin = _crop_resize(rgb, obj["bbox_xyxy"])
                    kpts = obj.get("kpts") or []
                    if not kpts:
                        continue
                    u, v = _to_crop_uv(kpts[0], origin)
                    self.items.append((crop, {"target": [u, v]}))

    def __len__(self) -> int:
        if self.kind == "sensor":
            return len(self.sensor_raw)
        return len(self.items)

    def __getitem__(self, idx: int):
        if self.kind == "sensor":
            rgb, depth, bbox, kpts = self.sensor_raw[idx]
            h, w = rgb.shape[:2]
            use_bbox = list(bbox)
            if self.augment and self.rng.random() < 0.85:
                use_bbox = _jitter_bbox(bbox, width=w, height=h, rng=self.rng)
            chw, origin = build_sensor_crop_chw(rgb, depth, use_bbox, pad=12)
            gu, gv = _to_crop_uv(kpts[0], origin)
            cth, sth = cos_sin_from_kpts(kpts)
            target = [
                float(np.clip(gu, 0, 1)),
                float(np.clip(gv, 0, 1)),
                float(cth),
                float(sth),
            ]
            if self.augment:
                chw, target = _augment_sensor_chw(chw, target, rng=self.rng)
            x = torch.from_numpy(np.asarray(chw, dtype=np.float32))
            y = torch.tensor(target, dtype=torch.float32)
            return x, y

        crop, meta = self.items[idx]
        x = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)
        if "class" in meta:
            y = torch.tensor(CLASS_TO_IDX[meta["class"]], dtype=torch.long)
            return x, y
        y = torch.tensor(meta["target"], dtype=torch.float32)
        return x, y


def _train_loop(model, loader, *, epochs: int, lr: float, classification: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred, y) if classification else F.smooth_l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {epoch + 1}/{epochs} loss={total / max(1, len(loader)):.4f}")


def train_heads(
    data_dir: Path,
    *,
    epochs: int = 60,
    out_dir: Path | None = None,
) -> None:
    out_dir = out_dir or MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    class_ds = CropDataset(data_dir, "class")
    if len(class_ds) == 0:
        raise RuntimeError(f"No class crops in {data_dir}")
    loader = DataLoader(class_ds, batch_size=min(16, len(class_ds)), shuffle=True)
    class_net = ClassNet(pretrained=True)
    print("[class_net]")
    _train_loop(class_net, loader, epochs=epochs, lr=1e-3, classification=True)
    path = out_dir / "class_net.pt"
    torch.save({"state_dict": class_net.state_dict(), "kind": "class"}, path)
    print(f"Saved {path}")

    sensor_ds = CropDataset(data_dir, "sensor", augment=True)
    if len(sensor_ds) > 0:
        loader = DataLoader(sensor_ds, batch_size=min(16, len(sensor_ds)), shuffle=True)
        sensor_head = SensorHead()
        print(f"[sensor_head] contact + cos/sin holder axis  RGBD+edge  n={len(sensor_ds)}")
        _train_loop(sensor_head, loader, epochs=epochs, lr=1e-3, classification=False)
        path = out_dir / "sensor_head.pt"
        torch.save(
            {
                "state_dict": sensor_head.state_dict(),
                "kind": "sensor_contact_axis",
                "in_channels": 5,
                "num_kpts": 2,
            },
            path,
        )
        print(f"Saved {path}")
    else:
        print(
            "[sensor_head] skipped — no contact+holder-dir sensor labels "
            "(relabel: contact, keep/set along holder)"
        )

    center_ds = CropDataset(data_dir, "center")
    if len(center_ds) > 0:
        loader = DataLoader(center_ds, batch_size=min(16, len(center_ds)), shuffle=True)
        center_head = CenterHead()
        print("[center_head]")
        _train_loop(center_head, loader, epochs=epochs, lr=1e-3, classification=False)
        path = out_dir / "center_head.pt"
        torch.save({"state_dict": center_head.state_dict(), "kind": "center"}, path)
        print(f"Saved {path}")
    else:
        print("[center_head] skipped — no post/other labels")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATASET_DIR)
    p.add_argument("--out", type=Path, default=MODEL_DIR)
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args(argv)
    train_heads(args.data, epochs=args.epochs, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
