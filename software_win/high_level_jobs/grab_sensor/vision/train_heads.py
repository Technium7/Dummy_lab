"""Train ClassNet + SensorHead + CenterHead from crop labels.

SensorHead: grasp UV + (cos θ, sin θ) holder axis from RGB + depth + edge.
CenterHead: post/other center UV from RGB YOLO crops.
ClassNet remains RGB-only (cascade backend).
"""

from __future__ import annotations

import argparse
import json
import math
import random
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

from high_level_jobs.grab_sensor.config_loader import (  # noqa: E402
    apply_runtime_calibrations,
    load_grab_sensor_config,
)
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
    contact_axis_geometry,
    cos_sin_from_kpts,
)

SENSOR_CROP_PAD = 16
CENTER_CROP_PAD = 16  # matches live_detect crop_padding_px for post/other


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
        self.sensor_raw: list[tuple[np.ndarray, np.ndarray | None, list[float], list, str]] = []
        self.center_raw: list[tuple[np.ndarray, list[float], list[float], str, str]] = []
        self.items: list[tuple[np.ndarray, dict]] = []
        for lp in iter_labels(data_dir):
            data = load_label(lp)
            img_rel = (data.get("files") or {}).get("image")
            if not img_rel:
                continue
            rgb = np.asarray(Image.open(data_dir / img_rel).convert("RGB"))
            depth = _load_depth(data_dir, data) if kind == "sensor" else None
            sid = str(data.get("id") or lp.stem)
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
                            sid,
                        )
                    )
                elif kind == "center" and cls in ("post", "other"):
                    kpts = obj.get("kpts") or []
                    if not kpts:
                        continue
                    self.center_raw.append(
                        (
                            rgb,
                            list(obj["bbox_xyxy"]),
                            [float(kpts[0][0]), float(kpts[0][1])],
                            sid,
                            cls,
                        )
                    )

    def subset(self, indices: list[int], *, augment: bool) -> CropDataset:
        other = CropDataset.__new__(CropDataset)
        other.kind = self.kind
        other.augment = bool(augment) and self.kind == "sensor"
        other.rng = np.random.default_rng(0)
        other.sensor_raw = [self.sensor_raw[i] for i in indices] if self.kind == "sensor" else []
        other.center_raw = [self.center_raw[i] for i in indices] if self.kind == "center" else []
        other.items = (
            [self.items[i] for i in indices] if self.kind not in ("sensor", "center") else []
        )
        return other

    def __len__(self) -> int:
        if self.kind == "sensor":
            return len(self.sensor_raw)
        if self.kind == "center":
            return len(self.center_raw)
        return len(self.items)

    def __getitem__(self, idx: int):
        if self.kind == "sensor":
            rgb, depth, bbox, kpts, _sid = self.sensor_raw[idx]
            h, w = rgb.shape[:2]
            use_bbox = list(bbox)
            if self.augment and self.rng.random() < 0.85:
                use_bbox = _jitter_bbox(bbox, width=w, height=h, rng=self.rng)
            chw, origin = build_sensor_crop_chw(rgb, depth, use_bbox, pad=SENSOR_CROP_PAD)
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

        if self.kind == "center":
            rgb, bbox, kpt, _sid, _cls = self.center_raw[idx]
            crop, origin = _crop_resize(rgb, bbox, pad=CENTER_CROP_PAD)
            u, v = _to_crop_uv(kpt, origin)
            x = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)
            y = torch.tensor(
                [float(np.clip(u, 0, 1)), float(np.clip(v, 0, 1))],
                dtype=torch.float32,
            )
            return x, y

        crop, meta = self.items[idx]
        x = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)
        if "class" in meta:
            y = torch.tensor(CLASS_TO_IDX[meta["class"]], dtype=torch.long)
            return x, y
        y = torch.tensor(meta["target"], dtype=torch.float32)
        return x, y


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(
        0.0, by1 - by0
    ) - inter
    return inter / union if union > 1e-6 else 0.0


def _match_yolo_box(
    gt_bbox: list[float],
    contact_uv: list[float],
    yolo_boxes: list[list[float]],
    *,
    taken: set[int],
    min_iou: float = 0.1,
) -> int | None:
    """Pick the unused YOLO sensor box that would be used at runtime for this label."""
    best_j, best_iou = None, min_iou
    for j, box in enumerate(yolo_boxes):
        if j in taken:
            continue
        iou = _iou_xyxy(gt_bbox, box)
        if iou > best_iou:
            best_iou, best_j = iou, j
    if best_j is not None:
        return best_j
    cu, cv = float(contact_uv[0]), float(contact_uv[1])
    best_j, best_d = None, 1e9
    for j, box in enumerate(yolo_boxes):
        if j in taken:
            continue
        x0, y0, x1, y1 = box
        if not (x0 - 16 <= cu <= x1 + 16 and y0 - 16 <= cv <= y1 + 16):
            continue
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        d = math.hypot(cx - cu, cy - cv)
        if d < best_d:
            best_d, best_j = d, j
    return best_j


def apply_yolo_sensor_boxes(ds: CropDataset) -> tuple[int, int]:
    """Replace GT label boxes with YOLO boxes so train/eval match live_detect."""
    from high_level_jobs.grab_sensor.vision.infer import _yolo_boxes

    cfg = apply_runtime_calibrations(load_grab_sensor_config())
    vision = cfg.vision
    by_sid: dict[str, list[int]] = {}
    for i, rec in enumerate(ds.sensor_raw):
        by_sid.setdefault(rec[4], []).append(i)

    kept: list = []
    n_drop = 0
    for _sid, idxs in by_sid.items():
        rgb = ds.sensor_raw[idxs[0]][0]
        yolo_boxes = [b for b, _s, name in _yolo_boxes(rgb, vision) if name == "sensor"]
        taken: set[int] = set()
        for i in idxs:
            _rgb, depth, gt_bbox, kpts, sid = ds.sensor_raw[i]
            j = _match_yolo_box(gt_bbox, kpts[0], yolo_boxes, taken=taken)
            if j is None:
                n_drop += 1
                continue
            taken.add(j)
            kept.append((_rgb, depth, list(yolo_boxes[j]), kpts, sid))
    ds.sensor_raw = kept
    return len(kept), n_drop


def apply_yolo_center_boxes(ds: CropDataset) -> tuple[int, int, dict[str, int], dict[str, int]]:
    """Replace GT post/other boxes with same-class YOLO boxes (live_detect crops)."""
    from high_level_jobs.grab_sensor.vision.infer import _yolo_boxes

    cfg = apply_runtime_calibrations(load_grab_sensor_config())
    vision = cfg.vision
    by_sid: dict[str, list[int]] = {}
    for i, rec in enumerate(ds.center_raw):
        by_sid.setdefault(rec[3], []).append(i)

    kept: list = []
    n_drop = 0
    keep_cls: dict[str, int] = {"post": 0, "other": 0}
    drop_cls: dict[str, int] = {"post": 0, "other": 0}
    for _sid, idxs in by_sid.items():
        rgb = ds.center_raw[idxs[0]][0]
        dets = _yolo_boxes(rgb, vision)
        taken: set[int] = set()
        for i in idxs:
            _rgb, gt_bbox, kpt, sid, cls = ds.center_raw[i]
            local_boxes: list[list[float]] = []
            local_to_global: list[int] = []
            for j, (box, _score, name) in enumerate(dets):
                if name == cls and j not in taken:
                    local_boxes.append(box)
                    local_to_global.append(j)
            lj = _match_yolo_box(gt_bbox, kpt, local_boxes, taken=set())
            if lj is None:
                n_drop += 1
                drop_cls[cls] = drop_cls.get(cls, 0) + 1
                continue
            gj = local_to_global[lj]
            taken.add(gj)
            kept.append((_rgb, list(dets[gj][0]), kpt, sid, cls))
            keep_cls[cls] = keep_cls.get(cls, 0) + 1
    ds.center_raw = kept
    return len(kept), n_drop, keep_cls, drop_cls


def _split_indices(n: int, test_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if n < 2 or test_frac <= 0:
        return list(range(n)), []
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_frac)))
    n_test = min(n_test, n - 1)
    return idx[n_test:], idx[:n_test]


def _chrono_frame_split(
    sensor_raw: list,
    data_dir: Path,
    *,
    train_first: int,
    test_last: int,
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Split by sorted label-frame time: first N frames train, last M frames test."""
    frame_ids = [p.stem for p in iter_labels(data_dir)]
    if train_first + test_last > len(frame_ids):
        print(
            f"[sensor_head] warn: {train_first}+{test_last} frames requested, "
            f"only {len(frame_ids)} label files"
        )
    train_sids = set(frame_ids[: max(0, train_first)])
    test_sids = set(frame_ids[len(frame_ids) - max(0, test_last) :] if test_last > 0 else [])
    overlap = train_sids & test_sids
    if overlap:
        train_sids -= overlap
        print(f"[sensor_head] warn: {len(overlap)} frames in both first/last; held out for test")
    train_idx = [i for i, rec in enumerate(sensor_raw) if rec[4] in train_sids]
    test_idx = [i for i, rec in enumerate(sensor_raw) if rec[4] in test_sids]
    return train_idx, test_idx, sorted(train_sids), sorted(test_sids)


def _summarize_px(errors: list[float]) -> dict[str, float]:
    arr = np.asarray(errors, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


@torch.no_grad()
def eval_center_keypoint_px(
    model: torch.nn.Module,
    ds: CropDataset,
    device: torch.device,
) -> list[dict[str, float | str]]:
    """Center L2 error in full-image pixels (YOLO bbox crop, no jitter)."""
    model.eval()
    rows: list[dict[str, float | str]] = []
    for rgb, bbox, kpt, sid, cls in ds.center_raw:
        crop, origin = _crop_resize(rgb, bbox, pad=CENTER_CROP_PAD)
        x = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        pred = model(x)[0].detach().cpu().numpy()
        x0, y0, x1, y1 = origin
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        pu = float(np.clip(pred[0], 0, 1)) * bw + x0
        pv = float(np.clip(pred[1], 0, 1)) * bh + y0
        rows.append(
            {
                "id": sid,
                "class": cls,
                "l2_px": float(math.hypot(pu - float(kpt[0]), pv - float(kpt[1]))),
            }
        )
    return rows


def _print_center_errors(name: str, rows: list[dict[str, float | str]]) -> dict[str, float]:
    stats = _summarize_px([float(r["l2_px"]) for r in rows])
    print(
        f"[center_head] {name} n={stats['n']}  "
        f"L2 mean={stats['mean']:.2f}px ({stats['mean']/PX_PER_IN:.3f}in)  "
        f"med={stats['median']:.2f}  p90={stats['p90']:.2f}px ({stats['p90']/PX_PER_IN:.3f}in)"
    )
    for cls in ("post", "other"):
        sub = [float(r["l2_px"]) for r in rows if r["class"] == cls]
        if not sub:
            continue
        cs = _summarize_px(sub)
        print(
            f"[center_head] {name} {cls} n={cs['n']}  "
            f"mean={cs['mean']:.2f}px ({cs['mean']/PX_PER_IN:.3f}in)  "
            f"p90={cs['p90']:.2f}px ({cs['p90']/PX_PER_IN:.3f}in)"
        )
        stats[f"{cls}_n"] = cs["n"]
        stats[f"{cls}_mean_px"] = cs["mean"]
        stats[f"{cls}_p90_px"] = cs["p90"]
    return stats


@torch.no_grad()
def eval_sensor_keypoint_px(
    model: torch.nn.Module,
    ds: CropDataset,
    device: torch.device,
) -> list[float]:
    """Contact L2 error in full-image pixels (YOLO bbox crop, no jitter)."""
    return [row["l2_px"] for row in eval_sensor_axis_errors(model, ds, device)]


@torch.no_grad()
def eval_sensor_axis_errors(
    model: torch.nn.Module,
    ds: CropDataset,
    device: torch.device,
) -> list[dict[str, float | str]]:
    """Contact error vs GT handle axis: along, perp, L2 (full-image px)."""
    model.eval()
    rows: list[dict[str, float | str]] = []
    for rgb, depth, bbox, kpts, sid in ds.sensor_raw:
        chw, origin = build_sensor_crop_chw(rgb, depth, bbox, pad=SENSOR_CROP_PAD)
        pred = model(torch.from_numpy(np.asarray(chw, dtype=np.float32))[None].to(device))[0]
        pred = pred.detach().cpu().numpy()
        x0, y0, x1, y1 = origin
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        pu = float(np.clip(pred[0], 0, 1)) * bw + x0
        pv = float(np.clip(pred[1], 0, 1)) * bh + y0
        geo = contact_axis_geometry(kpts)
        ax, ay = geo.along_uv
        dx = pu - float(geo.contact_uv[0])
        dy = pv - float(geo.contact_uv[1])
        along = dx * ax + dy * ay
        perp = dx * (-ay) + dy * ax
        rows.append(
            {
                "id": sid,
                "along_px": float(along),
                "perp_px": float(perp),
                "l2_px": float(math.hypot(dx, dy)),
            }
        )
    return rows


PX_PER_IN = 16.4


def _print_axis_errors(name: str, rows: list[dict[str, float | str]]) -> dict[str, float]:
    if not rows:
        print(f"[sensor_head] {name}  n=0")
        return {"n": 0}
    along = np.array([float(r["along_px"]) for r in rows], dtype=np.float64)
    perp = np.array([float(r["perp_px"]) for r in rows], dtype=np.float64)
    l2 = np.array([float(r["l2_px"]) for r in rows], dtype=np.float64)
    along_abs = np.abs(along)
    perp_abs = np.abs(perp)
    stats = {
        "n": int(len(rows)),
        "along_mean_px": float(along.mean()),
        "along_p90_px": float(np.percentile(along_abs, 90)),
        "perp_mean_px": float(perp.mean()),
        "perp_p90_px": float(np.percentile(perp_abs, 90)),
        "along_abs_mean_px": float(along_abs.mean()),
        "perp_abs_mean_px": float(perp_abs.mean()),
        "l2_mean_px": float(l2.mean()),
        "l2_median_px": float(np.median(l2)),
        "l2_p90_px": float(np.percentile(l2, 90)),
    }
    print(
        f"[sensor_head] {name} n={stats['n']}  "
        f"along mean={stats['along_mean_px']:+.2f}px ({stats['along_mean_px']/PX_PER_IN:+.3f}in)  "
        f"p90={stats['along_p90_px']:.2f}px ({stats['along_p90_px']/PX_PER_IN:.3f}in)  "
        f"perp mean={stats['perp_mean_px']:+.2f}px ({stats['perp_mean_px']/PX_PER_IN:+.3f}in)  "
        f"p90={stats['perp_p90_px']:.2f}px ({stats['perp_p90_px']/PX_PER_IN:.3f}in)"
    )
    return stats


def _print_keypoint_split(name: str, stats: dict[str, float]) -> None:
    print(
        f"[sensor_head] {name} contact error px  "
        f"n={stats['n']}  mean={stats['mean']:.2f}  med={stats['median']:.2f}  p90={stats['p90']:.2f}"
    )


def _train_loop(model, loader, *, epochs: int, lr: float, classification: bool) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
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
        mean_loss = total / max(1, len(loader))
        history.append(mean_loss)
        print(f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}")
    return history


def _save_loss_curve(history: list[float], path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history) + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv")
    csv_path.write_text(
        "epoch,loss\n" + "\n".join(f"{e},{v:.8f}" for e, v in zip(epochs, history)) + "\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history, color="#1f4e79", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Wrote loss curve {path}  (+ {csv_path.name})")


def train_heads(
    data_dir: Path,
    *,
    epochs: int = 60,
    out_dir: Path | None = None,
    only: str | None = None,
    test_frac: float = 0.1,
    seed: int = 0,
    train_first: int | None = None,
    test_last: int | None = None,
) -> None:
    out_dir = out_dir or MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    only = (only or "all").lower()
    if only not in ("all", "class", "sensor", "center"):
        raise ValueError(f"Unknown --only {only!r}; use all|class|sensor|center")

    if only in ("all", "class"):
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

    if only in ("all", "sensor"):
        sensor_all = CropDataset(data_dir, "sensor", augment=False)
        if len(sensor_all) == 0:
            print(
                "[sensor_head] skipped — no contact+holder-dir sensor labels "
                "(relabel: contact, keep/set along holder)"
            )
        else:
            n_keep, n_drop = apply_yolo_sensor_boxes(sensor_all)
            print(
                f"[sensor_head] YOLO boxes (runtime crops) kept={n_keep}  "
                f"dropped_no_det={n_drop}"
            )
            if n_keep < 2:
                print("[sensor_head] skipped — need at least 2 YOLO-matched sensors")
            else:
                # --train-first N alone: keep first N frames, then random test_frac.
                # --train-first + --test-last: chrono holdout.
                if train_first is not None and (test_last is None or int(test_last) <= 0):
                    frame_ids = [p.stem for p in iter_labels(data_dir)]
                    keep_sids = set(frame_ids[: int(train_first)])
                    kept_idx = [
                        i
                        for i, rec in enumerate(sensor_all.sensor_raw)
                        if rec[4] in keep_sids
                    ]
                    sensor_all = sensor_all.subset(kept_idx, augment=False)
                    train_idx, test_idx = _split_indices(len(sensor_all), test_frac, seed)
                    split_meta = {
                        "split": "random_in_train_first",
                        "train_first": train_first,
                        "test_frac": test_frac,
                        "seed": seed,
                    }
                    print(
                        f"[sensor_head] first {train_first} frames -> "
                        f"{len(sensor_all)} sensors, then random  "
                        f"train={len(train_idx)}  test={len(test_idx)}  "
                        f"test_frac={test_frac}  seed={seed}  crop=yolo"
                    )
                elif train_first is not None or test_last is not None:
                    train_idx, test_idx, train_frame_ids, test_frame_ids = _chrono_frame_split(
                        sensor_all.sensor_raw,
                        data_dir,
                        train_first=int(train_first or 0),
                        test_last=int(test_last or 0),
                    )
                    split_meta = {
                        "split": "chrono",
                        "train_first": train_first,
                        "test_last": test_last,
                        "train_frames": train_frame_ids,
                        "test_frames": test_frame_ids,
                    }
                    print(
                        f"[sensor_head] chrono split  first {train_first} frames -> "
                        f"train sensors={len(train_idx)}  last {test_last} frames -> "
                        f"test sensors={len(test_idx)}  crop=yolo"
                    )
                else:
                    train_idx, test_idx = _split_indices(len(sensor_all), test_frac, seed)
                    split_meta = {"split": "random", "test_frac": test_frac, "seed": seed}
                    print(
                        f"[sensor_head] contact + cos/sin holder axis  RGBD+edge  "
                        f"train={len(train_idx)}  test={len(test_idx)}  "
                        f"test_frac={test_frac}  seed={seed}  crop=yolo"
                    )
                if len(train_idx) < 1:
                    print("[sensor_head] skipped — empty train split")
                else:
                    train_ds = sensor_all.subset(train_idx, augment=True)
                    test_ds = sensor_all.subset(test_idx, augment=False)
                    print(
                        f"[sensor_head] contact + cos/sin holder axis  RGBD+edge  "
                        f"train={len(train_ds)}  test={len(test_ds)}  crop=yolo"
                    )
                    loader = DataLoader(train_ds, batch_size=min(16, len(train_ds)), shuffle=True)
                    sensor_head = SensorHead()
                    loss_hist = _train_loop(
                        sensor_head, loader, epochs=epochs, lr=1e-3, classification=False
                    )
                    _save_loss_curve(
                        loss_hist,
                        out_dir / "sensor_head_loss.png",
                        title=f"SensorHead train loss ({epochs} epochs)",
                    )
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    sensor_head = sensor_head.to(device)
                    train_rows = eval_sensor_axis_errors(sensor_head, train_ds, device)
                    test_rows = eval_sensor_axis_errors(sensor_head, test_ds, device)
                    train_axis = _print_axis_errors("train", train_rows)
                    test_axis = _print_axis_errors("test", test_rows)
                    split_payload = {
                        "crop": "yolo",
                        "px_per_in": PX_PER_IN,
                        "train_loss": loss_hist,
                        "train_ids": [sensor_all.sensor_raw[i][4] for i in train_idx],
                        "test_ids": [sensor_all.sensor_raw[i][4] for i in test_idx],
                        "train_axis_error": train_axis,
                        "test_axis_error": test_axis,
                        "test_rows": test_rows,
                    }
                    split_payload.update(split_meta)
                    split_path = out_dir / "sensor_head_split.json"
                    split_path.write_text(
                        json.dumps(split_payload, indent=2),
                        encoding="utf-8",
                    )
                    print(f"Wrote split {split_path}")
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

    if only in ("all", "center"):
        center_all = CropDataset(data_dir, "center")
        if len(center_all) == 0:
            print("[center_head] skipped — no post/other labels")
        else:
            n_keep, n_drop, keep_cls, drop_cls = apply_yolo_center_boxes(center_all)
            print(
                f"[center_head] YOLO boxes (runtime crops) kept={n_keep}  "
                f"dropped_no_det={n_drop}  "
                f"post keep/drop={keep_cls.get('post', 0)}/{drop_cls.get('post', 0)}  "
                f"other keep/drop={keep_cls.get('other', 0)}/{drop_cls.get('other', 0)}"
            )
            if n_keep < 2:
                print("[center_head] skipped — need at least 2 YOLO-matched post/other")
            else:
                train_idx, test_idx = _split_indices(len(center_all), test_frac, seed)
                train_ds = center_all.subset(train_idx, augment=False)
                test_ds = center_all.subset(test_idx, augment=False)
                print(
                    f"[center_head] RGB center UV  train={len(train_ds)}  test={len(test_ds)}  "
                    f"test_frac={test_frac}  seed={seed}  crop=yolo  pad={CENTER_CROP_PAD}"
                )
                loader = DataLoader(train_ds, batch_size=min(16, len(train_ds)), shuffle=True)
                center_head = CenterHead()
                _train_loop(center_head, loader, epochs=epochs, lr=1e-3, classification=False)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                center_head = center_head.to(device)
                train_rows = eval_center_keypoint_px(center_head, train_ds, device)
                test_rows = eval_center_keypoint_px(center_head, test_ds, device)
                train_stats = _print_center_errors("train", train_rows)
                test_stats = _print_center_errors("test", test_rows)
                split_path = out_dir / "center_head_split.json"
                split_path.write_text(
                    json.dumps(
                        {
                            "split": "random",
                            "test_frac": test_frac,
                            "seed": seed,
                            "crop": "yolo",
                            "pad_px": CENTER_CROP_PAD,
                            "px_per_in": PX_PER_IN,
                            "train_ids": [center_all.center_raw[i][3] for i in train_idx],
                            "test_ids": [center_all.center_raw[i][3] for i in test_idx],
                            "train_center_error_px": train_stats,
                            "test_center_error_px": test_stats,
                            "test_rows": test_rows,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"Wrote split {split_path}")
                path = out_dir / "center_head.pt"
                torch.save({"state_dict": center_head.state_dict(), "kind": "center"}, path)
                print(f"Saved {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATASET_DIR)
    p.add_argument("--out", type=Path, default=MODEL_DIR)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument(
        "--only",
        choices=("all", "class", "sensor", "center"),
        default="all",
        help="Train one head instead of all three",
    )
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--train-first",
        type=int,
        default=None,
        help="Train on first N label frames (sorted by id). Overrides --test-frac.",
    )
    p.add_argument(
        "--test-last",
        type=int,
        default=None,
        help="Evaluate on last M label frames (sorted by id).",
    )
    args = p.parse_args(argv)
    train_heads(
        args.data,
        epochs=args.epochs,
        out_dir=args.out,
        only=args.only,
        test_frac=args.test_frac,
        seed=args.seed,
        train_first=args.train_first,
        test_last=args.test_last,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
