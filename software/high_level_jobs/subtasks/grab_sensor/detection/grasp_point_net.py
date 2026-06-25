"""RGB+depth grasp-point CNN for YOLO crop inference."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

INPUT_W = 320
INPUT_H = 240
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_DEPTH_MIN_M = 0.25
DEFAULT_DEPTH_MAX_M = 0.85
NUM_OUTPUTS_GRASP_ONLY = 2
NUM_OUTPUTS_GRASP_BASE = 4
from .base_ellipse import (
    DEFAULT_ANGLE_DEG,
    DEFAULT_AXIS_A_PX,
    DEFAULT_AXIS_B_PX,
    NUM_OUTPUTS_BASE_ELLIPSE,
    BaseEllipse,
    ellipse_to_norm_crop,
    norm_crop_to_ellipse,
    parse_base_ellipse,
)
LabelTarget = str  # "grasp" | "base"


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int


@dataclass
class TrainResult:
    model_path: Path
    train_samples: int
    val_samples: int
    best_val_px_error: float
    best_val_base_px_error: float | None
    epochs_run: int
    num_outputs: int
    loss_plot_path: Path | None = None
    loss_history_path: Path | None = None


class GraspPointNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_outputs: int = NUM_OUTPUTS_GRASP_BASE) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_outputs = int(num_outputs)
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_outputs),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def default_model_path(data_dir: Path) -> Path:
    return data_dir.resolve() / "model" / "grasp_point.pt"


def default_base_model_path(data_dir: Path) -> Path:
    return data_dir.resolve() / "model" / "base_point.pt"


def letterbox_rgb(
    color_rgb: np.ndarray,
    dst_w: int = INPUT_W,
    dst_h: int = INPUT_H,
) -> tuple[np.ndarray, LetterboxMeta]:
    src_h, src_w = color_rgb.shape[:2]
    scale = min(float(dst_w) / max(src_w, 1), float(dst_h) / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = np.asarray(Image.fromarray(color_rgb).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    pad_x = (dst_w - new_w) / 2.0
    pad_y = (dst_h - new_h) / 2.0
    x0 = int(round(pad_x))
    y0 = int(round(pad_y))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas, LetterboxMeta(scale=scale, pad_x=pad_x, pad_y=pad_y, src_w=src_w, src_h=src_h)


def letterbox_depth(
    depth_m: np.ndarray,
    meta: LetterboxMeta,
    dst_w: int = INPUT_W,
    dst_h: int = INPUT_H,
) -> np.ndarray:
    import cv2

    new_w = max(1, int(round(meta.src_w * meta.scale)))
    new_h = max(1, int(round(meta.src_h * meta.scale)))
    resized = cv2.resize(depth_m.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((dst_h, dst_w), dtype=np.float32)
    x0 = int(round(meta.pad_x))
    y0 = int(round(meta.pad_y))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def normalize_depth_map(
    depth_m: np.ndarray,
    *,
    depth_min_m: float = DEFAULT_DEPTH_MIN_M,
    depth_max_m: float = DEFAULT_DEPTH_MAX_M,
) -> np.ndarray:
    span = max(depth_max_m - depth_min_m, 1e-6)
    valid = depth_m > 1e-4
    out = np.zeros_like(depth_m, dtype=np.float32)
    out[valid] = np.clip((depth_m[valid] - depth_min_m) / span, 0.0, 1.0)
    return out


def uv_to_norm_letterbox(grasp_uv: tuple[int, int], meta: LetterboxMeta) -> tuple[float, float]:
    u, v = float(grasp_uv[0]), float(grasp_uv[1])
    u_net = u * meta.scale + meta.pad_x
    v_net = v * meta.scale + meta.pad_y
    return (
        u_net / max(INPUT_W - 1, 1),
        v_net / max(INPUT_H - 1, 1),
    )


def norm_to_uv_letterbox(norm_uv: tuple[float, float], meta: LetterboxMeta) -> tuple[int, int]:
    u_net = float(norm_uv[0]) * max(INPUT_W - 1, 1)
    v_net = float(norm_uv[1]) * max(INPUT_H - 1, 1)
    u = (u_net - meta.pad_x) / max(meta.scale, 1e-9)
    v = (v_net - meta.pad_y) / max(meta.scale, 1e-9)
    u = int(np.clip(int(round(u)), 0, meta.src_w - 1))
    v = int(np.clip(int(round(v)), 0, meta.src_h - 1))
    return u, v


def crop_region(
    arr: np.ndarray,
    bbox_xy: tuple[int, int, int, int],
    padding_px: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    x, y, bw, bh = bbox_xy
    pad = max(0, int(padding_px))
    h, w = arr.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    return arr[y0:y1, x0:x1].copy(), (x0, y0)


def bootstrap_bbox_xy(
    grasp_uv: tuple[int, int],
    width: int,
    height: int,
    *,
    half_w: int = 100,
    half_h: int = 80,
) -> tuple[int, int, int, int]:
    u, v = grasp_uv
    x0 = max(0, int(u) - half_w)
    y0 = max(0, int(v) - half_h)
    x1 = min(width - 1, int(u) + half_w)
    y1 = min(height - 1, int(v) + half_h)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def point_in_bbox(
    grasp_uv: tuple[int, int],
    bbox_xy: tuple[int, int, int, int],
    *,
    margin_px: int = 0,
) -> bool:
    u, v = grasp_uv
    x, y, bw, bh = bbox_xy
    m = max(0, int(margin_px))
    return (x - m) <= u <= (x + bw + m) and (y - m) <= v <= (y + bh + m)


def preprocess_input(
    color_rgb: np.ndarray,
    depth_m: np.ndarray | None,
    *,
    use_depth: bool,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, LetterboxMeta]:
    boxed_rgb, meta = letterbox_rgb(color_rgb, dst_w=INPUT_W, dst_h=INPUT_H)
    rgb = np.asarray(boxed_rgb, dtype=np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    channels = [rgb]

    if use_depth:
        if depth_m is None:
            depth_m = np.zeros(color_rgb.shape[:2], dtype=np.float32)
        boxed_depth = letterbox_depth(depth_m, meta, dst_w=INPUT_W, dst_h=INPUT_H)
        depth_norm = normalize_depth_map(
            boxed_depth,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        channels.append(depth_norm[:, :, np.newaxis])

    stacked = np.concatenate(channels, axis=2)
    tensor = np.transpose(stacked, (2, 0, 1)).astype(np.float32)
    return tensor, meta


def list_labeled_samples(
    data_dir: Path,
    *,
    require_base: bool = False,
    base_only: bool = False,
) -> list[dict[str, Any]]:
    labels_dir = data_dir / "labels"
    if not labels_dir.is_dir():
        return []
    samples: list[dict[str, Any]] = []
    for label_path in sorted(labels_dir.glob("*.json")):
        data = json.loads(label_path.read_text(encoding="utf-8"))
        image_rel = data.get("files", {}).get("image")
        if not image_rel:
            continue
        image_path = data_dir / str(image_rel)
        if not image_path.is_file():
            continue
        depth_rel = data.get("files", {}).get("depth")
        depth_path = data_dir / str(depth_rel) if depth_rel else data_dir / "depth" / f"{label_path.stem}.npy"
        w = int(data.get("width", 640))
        h = int(data.get("height", 480))

        base_uv: tuple[int, int] | None = None
        base_ellipse: BaseEllipse | None = None
        if "base_u" in data and "base_v" in data:
            base_uv = (int(data["base_u"]), int(data["base_v"]))
        elif "base_uv" in data and len(data["base_uv"]) >= 2:
            base_uv = (int(data["base_uv"][0]), int(data["base_uv"][1]))
        if not base_only:
            base_ellipse = parse_base_ellipse(data)
            if base_ellipse is not None:
                base_uv = base_ellipse.center_uv

        if require_base and base_uv is None:
            continue
        if base_only:
            if base_uv is None:
                continue
            grasp_uv = base_uv
        else:
            u = int(data.get("grasp_u", data.get("grasp_uv", [0, 0])[0]))
            v = int(data.get("grasp_v", data.get("grasp_uv", [0, 0])[1]))
            grasp_uv = (u, v)

        samples.append(
            {
                "id": data.get("id", label_path.stem),
                "label_path": label_path,
                "image_path": image_path,
                "depth_path": depth_path,
                "grasp_uv": grasp_uv,
                "base_uv": base_uv,
                "base_ellipse": base_ellipse if not base_only else None,
                "width": w,
                "height": h,
            }
        )
    return samples


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_depth_m(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.zeros((480, 640), dtype=np.float32)
    depth_raw = np.load(path)
    if np.issubdtype(depth_raw.dtype, np.floating):
        valid = depth_raw[np.isfinite(depth_raw) & (depth_raw > 0)]
        if valid.size and float(np.median(valid)) < 5.0:
            return depth_raw.astype(np.float32)
    return depth_raw.astype(np.float32) * 0.001


def _bbox_covering_uvs(
    points: list[tuple[int, int]],
    width: int,
    height: int,
    *,
    half_w: int,
    half_h: int,
) -> tuple[int, int, int, int]:
    us = [int(p[0]) for p in points]
    vs = [int(p[1]) for p in points]
    x0 = max(0, min(us) - half_w)
    y0 = max(0, min(vs) - half_h)
    x1 = min(width, max(us) + half_w + 1)
    y1 = min(height, max(vs) + half_h + 1)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _resolve_training_bbox(
    rgb: np.ndarray,
    grasp_uv: tuple[int, int],
    *,
    secondary_uv: tuple[int, int] | None = None,
    yolo_model_path: Path | None,
    yolo_class: str,
    yolo_confidence: float,
    bootstrap_half_w: int,
    bootstrap_half_h: int,
) -> tuple[int, int, int, int]:
    width = rgb.shape[1]
    height = rgb.shape[0]
    points = [grasp_uv]
    if secondary_uv is not None:
        points.append(secondary_uv)
    bbox = _bbox_covering_uvs(
        points,
        width,
        height,
        half_w=bootstrap_half_w,
        half_h=bootstrap_half_h,
    )

    if yolo_model_path is not None and yolo_model_path.is_file():
        try:
            from ..models import YoloConfig
            from .yolo_detect import detect_sensor_bbox

            yolo_cfg = YoloConfig(
                model_path=yolo_model_path,
                class_name=yolo_class,
                confidence=yolo_confidence,
            )
            det = detect_sensor_bbox(rgb, yolo_cfg)
            if all(point_in_bbox(p, det.bbox_xy, margin_px=8) for p in points):
                bbox = det.bbox_xy
        except Exception:
            pass

    return bbox


class GraspPointCropDataset(Dataset):
    """YOLO-aligned crops with RGB + depth; one target point per sample (grasp or base)."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        *,
        target: LabelTarget = "grasp",
        augment: bool = False,
        use_depth: bool = True,
        yolo_model_path: Path | None = None,
        yolo_class: str = "sensor",
        yolo_confidence: float = 0.25,
        crop_padding_px: int = 16,
        bootstrap_half_w: int = 100,
        bootstrap_half_h: int = 80,
        depth_min_m: float = DEFAULT_DEPTH_MIN_M,
        depth_max_m: float = DEFAULT_DEPTH_MAX_M,
        base_label_mode: str = "ellipse",
    ) -> None:
        if target not in ("grasp", "base"):
            raise ValueError(f"target must be 'grasp' or 'base', got {target!r}")
        if base_label_mode not in ("ellipse", "point"):
            raise ValueError(f"base_label_mode must be 'ellipse' or 'point', got {base_label_mode!r}")
        self.samples = samples
        self.target = target
        self.base_label_mode = base_label_mode
        self.augment = augment
        self.use_depth = use_depth
        self.yolo_model_path = yolo_model_path
        self.yolo_class = yolo_class
        self.yolo_confidence = yolo_confidence
        self.crop_padding_px = crop_padding_px
        self.bootstrap_half_w = bootstrap_half_w
        self.bootstrap_half_h = bootstrap_half_h
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self._crop_dims: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        sample = self.samples[idx]
        rgb = _load_rgb(sample["image_path"])
        depth_m = _load_depth_m(sample["depth_path"])
        grasp_uv = sample["grasp_uv"]
        base_uv = sample.get("base_uv")
        base_ellipse: BaseEllipse | None = sample.get("base_ellipse")
        if self.target == "base":
            if base_uv is None and base_ellipse is None:
                raise ValueError(f"Sample {sample.get('id')} missing base label")
            if self.base_label_mode == "point" or base_ellipse is None:
                if base_uv is None:
                    base_uv = base_ellipse.center_uv if base_ellipse is not None else grasp_uv
                label_uv = base_uv
                base_ellipse = None
            else:
                label_uv = base_ellipse.center_uv
        else:
            label_uv = grasp_uv

        bbox = _resolve_training_bbox(
            rgb,
            grasp_uv,
            secondary_uv=base_ellipse.center_uv if base_ellipse is not None else base_uv,
            yolo_model_path=self.yolo_model_path,
            yolo_class=self.yolo_class,
            yolo_confidence=self.yolo_confidence,
            bootstrap_half_w=self.bootstrap_half_w,
            bootstrap_half_h=self.bootstrap_half_h,
        )

        crop_rgb, origin = crop_region(rgb, bbox, self.crop_padding_px)
        crop_depth, _ = crop_region(depth_m, bbox, self.crop_padding_px)
        crop_h, crop_w = crop_rgb.shape[:2]

        u_crop = int(label_uv[0]) - origin[0]
        v_crop = int(label_uv[1]) - origin[1]
        u_crop = int(np.clip(u_crop, 0, crop_w - 1))
        v_crop = int(np.clip(v_crop, 0, crop_h - 1))

        crop_ellipse: BaseEllipse | None = None
        if self.target == "base" and self.base_label_mode == "ellipse" and base_ellipse is not None:
            ox, oy = origin
            crop_ellipse = BaseEllipse(
                center_uv=(
                    int(np.clip(base_ellipse.center_uv[0] - ox, 0, crop_w - 1)),
                    int(np.clip(base_ellipse.center_uv[1] - oy, 0, crop_h - 1)),
                ),
                axis_a_px=base_ellipse.axis_a_px,
                axis_b_px=base_ellipse.axis_b_px,
                angle_deg=base_ellipse.angle_deg,
            )
            u_crop, v_crop = crop_ellipse.center_uv

        if self.augment and random.random() < 0.5:
            crop_rgb = np.ascontiguousarray(crop_rgb[:, ::-1, :])
            crop_depth = np.ascontiguousarray(crop_depth[:, ::-1])
            if crop_ellipse is not None:
                cu, cv = crop_ellipse.center_uv
                crop_ellipse = BaseEllipse(
                    center_uv=(crop_w - 1 - cu, cv),
                    axis_a_px=crop_ellipse.axis_a_px,
                    axis_b_px=crop_ellipse.axis_b_px,
                    angle_deg=-crop_ellipse.angle_deg,
                )
                u_crop, v_crop = crop_ellipse.center_uv
            else:
                u_crop = crop_w - 1 - u_crop

        tensor, meta = preprocess_input(
            crop_rgb,
            crop_depth,
            use_depth=self.use_depth,
            depth_min_m=self.depth_min_m,
            depth_max_m=self.depth_max_m,
        )
        if crop_ellipse is not None:
            target = ellipse_to_norm_crop(crop_ellipse, (0, 0), (crop_h, crop_w), meta)
        else:
            target = np.array(uv_to_norm_letterbox((u_crop, v_crop), meta), dtype=np.float32)
        return torch.from_numpy(tensor), torch.from_numpy(target), (crop_w, crop_h)


class GraspPointDataset(Dataset):
    """Legacy full-frame RGB-only dataset."""

    def __init__(self, samples: list[dict[str, Any]], *, augment: bool = False) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        rgb = _load_rgb(sample["image_path"])
        width = int(sample["width"])
        height = int(sample["height"])
        u, v = sample["grasp_uv"]

        if self.augment and random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            u = width - 1 - u

        tensor, meta = preprocess_input(
            rgb,
            None,
            use_depth=False,
            depth_min_m=DEFAULT_DEPTH_MIN_M,
            depth_max_m=DEFAULT_DEPTH_MAX_M,
        )
        target = np.array(uv_to_norm_letterbox((u, v), meta), dtype=np.float32)
        return torch.from_numpy(tensor), torch.from_numpy(target)


def _split_samples(samples: list[dict[str, Any]], val_frac: float, seed: int) -> tuple[list, list]:
    if len(samples) < 2 or val_frac <= 0:
        return samples, []
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_frac)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def _eval_px_error_crop(
    model: GraspPointNet,
    loader: DataLoader,
    device: torch.device,
) -> float:
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    errors: list[float] = []
    with torch.no_grad():
        for batch_x, batch_y, dims in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            diff = (pred - batch_y).abs().cpu().numpy()
            crop_w = dims[0].numpy()
            crop_h = dims[1].numpy()
            for i in range(diff.shape[0]):
                errors.append(
                    float(
                        np.hypot(
                            diff[i, 0] * max(crop_w[i] - 1, 1),
                            diff[i, 1] * max(crop_h[i] - 1, 1),
                        )
                    )
                )
    return float(np.mean(errors)) if errors else float("nan")


def _eval_px_error_full(model: GraspPointNet, loader: DataLoader, device: torch.device) -> float:
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    errors: list[float] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            diff = (pred - batch_y).abs().cpu().numpy()
            for i in range(diff.shape[0]):
                px_err = float(np.hypot(diff[i, 0] * (640 - 1), diff[i, 1] * (480 - 1)))
                errors.append(px_err)
    return float(np.mean(errors)) if errors else float("nan")


def _training_run_dir(model_path: Path, target: LabelTarget) -> Path:
    run_name = "grasp_point" if target == "grasp" else "base_point"
    return model_path.parent / "runs" / run_name


def _save_training_curve(
    run_dir: Path,
    *,
    target: LabelTarget,
    train_history: list[dict[str, float | int]],
    val_history: list[dict[str, float | int]],
) -> tuple[Path | None, Path | None]:
    """Write per-iteration train loss (+ val px error) as JSON and PNG."""
    if not train_history:
        return None, None

    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "training_loss.json"
    payload = {
        "target": target,
        "train_loss": train_history,
        "val_px_error": val_history,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    png_path: Path | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return json_path, None

    train_iters = [int(h["iteration"]) for h in train_history]
    train_loss = [float(h["loss"]) for h in train_history]
    title = "Grasp point CNN" if target == "grasp" else "Base ellipse CNN"

    if val_history:
        fig, (ax_loss, ax_val) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    else:
        fig, ax_loss = plt.subplots(1, 1, figsize=(8, 4))
        ax_val = None

    fig.suptitle(title)
    ax_loss.plot(train_iters, train_loss, color="#1f77b4", linewidth=0.8, alpha=0.65, label="train batch")
    if len(train_loss) >= 8:
        window = min(16, max(4, len(train_loss) // 20))
        kernel = np.ones(window, dtype=np.float64) / window
        smooth = np.convolve(train_loss, kernel, mode="valid")
        smooth_iters = train_iters[window - 1 :]
        ax_loss.plot(smooth_iters, smooth, color="#d62728", linewidth=1.5, label=f"{window}-batch avg")
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_xlabel("iteration")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(True, alpha=0.25)

    if ax_val is not None and val_history:
        val_epochs = [int(h["epoch"]) for h in val_history]
        val_err = [float(h["val_px_error"]) for h in val_history]
        ax_val.plot(val_epochs, val_err, "o-", color="#2ca02c", label="val px error")
        ax_val.set_ylabel("val px error")
        ax_val.set_xlabel("epoch")
        ax_val.legend(loc="upper right", fontsize=8)
        ax_val.grid(True, alpha=0.25)

    fig.tight_layout()
    png_path = run_dir / "training_loss.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return json_path, png_path


def train_point_model(
    data_dir: Path,
    *,
    model_path: Path | None = None,
    target: LabelTarget = "grasp",
    epochs: int = 120,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    seed: int = 0,
    device: str | None = None,
    crop_mode: bool = True,
    use_depth: bool = True,
    yolo_model_path: Path | None = None,
    yolo_class: str = "sensor",
    yolo_confidence: float = 0.25,
    crop_padding_px: int = 16,
    bootstrap_half_w: int = 100,
    bootstrap_half_h: int = 80,
    depth_min_m: float = DEFAULT_DEPTH_MIN_M,
    depth_max_m: float = DEFAULT_DEPTH_MAX_M,
    base_label_mode: str = "ellipse",
    base_only: bool = False,
) -> TrainResult:
    if target not in ("grasp", "base"):
        raise ValueError(f"target must be 'grasp' or 'base', got {target!r}")
    if base_label_mode not in ("ellipse", "point"):
        raise ValueError(f"base_label_mode must be 'ellipse' or 'point', got {base_label_mode!r}")
    data_dir = data_dir.resolve()
    if model_path is None:
        model_path = default_base_model_path(data_dir) if target == "base" else default_model_path(data_dir)
    model_path = model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    require_base = target == "base"
    samples = list_labeled_samples(data_dir, require_base=require_base, base_only=base_only)
    if len(samples) < 2:
        if target == "base":
            total = len(list_labeled_samples(data_dir))
            raise ValueError(
                f"Need at least 2 samples with base labels in {data_dir}, "
                f"found {len(samples)} ({total} total grasp labels). "
                "Label bases with live_label or relabel."
            )
        raise ValueError(f"Need at least 2 labeled samples in {data_dir}, found {len(samples)}")

    train_samples, val_samples = _split_samples(samples, val_frac, seed)
    in_channels = 4 if use_depth else 3
    use_ellipse_base = target == "base" and base_label_mode == "ellipse"
    num_outputs = NUM_OUTPUTS_BASE_ELLIPSE if use_ellipse_base else NUM_OUTPUTS_GRASP_ONLY

    crop_ds_kw = dict(
        target=target,
        use_depth=use_depth,
        yolo_model_path=yolo_model_path,
        yolo_class=yolo_class,
        yolo_confidence=yolo_confidence,
        crop_padding_px=crop_padding_px,
        bootstrap_half_w=bootstrap_half_w,
        bootstrap_half_h=bootstrap_half_h,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        base_label_mode=base_label_mode,
    )

    if crop_mode:
        train_ds = GraspPointCropDataset(train_samples, augment=True, **crop_ds_kw)
        val_ds = (
            GraspPointCropDataset(val_samples, augment=False, **crop_ds_kw)
            if val_samples
            else None
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
        eval_fn = _eval_px_error_crop
    else:
        if target == "base":
            raise ValueError("Base model training requires crop_mode=True (YOLO-aligned crops).")
        train_ds = GraspPointDataset(train_samples, augment=True)
        val_ds = GraspPointDataset(val_samples, augment=False) if val_samples else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
        eval_fn = _eval_px_error_full

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GraspPointNet(in_channels=in_channels, num_outputs=num_outputs).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_run = 0
    iteration = 0
    train_history: list[dict[str, float | int]] = []
    val_history: list[dict[str, float | int]] = []
    run_dir = _training_run_dir(model_path, target)
    label = "grasp" if target == "grasp" else "base"

    for epoch in range(epochs):
        epochs_run += 1
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch_x, batch_y = batch[0], batch[1]
            batch_x = batch_x.to(torch_device)
            batch_y = batch_y.to(torch_device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            iteration += 1
            loss_val = float(loss.item())
            epoch_losses.append(loss_val)
            train_history.append(
                {"iteration": iteration, "epoch": epoch + 1, "loss": loss_val},
            )

        avg_train = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_err = float("nan")
        if val_loader is not None:
            val_err = eval_fn(model, val_loader, torch_device)
            val_history.append(
                {"epoch": epoch + 1, "iteration": iteration, "val_px_error": val_err},
            )
            if val_err < best_val:
                best_val = val_err
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"  [{label}] epoch {epoch + 1}/{epochs}  iter {iteration}  "
                f"train_loss={avg_train:.6f}  val_px={val_err:.2f}",
                flush=True,
            )
        elif best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"  [{label}] epoch {epoch + 1}/{epochs}  iter {iteration}  "
                f"train_loss={avg_train:.6f}",
                flush=True,
            )

    loss_history_path, loss_plot_path = _save_training_curve(
        run_dir,
        target=target,
        train_history=train_history,
        val_history=val_history,
    )
    if loss_plot_path is not None:
        print(f"  [{label}] loss plot: {loss_plot_path}", flush=True)
    elif loss_history_path is not None:
        print(f"  [{label}] loss history: {loss_history_path}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    best_val_px = None if best_val == float("inf") else best_val
    meta = {
        "input_w": INPUT_W,
        "input_h": INPUT_H,
        "in_channels": in_channels,
        "num_outputs": num_outputs,
        "use_depth": use_depth,
        "crop_mode": crop_mode,
        "target": (
            "base_ellipse"
            if use_ellipse_base
            else ("base_point" if target == "base" else target)
        ),
        "base_label_mode": base_label_mode if target == "base" else None,
        "predict_base": False,
        "depth_min_m": depth_min_m,
        "depth_max_m": depth_max_m,
        "crop_padding_px": crop_padding_px,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_px_error": best_val_px,
        "best_val_grasp_px_error": best_val_px if target == "grasp" else None,
        "best_val_base_px_error": best_val_px if target == "base" else None,
        "epochs": epochs_run,
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, model_path)

    return TrainResult(
        model_path=model_path,
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        best_val_px_error=best_val_px or float("nan"),
        best_val_base_px_error=best_val_px if target == "base" else None,
        epochs_run=epochs_run,
        num_outputs=num_outputs,
        loss_plot_path=loss_plot_path,
        loss_history_path=loss_history_path,
    )


def _infer_num_outputs(meta: dict[str, Any], state_dict: dict[str, Any]) -> int:
    if "num_outputs" in meta:
        return int(meta["num_outputs"])
    if "predict_base" in meta:
        return NUM_OUTPUTS_GRASP_BASE if meta["predict_base"] else NUM_OUTPUTS_GRASP_ONLY
    weight = state_dict.get("head.4.weight")
    if weight is not None:
        return int(weight.shape[0])
    return NUM_OUTPUTS_GRASP_ONLY


def load_grasp_model(model_path: Path, *, device: str | None = None) -> tuple[GraspPointNet, dict[str, Any]]:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(model_path, map_location=torch_device, weights_only=False)
    meta = payload.get("meta", {})
    state_dict = payload["state_dict"]
    in_channels = int(meta.get("in_channels", 3 if not meta.get("use_depth") else 4))
    num_outputs = _infer_num_outputs(meta, state_dict)
    model = GraspPointNet(in_channels=in_channels, num_outputs=num_outputs).to(torch_device)
    model.load_state_dict(state_dict)
    meta["num_outputs"] = num_outputs
    meta["predict_base"] = num_outputs >= NUM_OUTPUTS_GRASP_BASE
    model.eval()
    return model, meta


def _predict_norm(model: GraspPointNet, color_rgb: np.ndarray, depth_m: np.ndarray | None, **kwargs) -> np.ndarray:
    torch_device = torch.device(kwargs.get("device") or next(model.parameters()).device)
    model = model.to(torch_device)
    model.eval()

    meta_cfg = kwargs.get("model_meta") or {}
    use_depth = bool(meta_cfg.get("use_depth", depth_m is not None))
    if depth_m is None:
        depth_m = np.zeros(color_rgb.shape[:2], dtype=np.float32)

    tensor, _letterbox = preprocess_input(
        color_rgb,
        depth_m,
        use_depth=use_depth,
        depth_min_m=float(meta_cfg.get("depth_min_m", kwargs.get("depth_min_m", DEFAULT_DEPTH_MIN_M))),
        depth_max_m=float(meta_cfg.get("depth_max_m", kwargs.get("depth_max_m", DEFAULT_DEPTH_MAX_M))),
    )
    x = torch.from_numpy(tensor).unsqueeze(0).to(torch_device)
    return model(x)[0].cpu().numpy(), _letterbox


def is_legacy_combined_model(meta: dict[str, Any]) -> bool:
    """True for old single-net models that output grasp+base together."""
    if meta.get("target") in ("grasp", "base", "base_ellipse"):
        return False
    return int(meta.get("num_outputs", NUM_OUTPUTS_GRASP_ONLY)) >= NUM_OUTPUTS_GRASP_BASE or bool(
        meta.get("predict_base")
    )


@torch.no_grad()
def predict_base_ellipse(
    model: GraspPointNet,
    color_rgb: np.ndarray,
    depth_m: np.ndarray | None = None,
    *,
    model_meta: dict[str, Any] | None = None,
    device: str | None = None,
    depth_min_m: float = DEFAULT_DEPTH_MIN_M,
    depth_max_m: float = DEFAULT_DEPTH_MAX_M,
) -> BaseEllipse:
    """Predict elliptical base counter in crop pixel coordinates."""
    pred, letterbox = _predict_norm(
        model,
        color_rgb,
        depth_m,
        model_meta=model_meta,
        device=device,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    crop_h, crop_w = color_rgb.shape[:2]
    if len(pred) >= NUM_OUTPUTS_BASE_ELLIPSE:
        return norm_crop_to_ellipse(pred, (0, 0), (crop_h, crop_w), letterbox)
    center = predict_grasp_uv(
        model,
        color_rgb,
        depth_m,
        model_meta=model_meta,
        device=device,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    return BaseEllipse.from_center_uv(center)


@torch.no_grad()
def predict_grasp_base_on_crop(
    crop_rgb: np.ndarray,
    crop_depth: np.ndarray,
    *,
    grasp_model_path: Path,
    base_model_path: Path | None = None,
    depth_min_m: float = DEFAULT_DEPTH_MIN_M,
    depth_max_m: float = DEFAULT_DEPTH_MAX_M,
) -> tuple[tuple[int, int], BaseEllipse | None, str]:
    """
    Predict grasp and base ellipse in a YOLO crop.

    Base is an ellipse (center + axes + angle). Legacy 2-output base models
    return a default-size ellipse centered on the predicted point.
    """
    grasp_model, grasp_meta = load_grasp_model(grasp_model_path)
    infer_kw = {
        "depth_min_m": depth_min_m,
        "depth_max_m": depth_max_m,
    }

    base_path = base_model_path.resolve() if base_model_path else None
    if base_path is not None and base_path.is_file():
        grasp_uv = predict_grasp_uv(
            grasp_model,
            crop_rgb,
            crop_depth,
            model_meta=grasp_meta,
            **infer_kw,
        )
        base_model, base_meta = load_grasp_model(base_path)
        base_ellipse = predict_base_ellipse(
            base_model,
            crop_rgb,
            crop_depth,
            model_meta=base_meta,
            **infer_kw,
        )
        source = (
            "base_ellipse_net"
            if int(base_meta.get("num_outputs", 0)) >= NUM_OUTPUTS_BASE_ELLIPSE
            else "base_point_net"
        )
        return grasp_uv, base_ellipse, source

    if is_legacy_combined_model(grasp_meta):
        grasp_uv, base_uv = predict_grasp_and_base_uv(
            grasp_model,
            crop_rgb,
            crop_depth,
            model_meta=grasp_meta,
            **infer_kw,
        )
        base_ellipse = (
            BaseEllipse.from_center_uv(base_uv) if base_uv is not None else None
        )
        return grasp_uv, base_ellipse, "legacy_combined"

    grasp_uv = predict_grasp_uv(
        grasp_model,
        crop_rgb,
        crop_depth,
        model_meta=grasp_meta,
        **infer_kw,
    )
    return grasp_uv, None, "grasp_only"


@torch.no_grad()
def predict_grasp_uv(
    model: GraspPointNet,
    color_rgb: np.ndarray,
    depth_m: np.ndarray | None = None,
    *,
    model_meta: dict[str, Any] | None = None,
    device: str | None = None,
    depth_min_m: float = DEFAULT_DEPTH_MIN_M,
    depth_max_m: float = DEFAULT_DEPTH_MAX_M,
) -> tuple[int, int]:
    pred, letterbox = _predict_norm(
        model,
        color_rgb,
        depth_m,
        model_meta=model_meta,
        device=device,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    return norm_to_uv_letterbox((float(pred[0]), float(pred[1])), letterbox)


@torch.no_grad()
def predict_grasp_and_base_uv(
    model: GraspPointNet,
    color_rgb: np.ndarray,
    depth_m: np.ndarray | None = None,
    *,
    model_meta: dict[str, Any] | None = None,
    device: str | None = None,
    depth_min_m: float = DEFAULT_DEPTH_MIN_M,
    depth_max_m: float = DEFAULT_DEPTH_MAX_M,
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    pred, letterbox = _predict_norm(
        model,
        color_rgb,
        depth_m,
        model_meta=model_meta,
        device=device,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    grasp_uv = norm_to_uv_letterbox((float(pred[0]), float(pred[1])), letterbox)
    base_uv = None
    if len(pred) >= NUM_OUTPUTS_GRASP_BASE:
        base_uv = norm_to_uv_letterbox((float(pred[2]), float(pred[3])), letterbox)
    return grasp_uv, base_uv
