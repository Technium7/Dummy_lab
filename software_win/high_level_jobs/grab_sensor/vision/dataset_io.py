"""Dataset IO helpers for multi-object top-view labels."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CLASS_NAMES = ("sensor", "post", "other")
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

# Auto-bbox margins when labeling keypoints only (used for YOLO export / crop train).
# Larger pad so sensor crops include holder arm + body (grasp+dir schema).
SENSOR_BBOX_PAD_PX = 72
POST_BBOX_HALF_PX = 28  # tighter — posts are ~1" cylinders in top view


def bbox_from_keypoints(
    kpts: list,
    *,
    class_name: str,
    width: int,
    height: int,
    sensor_pad_px: int = SENSOR_BBOX_PAD_PX,
    post_half_px: int = POST_BBOX_HALF_PX,
) -> list[float]:
    """Build a bbox from keypoints (no manual box drawing)."""
    pts = [(float(p[0]), float(p[1])) for p in kpts]
    if not pts:
        raise ValueError("bbox_from_keypoints requires at least one keypoint")
    cls = str(class_name).lower()
    if cls == "sensor":
        # Use all provided points (prefer 5: rect + holder)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = float(sensor_pad_px)
        x0 = min(xs) - pad
        y0 = min(ys) - pad
        x1 = max(xs) + pad
        y1 = max(ys) + pad
    else:
        cx, cy = pts[0]
        half = float(post_half_px)
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    x0 = float(np.clip(x0, 0, max(0, width - 1)))
    y0 = float(np.clip(y0, 0, max(0, height - 1)))
    x1 = float(np.clip(x1, 0, max(0, width - 1)))
    y1 = float(np.clip(y1, 0, max(0, height - 1)))
    if x1 <= x0:
        x1 = min(width - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(height - 1, y0 + 1)
    return [x0, y0, x1, y1]


def next_sample_id(out_dir: Path) -> str:
    return time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"


def ensure_dataset_dirs(out_dir: Path) -> None:
    for name in ("images", "depth", "labels", "previews", "model"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def save_labeled_sample(
    out_dir: Path,
    *,
    color_rgb: np.ndarray,
    depth_m: np.ndarray | None,
    objects: list[dict[str, Any]],
    intrinsics: dict[str, Any] | None = None,
    sample_id: str | None = None,
) -> dict[str, Any]:
    ensure_dataset_dirs(out_dir)
    sid = sample_id or next_sample_id(out_dir)
    img_rel = f"images/{sid}.png"
    depth_rel = f"depth/{sid}.npy"
    label_rel = f"labels/{sid}.json"

    Image.fromarray(color_rgb.astype(np.uint8)).save(out_dir / img_rel)
    if depth_m is not None:
        np.save(out_dir / depth_rel, depth_m.astype(np.float32))

    h, w = color_rgb.shape[:2]
    record = {
        "id": sid,
        "width": int(w),
        "height": int(h),
        "objects": objects,
        "intrinsics": intrinsics or {},
        "files": {"image": img_rel, "depth": depth_rel if depth_m is not None else None},
    }
    (out_dir / label_rel).write_text(json.dumps(record, indent=2), encoding="utf-8")

    manifest = out_dir / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": sid, "label": label_rel}) + "\n")
    return record


def load_label(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_labels(out_dir: Path) -> list[Path]:
    labels_dir = out_dir / "labels"
    if not labels_dir.is_dir():
        return []
    return sorted(labels_dir.glob("*.json"))


def delete_sample(out_dir: Path, label_path: Path) -> str:
    """Move sample files to ``out_dir/.trash/<id>/`` (recoverable) instead of permanent delete."""
    data = load_label(label_path)
    sid = str(data.get("id", label_path.stem))
    trash_root = out_dir / ".trash" / sid
    trash_root.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []

    def _move(src: Path, dest_name: str | None = None) -> None:
        if not src.is_file():
            return
        dst = trash_root / (dest_name or src.name)
        if dst.exists():
            dst.unlink()
        src.replace(dst)
        moved.append(str(dst.relative_to(out_dir)))

    for key in ("image", "depth"):
        rel = (data.get("files") or {}).get(key)
        if rel:
            _move(out_dir / rel)
    _move(out_dir / "previews" / f"{sid}.png")
    # keep label JSON in trash for restore
    label_dst = trash_root / f"{sid}.json"
    if label_dst.exists():
        label_dst.unlink()
    label_path.replace(label_dst)
    moved.append(str(label_dst.relative_to(out_dir)))

    manifest = out_dir / "manifest.jsonl"
    if manifest.is_file():
        lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
        kept = []
        for ln in lines:
            try:
                if json.loads(ln).get("id") != sid:
                    kept.append(ln)
            except json.JSONDecodeError:
                kept.append(ln)
        manifest.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return f"Moved sample {sid} to .trash/{sid}/ (not permanent). Restore with restore_sample_from_trash."


def restore_sample_from_trash(out_dir: Path, sample_id: str) -> str:
    """Restore a sample previously moved by ``delete_sample``."""
    trash = out_dir / ".trash" / sample_id
    if not trash.is_dir():
        return f"No trash folder for {sample_id}"
    label_src = trash / f"{sample_id}.json"
    if not label_src.is_file():
        return f"Missing label in trash for {sample_id}"
    data = load_label(label_src)
    ensure_dataset_dirs(out_dir)

    def _restore(name: str, dest_rel: str) -> None:
        src = trash / name
        if not src.is_file():
            return
        dst = out_dir / dest_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        src.replace(dst)

    files = data.get("files") or {}
    if files.get("image"):
        _restore(Path(files["image"]).name, files["image"])
    if files.get("depth"):
        _restore(Path(files["depth"]).name, files["depth"])
    _restore(f"{sample_id}.png", f"previews/{sample_id}.png")
    label_dst = out_dir / "labels" / f"{sample_id}.json"
    if label_dst.exists():
        label_dst.unlink()
    label_src.replace(label_dst)
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": sample_id, "label": f"labels/{sample_id}.json"}) + "\n")
    try:
        trash.rmdir()
    except OSError:
        pass
    return f"Restored sample {sample_id}"


def update_sample_objects(
    out_dir: Path,
    label_path: Path,
    objects: list[dict[str, Any]],
    *,
    color_rgb: np.ndarray | None = None,
) -> dict[str, Any]:
    """Overwrite objects on an existing label (keep image/depth/id)."""
    data = load_label(label_path)
    data["objects"] = objects
    if color_rgb is not None:
        h, w = color_rgb.shape[:2]
        data["width"] = int(w)
        data["height"] = int(h)
    label_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
