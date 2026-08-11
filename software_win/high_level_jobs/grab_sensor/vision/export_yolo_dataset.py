#!/usr/bin/env python3
"""Export multi-class YOLO dataset from keypoint labels (auto-bboxes)."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.grab_sensor.paths import DATASET_DIR, MODEL_DIR  # noqa: E402
from high_level_jobs.grab_sensor.vision.dataset_io import (  # noqa: E402
    CLASS_NAMES,
    CLASS_TO_IDX,
    bbox_from_keypoints,
    iter_labels,
    load_label,
)

DEFAULT_YOLO_EXPORT = DATASET_DIR / "sensor_yolo"


def _xyxy_to_yolo(
    bbox: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    cx = (x0 + x1) * 0.5 / max(width, 1)
    cy = (y0 + y1) * 0.5 / max(height, 1)
    return cx, cy, bw / max(width, 1), bh / max(height, 1)


def export_yolo_dataset(
    data_dir: Path,
    out_dir: Path,
    *,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Path:
    data_dir = data_dir.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    labels = iter_labels(data_dir)
    if len(labels) < 2:
        raise ValueError(f"Need at least 2 labeled frames in {data_dir}, found {len(labels)}")

    samples = [load_label(p) for p in labels]
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_frac)))
    val_count = min(val_count, len(shuffled) - 1)
    val_ids = {s["id"] for s in shuffled[:val_count]}

    n_obj = 0
    for sample in shuffled:
        split = "val" if sample["id"] in val_ids else "train"
        sid = sample["id"]
        img_rel = (sample.get("files") or {}).get("image")
        if not img_rel:
            continue
        src = data_dir / img_rel
        if not src.is_file():
            continue
        dst_img = out_dir / "images" / split / f"{sid}.png"
        shutil.copy2(src, dst_img)

        w = int(sample.get("width") or Image.open(src).size[0])
        h = int(sample.get("height") or Image.open(src).size[1])
        lines: list[str] = []
        for obj in sample.get("objects") or []:
            cls = str(obj.get("class", "other")).lower()
            if cls not in CLASS_TO_IDX:
                continue
            kpts = obj.get("kpts") or []
            bbox = obj.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                if not kpts:
                    continue
                bbox = bbox_from_keypoints(kpts, class_name=cls, width=w, height=h)
            cx, cy, nw, nh = _xyxy_to_yolo(bbox, w, h)
            lines.append(f"{CLASS_TO_IDX[cls]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_obj += 1
        (out_dir / "labels" / split / f"{sid}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    data_yaml = {
        "path": str(out_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"Exported {len(samples)} frames / {n_obj} objects -> {out_dir}")
    print(f"  train={len(samples) - val_count}  val={val_count}  classes={list(CLASS_NAMES)}")
    return yaml_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export YOLO dataset from grab_sensor keypoint labels")
    p.add_argument("--data", type=Path, default=DATASET_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_YOLO_EXPORT)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    export_yolo_dataset(args.data, args.out, val_frac=args.val_frac, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
