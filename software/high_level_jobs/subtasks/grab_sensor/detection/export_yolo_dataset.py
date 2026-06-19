#!/usr/bin/env python3
"""
Bootstrap YOLO bbox labels from grasp-point dataset (box centered on each grasp click).

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.export_yolo_dataset
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

from .grasp_point_net import list_labeled_samples


def _grasp_to_yolo_bbox(
    grasp_uv: tuple[int, int],
    width: int,
    height: int,
    *,
    half_w: int,
    half_h: int,
) -> tuple[float, float, float, float]:
    u, v = grasp_uv
    x0 = max(0, u - half_w)
    y0 = max(0, v - half_h)
    x1 = min(width - 1, u + half_w)
    y1 = min(height - 1, v + half_h)
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = (x0 + x1) / 2.0 / max(width, 1)
    cy = (y0 + y1) / 2.0 / max(height, 1)
    nw = bw / max(width, 1)
    nh = bh / max(height, 1)
    return cx, cy, nw, nh


def export_yolo_dataset(
    grasp_data_dir: Path,
    out_dir: Path,
    *,
    class_name: str = "sensor",
    bbox_half_w: int = 100,
    bbox_half_h: int = 80,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Path:
    grasp_data_dir = grasp_data_dir.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    samples = list_labeled_samples(grasp_data_dir)
    if len(samples) < 2:
        raise ValueError(f"Need at least 2 grasp labels in {grasp_data_dir}, found {len(samples)}")

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_frac)))
    val_count = min(val_count, len(shuffled) - 1)
    val_ids = {s["id"] for s in shuffled[:val_count]}

    for sample in shuffled:
        split = "val" if sample["id"] in val_ids else "train"
        sid = sample["id"]
        src = sample["image_path"]
        dst_img = out_dir / "images" / split / f"{sid}.png"
        shutil.copy2(src, dst_img)

        cx, cy, nw, nh = _grasp_to_yolo_bbox(
            sample["grasp_uv"],
            sample["width"],
            sample["height"],
            half_w=bbox_half_w,
            half_h=bbox_half_h,
        )
        label_path = out_dir / "labels" / split / f"{sid}.txt"
        label_path.write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n", encoding="utf-8")

    data_yaml = {
        "path": str(out_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"Exported {len(samples)} samples to {out_dir}")
    print(f"  train={len(samples) - val_count}  val={val_count}")
    return yaml_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_grasp = Path(__file__).resolve().parent / "dataset" / "grasp_keypoints"
    default_out = Path(__file__).resolve().parent / "dataset" / "sensor_yolo"
    p = argparse.ArgumentParser(description="Export YOLO dataset from grasp keypoint labels.")
    p.add_argument("--grasp-data-dir", type=Path, default=default_grasp)
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument("--class-name", type=str, default="sensor")
    p.add_argument("--bbox-half-w", type=int, default=100)
    p.add_argument("--bbox-half-h", type=int, default=80)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    export_yolo_dataset(
        args.grasp_data_dir.resolve(),
        args.out_dir.resolve(),
        class_name=args.class_name.strip(),
        bbox_half_w=args.bbox_half_w,
        bbox_half_h=args.bbox_half_h,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
