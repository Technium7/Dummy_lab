#!/usr/bin/env python3
"""
Train the base-point CNN on labeled samples (separate from grasp net).

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.train_base_model
  python -m high_level_jobs.subtasks.grab_sensor.detection.train_base_model --epochs 120 --crop --depth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .grasp_point_net import default_base_model_path, list_labeled_samples, train_base_model


def _default_yolo_model() -> Path:
    return Path(__file__).resolve().parent / "dataset" / "grasp_keypoints" / "model" / "sensor_yolo.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent / "dataset" / "grasp_keypoints"
    p = argparse.ArgumentParser(description="Train base-point CNN (YOLO crop + RGBD).")
    p.add_argument("--data-dir", type=Path, default=default_data)
    p.add_argument("--model-path", type=Path, default=None, help="Output base_point.pt path")
    p.add_argument("--yolo-model", type=Path, default=_default_yolo_model())
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--crop-padding", type=int, default=16)
    p.add_argument("--depth-min-m", type=float, default=0.25)
    p.add_argument("--depth-max-m", type=float, default=0.85)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    model_path = (args.model_path or default_base_model_path(data_dir)).resolve()
    yolo_model = args.yolo_model.resolve() if args.crop and args.yolo_model.is_file() else None

    count = len(list_labeled_samples(data_dir, require_base=True))
    if count < 2:
        total = len(list_labeled_samples(data_dir))
        print(f"Need at least 2 samples with base labels in {data_dir}, found {count} ({total} grasp labels).")
        print("Label bases with: python -m high_level_jobs.subtasks.grab_sensor.detection.live_label")
        return 1

    print(f"Training base net (crop+RGBD) on {count} samples from {data_dir}")
    if args.crop and yolo_model:
        print(f"  YOLO crop model: {yolo_model}")

    result = train_base_model(
        data_dir,
        model_path=model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        crop_mode=args.crop,
        use_depth=args.depth,
        yolo_model_path=yolo_model,
        crop_padding_px=args.crop_padding,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
    )
    print(f"Saved model to {result.model_path}")
    err = result.best_val_base_px_error or result.best_val_px_error
    print(
        f"Train={result.train_samples}  Val={result.val_samples}  "
        f"Val base err ~ {err:.1f}px  Epochs={result.epochs_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
