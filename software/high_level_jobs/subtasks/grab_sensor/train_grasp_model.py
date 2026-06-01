#!/usr/bin/env python3
"""
Train the grasp-point CNN on labeled samples.

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.train_grasp_model
  python -m high_level_jobs.subtasks.grab_sensor.train_grasp_model --epochs 150 --crop --depth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .grasp_point_net import default_model_path, list_labeled_samples, train_grasp_model


def _default_yolo_model() -> Path:
    return Path(__file__).resolve().parent / "dataset" / "grasp_keypoints" / "model" / "sensor_yolo.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent / "dataset" / "grasp_keypoints"
    p = argparse.ArgumentParser(description="Train grasp-point CNN (crop + RGBD by default).")
    p.add_argument("--data-dir", type=Path, default=default_data, help="Labeled dataset directory")
    p.add_argument("--model-path", type=Path, default=None, help="Output .pt path")
    p.add_argument("--yolo-model", type=Path, default=_default_yolo_model(), help="YOLO weights for training crops")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True, help="Train on YOLO-aligned crops")
    p.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True, help="Include depth channel")
    p.add_argument("--crop-padding", type=int, default=16)
    p.add_argument("--depth-min-m", type=float, default=0.25)
    p.add_argument("--depth-max-m", type=float, default=0.85)
    p.add_argument(
        "--review",
        action="store_true",
        help="Open live review GUI after training (confirm / reset loop)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    model_path = (args.model_path or default_model_path(data_dir)).resolve()
    yolo_model = args.yolo_model.resolve() if args.crop and args.yolo_model.is_file() else None

    count = len(list_labeled_samples(data_dir))
    if count < 2:
        print(f"Need at least 2 labeled samples in {data_dir}, found {count}.")
        print("Collect data first: python -m high_level_jobs.subtasks.grab_sensor.label_grasp")
        return 1

    mode = "crop+RGBD" if args.crop and args.depth else ("crop" if args.crop else "full RGB")
    print(f"Training ({mode}) on {count} samples from {data_dir}")
    if args.crop and yolo_model:
        print(f"  YOLO crop model: {yolo_model}")

    result = train_grasp_model(
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
    print(
        f"Train={result.train_samples}  Val={result.val_samples}  "
        f"Val crop error ~ {result.best_val_px_error:.1f}px  Epochs={result.epochs_run}"
    )

    if args.review:
        from .review_grasp import run_review_gui

        return run_review_gui(data_dir=data_dir, model_path=model_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
