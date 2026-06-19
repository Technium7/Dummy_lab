#!/usr/bin/env python3
"""
Train grasp + base CNNs (separate networks) on labeled samples.

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.train_grasp_model
  python -m high_level_jobs.subtasks.grab_sensor.detection.train_grasp_model --epochs 150 --crop --depth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .grasp_point_net import (
    default_base_model_path,
    default_model_path,
    list_labeled_samples,
    train_base_model,
    train_grasp_model,
)


def _default_yolo_model() -> Path:
    return Path(__file__).resolve().parent / "dataset" / "grasp_keypoints" / "model" / "sensor_yolo.pt"


def _train_kwargs(args: argparse.Namespace, yolo_model: Path | None) -> dict:
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "crop_mode": args.crop,
        "use_depth": args.depth,
        "yolo_model_path": yolo_model,
        "crop_padding_px": args.crop_padding,
        "depth_min_m": args.depth_min_m,
        "depth_max_m": args.depth_max_m,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent / "dataset" / "grasp_keypoints"
    p = argparse.ArgumentParser(description="Train grasp + base CNNs (separate nets, crop + RGBD).")
    p.add_argument("--data-dir", type=Path, default=default_data, help="Labeled dataset directory")
    p.add_argument("--model-path", type=Path, default=None, help="Grasp .pt output path")
    p.add_argument("--base-model-path", type=Path, default=None, help="Base .pt output path")
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
        "--grasp-only",
        action="store_true",
        help="Train grasp net only (skip separate base net)",
    )
    p.add_argument(
        "--review",
        action="store_true",
        help="Open live review GUI after training",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    grasp_path = (args.model_path or default_model_path(data_dir)).resolve()
    base_path = (args.base_model_path or default_base_model_path(data_dir)).resolve()
    yolo_model = args.yolo_model.resolve() if args.crop and args.yolo_model.is_file() else None
    kw = _train_kwargs(args, yolo_model)

    grasp_count = len(list_labeled_samples(data_dir))
    base_count = len(list_labeled_samples(data_dir, require_base=True))
    if grasp_count < 2:
        print(f"Need at least 2 labeled grasp samples in {data_dir}, found {grasp_count}.")
        print("Collect data first: python -m high_level_jobs.subtasks.grab_sensor.detection.live_label")
        return 1
    if not args.grasp_only and base_count < 2:
        print(f"Need at least 2 samples with base labels for base net, found {base_count}.")
        print("Label with live_label or use --grasp-only to skip base training.")
        return 1

    mode = "crop+RGBD" if args.crop and args.depth else ("crop" if args.crop else "full RGB")
    print(f"Training grasp net ({mode}) on {grasp_count} samples from {data_dir}")
    if args.crop and yolo_model:
        print(f"  YOLO crop model: {yolo_model}")

    grasp_result = train_grasp_model(data_dir, model_path=grasp_path, target="grasp", **kw)
    print(f"Saved grasp model to {grasp_result.model_path}")
    print(
        f"  grasp val err ~ {grasp_result.best_val_px_error:.1f}px  "
        f"Epochs={grasp_result.epochs_run}"
    )

    if not args.grasp_only:
        print(f"Training base net ({mode}) on {base_count} samples")
        base_result = train_base_model(data_dir, model_path=base_path, **kw)
        print(f"Saved base model to {base_result.model_path}")
        base_err = base_result.best_val_base_px_error or base_result.best_val_px_error
        print(f"  base val err ~ {base_err:.1f}px  Epochs={base_result.epochs_run}")

    if args.review:
        from .live_review import run_live_review_gui
        from ..config_loader import default_config_path, load_grab_sensor_config

        cfg = load_grab_sensor_config(default_config_path())
        return run_live_review_gui(
            data_dir=data_dir,
            cfg=cfg,
            model_path=grasp_path,
            retrain_epochs=args.epochs,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
