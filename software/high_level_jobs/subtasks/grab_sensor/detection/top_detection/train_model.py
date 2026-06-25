#!/usr/bin/env python3
"""
Train top-view base point CNN (YOLO crop + RGBD, 2-output point — no ellipse).

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.train_model --crop --depth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..grasp_point_net import TrainResult, default_base_model_path, list_labeled_samples, train_point_model
from .paths import DEFAULT_BASE_MODEL, DEFAULT_DATA_DIR, DEFAULT_YOLO_MODEL


def default_yolo_model() -> Path:
    return DEFAULT_YOLO_MODEL


def make_train_kwargs(
    *,
    epochs: int = 120,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    seed: int = 0,
    crop_mode: bool = True,
    use_depth: bool = True,
    yolo_model_path: Path | None = None,
    crop_padding_px: int = 16,
    depth_min_m: float = 0.25,
    depth_max_m: float = 0.85,
) -> dict:
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_frac": val_frac,
        "seed": seed,
        "crop_mode": crop_mode,
        "use_depth": use_depth,
        "yolo_model_path": yolo_model_path,
        "crop_padding_px": crop_padding_px,
        "depth_min_m": depth_min_m,
        "depth_max_m": depth_max_m,
        "base_label_mode": "point",
        "base_only": True,
    }


def run_training(
    data_dir: Path,
    *,
    base_path: Path | None = None,
    train_kwargs: dict | None = None,
    yolo_model_path: Path | None = None,
    verbose: bool = True,
) -> TrainResult:
    data_dir = data_dir.resolve()
    base_path = (base_path or default_base_model_path(data_dir)).resolve()
    kw = dict(train_kwargs or make_train_kwargs(yolo_model_path=yolo_model_path))

    base_count = len(list_labeled_samples(data_dir, require_base=True, base_only=True))
    if base_count < 2:
        raise ValueError(
            f"Need at least 2 base point labels in {data_dir}, found {base_count}."
        )

    crop_mode = bool(kw.get("crop_mode", True))
    use_depth = bool(kw.get("use_depth", True))
    mode = "crop+RGBD" if crop_mode and use_depth else ("crop" if crop_mode else "full RGB")
    yolo_model = kw.get("yolo_model_path")
    if verbose and crop_mode and yolo_model:
        print(f"YOLO crop model: {yolo_model}")
    if verbose:
        print(f"Training base point net ({mode}) on {base_count} samples from {data_dir}")

    result = train_point_model(
        data_dir,
        model_path=base_path,
        target="base",
        **kw,
    )
    if verbose:
        err = result.best_val_base_px_error or result.best_val_px_error
        print(f"Saved base model to {result.model_path}")
        print(f"  base val err ~ {err:.1f}px  Epochs={result.epochs_run}")
        if result.loss_plot_path:
            print(f"  loss plot: {result.loss_plot_path}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train top-view base point CNN (YOLO crop + RGBD).")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--base-model-path", type=Path, default=None)
    p.add_argument("--yolo-model", type=Path, default=default_yolo_model())
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
    base_path = (args.base_model_path or DEFAULT_BASE_MODEL).resolve()
    yolo_model = args.yolo_model.resolve() if args.crop and args.yolo_model.is_file() else None

    try:
        run_training(
            data_dir,
            base_path=base_path,
            train_kwargs=make_train_kwargs(
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
            ),
            yolo_model_path=yolo_model,
        )
    except ValueError as exc:
        print(exc)
        print("Collect data: python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_label")
        return 1

    if args.review:
        from .live_review import run_live_review_gui
        from ...config_loader import default_config_path, load_grab_sensor_config

        cfg = load_grab_sensor_config(default_config_path())
        return run_live_review_gui(data_dir=data_dir, cfg=cfg, retrain_epochs=args.epochs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
