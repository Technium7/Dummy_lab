#!/usr/bin/env python3
"""
Train grasp and/or base CNNs on labeled samples.

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.train_model
  python -m high_level_jobs.subtasks.grab_sensor.detection.train_model --crop --depth --epochs 120
  python -m high_level_jobs.subtasks.grab_sensor.detection.train_model --grasp-only
  python -m high_level_jobs.subtasks.grab_sensor.detection.train_model --base-only
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from .grasp_point_net import (
    TrainResult,
    default_base_model_path,
    default_model_path,
    list_labeled_samples,
    train_point_model,
)


def default_yolo_model() -> Path:
    return (
        Path(__file__).resolve().parent
        / "side_detection"
        / "dataset"
        / "grasp_keypoints"
        / "model"
        / "sensor_yolo.pt"
    )


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
    }


@dataclass
class TrainingResults:
    grasp: TrainResult | None = None
    base: TrainResult | None = None


def _train_kwargs_from_args(args: argparse.Namespace, yolo_model: Path | None) -> dict:
    return make_train_kwargs(
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


def _print_result(label: str, result: TrainResult) -> None:
    err = result.best_val_base_px_error if label == "base" else result.best_val_px_error
    if label == "base" and result.best_val_base_px_error is None:
        err = result.best_val_px_error
    print(f"Saved {label} model to {result.model_path}")
    print(f"  {label} val err ~ {err:.1f}px  Epochs={result.epochs_run}")
    if result.loss_plot_path:
        print(f"  {label} loss plot: {result.loss_plot_path}")


def run_training(
    data_dir: Path,
    *,
    grasp_path: Path | None = None,
    base_path: Path | None = None,
    train_grasp: bool = True,
    train_base: bool = True,
    train_kwargs: dict | None = None,
    yolo_model_path: Path | None = None,
    verbose: bool = True,
) -> TrainingResults:
    """Train grasp and/or base point CNNs. Raises ValueError if sample counts are too low."""
    data_dir = data_dir.resolve()
    grasp_path = (grasp_path or default_model_path(data_dir)).resolve()
    base_path = (base_path or default_base_model_path(data_dir)).resolve()
    kw = dict(train_kwargs or make_train_kwargs(yolo_model_path=yolo_model_path))

    grasp_count = len(list_labeled_samples(data_dir))
    base_count = len(list_labeled_samples(data_dir, require_base=True))
    if train_grasp and grasp_count < 2:
        raise ValueError(
            f"Need at least 2 labeled grasp samples in {data_dir}, found {grasp_count}."
        )
    if train_base and base_count < 2:
        total = len(list_labeled_samples(data_dir))
        raise ValueError(
            f"Need at least 2 samples with base labels in {data_dir}, "
            f"found {base_count} ({total} grasp labels)."
        )

    crop_mode = bool(kw.get("crop_mode", True))
    use_depth = bool(kw.get("use_depth", True))
    mode = "crop+RGBD" if crop_mode and use_depth else ("crop" if crop_mode else "full RGB")
    yolo_model = kw.get("yolo_model_path")
    if verbose and crop_mode and yolo_model:
        print(f"YOLO crop model: {yolo_model}")

    results = TrainingResults()
    if train_grasp:
        if verbose:
            print(f"Training grasp net ({mode}) on {grasp_count} samples from {data_dir}")
        results.grasp = train_point_model(data_dir, model_path=grasp_path, target="grasp", **kw)
        if verbose and results.grasp is not None:
            _print_result("grasp", results.grasp)

    if train_base:
        if verbose:
            print(f"Training base net ({mode}) on {base_count} samples from {data_dir}")
        results.base = train_point_model(data_dir, model_path=base_path, target="base", **kw)
        if verbose and results.base is not None:
            _print_result("base", results.base)

    return results


def _resolve_targets(args: argparse.Namespace) -> tuple[bool, bool]:
    if args.grasp_only:
        return True, False
    if args.base_only:
        return False, True
    return True, True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_data = (
        Path(__file__).resolve().parent / "side_detection" / "dataset" / "grasp_keypoints"
    )
    p = argparse.ArgumentParser(
        description="Train grasp and/or base CNNs (separate nets, crop + RGBD). Default: both.",
    )
    p.add_argument("--data-dir", type=Path, default=default_data, help="Labeled dataset directory")
    p.add_argument("--model-path", type=Path, default=None, help="Grasp .pt output path")
    p.add_argument("--base-model-path", type=Path, default=None, help="Base .pt output path")
    p.add_argument("--yolo-model", type=Path, default=default_yolo_model(), help="YOLO weights for training crops")
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
    targets = p.add_mutually_exclusive_group()
    targets.add_argument(
        "--grasp-only",
        action="store_true",
        help="Train grasp net only (grasp_point.pt)",
    )
    targets.add_argument(
        "--base-only",
        action="store_true",
        help="Train base net only (base_point.pt)",
    )
    targets.add_argument(
        "--both",
        action="store_true",
        help="Train grasp + base nets (default when no target flag is set)",
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
    train_grasp, train_base = _resolve_targets(args)
    data_dir = args.data_dir.resolve()
    grasp_path = (args.model_path or default_model_path(data_dir)).resolve()
    base_path = (args.base_model_path or default_base_model_path(data_dir)).resolve()
    yolo_model = args.yolo_model.resolve() if args.crop and args.yolo_model.is_file() else None

    try:
        run_training(
            data_dir,
            grasp_path=grasp_path,
            base_path=base_path,
            train_grasp=train_grasp,
            train_base=train_base,
            train_kwargs=_train_kwargs_from_args(args, yolo_model),
        )
    except ValueError as exc:
        print(exc)
        if train_grasp and train_base:
            print("Collect data: python -m high_level_jobs.subtasks.grab_sensor.detection.live_label")
            print("Or use --grasp-only / --base-only to train one net.")
        elif train_grasp:
            print("Collect data: python -m high_level_jobs.subtasks.grab_sensor.detection.live_label")
        else:
            print("Label bases: python -m high_level_jobs.subtasks.grab_sensor.detection.relabel")
        return 1

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
