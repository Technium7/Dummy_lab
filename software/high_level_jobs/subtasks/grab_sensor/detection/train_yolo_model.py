#!/usr/bin/env python3
"""
Train YOLO sensor detector from exported dataset.

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.export_yolo_dataset
  python -m high_level_jobs.subtasks.grab_sensor.detection.train_yolo_model
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .export_yolo_dataset import export_yolo_dataset


def train_yolo_model(
    data_yaml: Path,
    *,
    model_path: Path,
    base_weights: str = "yolov8n.pt",
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = 8,
    seed: int = 0,
) -> Path:
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics: pip install ultralytics") from exc

    data_yaml = data_yaml.resolve()
    model_path = model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_weights)
    results = model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        seed=int(seed),
        project=str(model_path.parent / "runs"),
        name="sensor_yolo",
        exist_ok=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished but weights missing: {best}")

    shutil.copy2(best, model_path)
    print(f"Saved YOLO model to {model_path}")
    return model_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pkg = Path(__file__).resolve().parent
    default_yaml = pkg / "dataset" / "sensor_yolo" / "data.yaml"
    default_model = pkg / "dataset" / "grasp_keypoints" / "model" / "sensor_yolo.pt"
    default_grasp = pkg / "dataset" / "grasp_keypoints"
    p = argparse.ArgumentParser(description="Train YOLO sensor detector.")
    p.add_argument("--data-yaml", type=Path, default=default_yaml)
    p.add_argument("--model-path", type=Path, default=default_model)
    p.add_argument("--grasp-data-dir", type=Path, default=default_grasp)
    p.add_argument("--export-first", action="store_true", help="Re-export YOLO dataset before training")
    p.add_argument("--base-weights", type=str, default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_yaml = args.data_yaml.resolve()

    if args.export_first or not data_yaml.is_file():
        print("Exporting YOLO dataset from grasp labels…")
        data_yaml = export_yolo_dataset(args.grasp_data_dir.resolve(), data_yaml.parent)

    train_yolo_model(
        data_yaml,
        model_path=args.model_path.resolve(),
        base_weights=args.base_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
