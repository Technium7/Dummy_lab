#!/usr/bin/env python3
"""Train top-view YOLO sensor detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..train_yolo_model import train_yolo_model
from ..export_yolo_dataset import export_yolo_dataset
from .paths import DEFAULT_DATA_DIR, DEFAULT_YOLO_EXPORT_DIR, DEFAULT_YOLO_MODEL


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train top-view YOLO sensor detector.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--data-yaml", type=Path, default=DEFAULT_YOLO_EXPORT_DIR / "data.yaml")
    p.add_argument("--model-path", type=Path, default=DEFAULT_YOLO_MODEL)
    p.add_argument("--export-first", action="store_true")
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
        print("Exporting YOLO dataset from base point labels…")
        data_yaml = export_yolo_dataset(
            args.data_dir.resolve(),
            data_yaml.parent,
            point_key="base",
            base_only=True,
        )

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
