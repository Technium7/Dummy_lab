#!/usr/bin/env python3
"""Train multi-class YOLO (sensor/post/other) for Windows grab_sensor bboxes."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.grab_sensor.paths import DATASET_DIR, MODEL_DIR  # noqa: E402
from high_level_jobs.grab_sensor.vision.export_yolo_dataset import (  # noqa: E402
    DEFAULT_YOLO_EXPORT,
    export_yolo_dataset,
)


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
        name="grab_sensor_yolo",
        exist_ok=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished but weights missing: {best}")
    shutil.copy2(best, model_path)
    print(f"Saved YOLO model to {model_path}")
    return model_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train YOLO bbox detector for grab_sensor")
    p.add_argument("--data", type=Path, default=DATASET_DIR)
    p.add_argument("--export-dir", type=Path, default=DEFAULT_YOLO_EXPORT)
    p.add_argument("--model-path", type=Path, default=MODEL_DIR / "sensor_yolo.pt")
    p.add_argument("--export-first", action="store_true", default=True)
    p.add_argument("--no-export", action="store_true", help="Skip re-export")
    p.add_argument("--base-weights", type=str, default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    data_yaml = args.export_dir / "data.yaml"
    if args.no_export:
        if not data_yaml.is_file():
            raise SystemExit(f"Missing {data_yaml}; run without --no-export")
    else:
        data_yaml = export_yolo_dataset(args.data, args.export_dir)

    train_yolo_model(
        data_yaml,
        model_path=args.model_path,
        base_weights=args.base_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
