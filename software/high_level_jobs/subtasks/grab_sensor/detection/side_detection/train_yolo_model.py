#!/usr/bin/env python3
"""Train side-view YOLO sensor detector."""

from __future__ import annotations

import sys

from ..train_yolo_model import main as _main
from .paths import DEFAULT_DATA_DIR, DEFAULT_YOLO_EXPORT_DIR, DEFAULT_YOLO_MODEL


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not any(a.startswith("--grasp-data-dir") for a in argv):
        argv = ["--grasp-data-dir", str(DEFAULT_DATA_DIR), *argv]
    if not any(a.startswith("--data-yaml") for a in argv):
        argv = ["--data-yaml", str(DEFAULT_YOLO_EXPORT_DIR / "data.yaml"), *argv]
    if not any(a.startswith("--model-path") for a in argv):
        argv = ["--model-path", str(DEFAULT_YOLO_MODEL), *argv]
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
