#!/usr/bin/env python3
"""Export YOLO dataset from side-view grasp labels."""

from __future__ import annotations

import sys

from ..export_yolo_dataset import main as _main
from .paths import DEFAULT_DATA_DIR, DEFAULT_YOLO_EXPORT_DIR


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not any(a.startswith("--grasp-data-dir") for a in argv):
        argv = ["--grasp-data-dir", str(DEFAULT_DATA_DIR), *argv]
    if not any(a.startswith("--out-dir") for a in argv):
        argv = ["--out-dir", str(DEFAULT_YOLO_EXPORT_DIR), *argv]
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
