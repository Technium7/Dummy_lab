#!/usr/bin/env python3
"""Train side-view grasp + base ellipse CNNs."""

from __future__ import annotations

import sys

from ..train_model import main as _main
from .paths import DEFAULT_DATA_DIR, DEFAULT_YOLO_MODEL


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not any(a.startswith("--data-dir") for a in argv):
        argv = ["--data-dir", str(DEFAULT_DATA_DIR), *argv]
    if not any(a.startswith("--yolo-model") for a in argv):
        argv = ["--yolo-model", str(DEFAULT_YOLO_MODEL), *argv]
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
