#!/usr/bin/env python3
"""Live side-view review: YOLO + grasp + base ellipse NN proposals."""

from __future__ import annotations

import sys

from ..live_review import main as _main
from .paths import DEFAULT_DATA_DIR


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not any(a.startswith("--data-dir") for a in argv):
        argv = ["--data-dir", str(DEFAULT_DATA_DIR), *argv]
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
