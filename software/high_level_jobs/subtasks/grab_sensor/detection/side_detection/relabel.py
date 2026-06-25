#!/usr/bin/env python3
"""Offline relabel base ellipses on side-view dataset images."""

from __future__ import annotations

import sys

from ..relabel import main as _main
from .paths import DEFAULT_DATA_DIR


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not any(a.startswith("--data-dir") for a in argv):
        argv = ["--data-dir", str(DEFAULT_DATA_DIR), *argv]
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
