#!/usr/bin/env python3
"""Train and save the synthetic hole-patch classifier used by calibration_table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TABLE_CALIB_DIR = Path(__file__).resolve().parent
_SOFTWARE_DIR = _TABLE_CALIB_DIR.parents[1]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from camera.calibration_table.hole_classifier import DEFAULT_HOLE_MODEL, train_and_save_synthetic


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap hole patch NN weights on synthetic data.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_HOLE_MODEL,
        help="Output weights path",
    )
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args()
    path = train_and_save_synthetic(args.out.resolve(), epochs=args.epochs)
    print(f"Saved hole classifier to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
