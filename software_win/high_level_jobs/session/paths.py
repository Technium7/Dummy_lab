"""Shared paths for the continuous high-level session."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SOFTWARE_WIN_DIR = PACKAGE_DIR.parents[1]
REPO_ROOT = PACKAGE_DIR.parents[2]

# General place for recorded motions (grab put-back, power return, etc.)
RECORDS_DIR = PACKAGE_DIR / "records"
# Named reusable trajectories (save / replay across plans)
LIBRARY_DIR = RECORDS_DIR / "library"
# Append-only JSONL of record_power samples
POWER_LOG_DIR = RECORDS_DIR / "power_logs"

GRAB_SENSOR_RECORD = "grab_sensor"
POWER_MEASURE_RECORD = "power_measure"
POWER_LOG_NAME = "record_power.jsonl"
