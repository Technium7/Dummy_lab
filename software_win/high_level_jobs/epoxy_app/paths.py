"""Path helpers for epoxy_app approach planning."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SOFTWARE_WIN_DIR = PACKAGE_DIR.parents[1]
REPO_ROOT = PACKAGE_DIR.parents[2]
DEFAULT_APPROACH_CONFIG = PACKAGE_DIR / "config" / "default_epoxy_approach.yaml"
DEBUG_DIR = PACKAGE_DIR / "debug"
