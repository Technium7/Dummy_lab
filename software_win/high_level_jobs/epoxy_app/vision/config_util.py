"""Load vision YAML and resolve paths relative to the Dummy_lab repo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

VISION_DIR = Path(__file__).resolve().parent
EPOXY_APP_DIR = VISION_DIR.parent
SOFTWARE_WIN = EPOXY_APP_DIR.parents[1]
REPO_ROOT = SOFTWARE_WIN.parent


def load_vision_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else VISION_DIR / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid vision config: {cfg_path}")
    return cfg


def resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path
