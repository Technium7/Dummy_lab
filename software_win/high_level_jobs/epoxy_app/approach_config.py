"""Load epoxy rough-approach YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import EpoxyApproachConfig
from .paths import DEFAULT_APPROACH_CONFIG, REPO_ROOT


def default_approach_config_path() -> Path:
    return DEFAULT_APPROACH_CONFIG


def load_epoxy_approach_config(path: Path | str | None = None) -> EpoxyApproachConfig:
    cfg_path = Path(path) if path is not None else default_approach_config_path()
    cfg_path = cfg_path.resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"epoxy approach config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    return EpoxyApproachConfig.from_mapping(data, base_dir=REPO_ROOT)


def apply_runtime_calibrations(
    cfg: EpoxyApproachConfig,
    params: Mapping[str, Any] | None = None,
) -> EpoxyApproachConfig:
    """Hook for future overrides; approach uses fixed table inches (no calib)."""
    del params
    return cfg
