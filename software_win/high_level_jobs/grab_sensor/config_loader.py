from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import GrabSensorConfig
from .paths import DEFAULT_CONFIG, REPO_ROOT
from .table_position import resolve_table_calib_path


def default_config_path() -> Path:
    return DEFAULT_CONFIG


def _repo_root() -> Path:
    return REPO_ROOT


def load_grab_sensor_config(path: Path | None = None) -> GrabSensorConfig:
    cfg_path = (path or default_config_path()).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Grab sensor config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    return GrabSensorConfig.from_mapping(data, base_dir=_repo_root())


def apply_runtime_calibrations(
    cfg: GrabSensorConfig,
    params: Mapping[str, Any] | None = None,
) -> GrabSensorConfig:
    params = params or {}
    table_configured = cfg.table_calib_path
    table_override = params.get("table_calib_path")
    if table_override:
        table_configured = Path(str(table_override))
        if not table_configured.is_absolute():
            table_configured = (_repo_root() / table_configured).resolve()
    table_path = resolve_table_calib_path(table_configured)
    if table_path is None:
        return cfg
    if cfg.table_calib_path is not None and cfg.table_calib_path.resolve() == table_path.resolve():
        return cfg
    print(f"[grab_sensor] table calib: {table_path}")
    return replace(cfg, table_calib_path=table_path)
