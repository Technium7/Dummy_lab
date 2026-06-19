from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import GrabSensorConfig
from .table_position import resolve_table_calib_path


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "default_grasp.yaml"


def load_grab_sensor_config(path: Path | None = None) -> GrabSensorConfig:
    cfg_path = (path or default_config_path()).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Grab sensor config not found: {cfg_path}")

    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")

    return GrabSensorConfig.from_mapping(data, base_dir=_repo_root())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_config_path(params: Mapping[str, Any], ctx: Mapping[str, Any]) -> Path:
    raw = params.get("config") or params.get("config_path") or ctx.get("grab_sensor_config_path")
    if raw:
        p = Path(str(raw))
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()
    return default_config_path()


def apply_runtime_calibrations(
    cfg: GrabSensorConfig,
    params: Mapping[str, Any] | None = None,
) -> GrabSensorConfig:
    """Pin the newest table calib before detect/plan/execute (unless overridden)."""
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
