from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .types import GrabSensorConfig


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
