from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from high_level_jobs.subtasks.robot_actions import _v

from .config_loader import default_config_path, load_insert_sensor_config
from .deploy import PolicyRunner
from .train import train_policy

logger = logging.getLogger(__name__)


def insert_sensor_train(params: Mapping[str, Any], ctx: dict[str, Any]) -> None:
    """Train insertion policy from teleop episodes."""
    cfg_path = Path(str(params.get("config", default_config_path())))
    cfg = load_insert_sensor_config(cfg_path)
    if ctx.get("dry_run"):
        _v(ctx, f"[dry-run] insert_sensor_train: would train on {cfg.dataset_dir}")
        return
    out = train_policy(cfg, device=params.get("device"), out_path=params.get("out"))
    ctx["insert_sensor_model_path"] = str(out)
    _v(ctx, f"Trained insert_sensor policy -> {out}")


def insert_sensor_load_policy(params: Mapping[str, Any], ctx: dict[str, Any]) -> None:
    """Load trained policy into ctx for downstream deploy steps."""
    cfg_path = Path(str(params.get("config", default_config_path())))
    cfg = load_insert_sensor_config(cfg_path)
    model_path = params.get("model") or cfg.model_path
    if ctx.get("dry_run"):
        _v(ctx, f"[dry-run] insert_sensor_load_policy: would load {model_path}")
        return
    runner = PolicyRunner(cfg, device=params.get("device"), model_path=Path(str(model_path)))
    ctx["insert_sensor_policy"] = runner
    ctx["insert_sensor_config"] = cfg
    _v(ctx, f"Loaded insert_sensor policy from {model_path}")


INSERT_SENSOR_SUBTASKS = {
    "insert_sensor_train": insert_sensor_train,
    "insert_sensor_load_policy": insert_sensor_load_policy,
}
