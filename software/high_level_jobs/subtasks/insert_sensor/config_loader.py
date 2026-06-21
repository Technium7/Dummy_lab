from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from high_level_jobs.subtasks.grab_sensor.config_loader import (
    apply_runtime_calibrations,
    load_grab_sensor_config,
)
from high_level_jobs.subtasks.grab_sensor.models import GrabSensorConfig

from .model import InsertionPolicyConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "default_insert_sensor.yaml"


@dataclass(frozen=True)
class InsertSensorConfig:
    grasp_config_path: Path
    dataset_dir: Path
    model_path: Path
    policy: InsertionPolicyConfig
    epochs: int = 80
    batch_size: int = 16
    lr: float = 3e-4
    val_frac: float = 0.15
    seed: int = 42
    num_workers: int = 0
    record_hz: float = 15.0
    record_camera_serial: str = ""
    record_keypoint_online: bool = False
    deploy_hz: float = 15.0
    action_scale: float = 1.0
    gripper_close_threshold: float = 0.5

    @staticmethod
    def from_mapping(data: Mapping[str, Any], base_dir: Path | None = None) -> InsertSensorConfig:
        base = base_dir or _repo_root()

        def _resolve(raw: str | None, default: str) -> Path:
            p = Path(str(raw or default))
            if not p.is_absolute():
                p = (base / p).resolve()
            return p

        grasp_path = _resolve(
            data.get("grasp_config"),
            "software/high_level_jobs/subtasks/grab_sensor/config/default_grasp.yaml",
        )
        return InsertSensorConfig(
            grasp_config_path=grasp_path,
            dataset_dir=_resolve(
                data.get("dataset_dir"),
                "software/high_level_jobs/subtasks/insert_sensor/dataset/episodes",
            ),
            model_path=_resolve(
                data.get("model_path"),
                "software/high_level_jobs/subtasks/insert_sensor/runs/latest/policy.pt",
            ),
            policy=InsertionPolicyConfig.from_mapping(dict(data)),
            epochs=int(data.get("epochs", 80)),
            batch_size=int(data.get("batch_size", 16)),
            lr=float(data.get("lr", 3e-4)),
            val_frac=float(data.get("val_frac", 0.15)),
            seed=int(data.get("seed", 42)),
            num_workers=int(data.get("num_workers", 0)),
            record_hz=float(data.get("record_hz", 15.0)),
            record_camera_serial=str(data.get("record_camera_serial", "")),
            record_keypoint_online=bool(data.get("record_keypoint_online", False)),
            deploy_hz=float(data.get("deploy_hz", 15.0)),
            action_scale=float(data.get("action_scale", 1.0)),
            gripper_close_threshold=float(data.get("gripper_close_threshold", 0.5)),
        )


def load_insert_sensor_config(path: Path | None = None) -> InsertSensorConfig:
    cfg_path = (path or default_config_path()).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Insert sensor config not found: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    return InsertSensorConfig.from_mapping(data, base_dir=_repo_root())


def load_grasp_config(cfg: InsertSensorConfig) -> GrabSensorConfig:
    grasp = load_grab_sensor_config(cfg.grasp_config_path)
    return apply_runtime_calibrations(grasp)
