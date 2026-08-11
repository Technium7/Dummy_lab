"""Load epoxy_app recording config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def _software_win_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "default_epoxy.yaml"


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    # Prefer repo-relative paths like software_win/...
    candidate = _repo_root() / path
    if candidate.exists() or str(path).startswith(("software_win", "software")):
        return candidate
    return _software_win_root() / path


@dataclass(frozen=True)
class ZeluxCamConfig:
    serial: str = ""
    index: int = 0
    exposure_ms: float = 10.0
    gain: float = 0.0


@dataclass(frozen=True)
class InfinityCamConfig:
    camera_number: int = 1
    opencv_index: int | None = None
    exposure_ms: float = 25.0
    gain: float = 1.0
    bayer: str = "BG"
    prefer_opencv: bool = False


@dataclass(frozen=True)
class EpoxyConfig:
    dataset_dir: Path
    record_result_dir: Path
    record_hz: float = 15.0
    zelux: ZeluxCamConfig = ZeluxCamConfig()
    infinity: InfinityCamConfig = InfinityCamConfig()

    def camera_roles(self) -> tuple[str, ...]:
        return ("zelux", "infinity")


def _zelux_from_mapping(raw: Mapping[str, Any] | None) -> ZeluxCamConfig:
    raw = raw or {}
    return ZeluxCamConfig(
        serial=str(raw.get("serial", "") or ""),
        index=int(raw.get("index", 0)),
        exposure_ms=float(raw.get("exposure_ms", 10.0)),
        gain=float(raw.get("gain", 0.0)),
    )


def _infinity_from_mapping(raw: Mapping[str, Any] | None) -> InfinityCamConfig:
    raw = raw or {}
    opencv_raw = raw.get("opencv_index", None)
    opencv_index = None if opencv_raw is None else int(opencv_raw)
    return InfinityCamConfig(
        camera_number=int(raw.get("camera_number", 1)),
        opencv_index=opencv_index,
        exposure_ms=float(raw.get("exposure_ms", 25.0)),
        gain=float(raw.get("gain", 1.0)),
        bayer=str(raw.get("bayer", "BG") or "BG"),
        prefer_opencv=bool(raw.get("prefer_opencv", False)),
    )


def load_epoxy_config(path: Path | str | None = None) -> EpoxyConfig:
    cfg_path = Path(path) if path is not None else default_config_path()
    cfg_path = cfg_path.resolve()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cams = data.get("cameras") or {}
    dataset_raw = data.get("dataset_dir") or "software_win/high_level_jobs/epoxy_app/dataset/episodes"
    result_raw = data.get("record_result_dir") or (
        "software_win/high_level_jobs/epoxy_app/dataset/record_result"
    )
    return EpoxyConfig(
        dataset_dir=_resolve_path(dataset_raw),
        record_result_dir=_resolve_path(result_raw),
        record_hz=float(data.get("record_hz", 15.0)),
        zelux=_zelux_from_mapping(cams.get("zelux")),
        infinity=_infinity_from_mapping(cams.get("infinity")),
    )
