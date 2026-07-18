from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from .cameras import (
    OVERHEAD_ROLE,
    POLICY_CAMERA_ROLES,
    SIDE2_ROLE,
    SIDE_ROLE,
    CameraSpec,
)

ArchName = Literal["act", "diffusion"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_config_path(*, arch: ArchName = "act") -> Path:
    name = "default_diffusion.yaml" if arch == "diffusion" else "default_insert_sensor.yaml"
    return Path(__file__).resolve().parent / "config" / name


@dataclass(frozen=True)
class InsertSensorConfig:
    dataset_dir: Path
    model_path: Path
    policy: Any
    cameras: tuple[CameraSpec, ...]
    arch: ArchName = "act"
    epochs: int = 80
    batch_size: int = 16
    lr: float = 3e-4
    val_frac: float = 0.15
    seed: int = 42
    num_workers: int = 0
    record_hz: float = 15.0
    deploy_hz: float = 15.0
    action_scale: float = 1.0
    deploy_speed_scale: float = 2.0
    gripper_close_threshold: float = 0.5
    require_camera_serials: bool = True

    @property
    def num_cameras(self) -> int:
        """Policy / train / deploy camera count (1–3; ACT typically 2, diffusion up to 3)."""
        return self.policy.num_cameras

    @property
    def overhead_serial(self) -> str:
        for cam in self.cameras:
            if cam.role == OVERHEAD_ROLE:
                return cam.serial
        return self.cameras[0].serial if self.cameras else ""

    @property
    def side_serial(self) -> str:
        for cam in self.cameras:
            if cam.role == SIDE_ROLE:
                return cam.serial
        return ""

    @property
    def side2_serial(self) -> str:
        for cam in self.cameras:
            if cam.role == SIDE2_ROLE:
                return cam.serial
        return ""

    @property
    def wrist_serial(self) -> str:
        """Deprecated alias for side_serial."""
        return self.side_serial

    @property
    def record_camera_serial(self) -> str:
        """Legacy alias for overhead serial."""
        return self.overhead_serial

    def policy_cameras(self) -> tuple[CameraSpec, ...]:
        """Cameras used by train / deploy for ``policy.num_cameras``."""
        by_role = {c.role: c for c in self.cameras}
        if self.num_cameras <= 1:
            cam = by_role.get(OVERHEAD_ROLE) or (self.cameras[0] if self.cameras else None)
            return (cam,) if cam is not None else ()
        out: list[CameraSpec] = []
        for role in POLICY_CAMERA_ROLES[: self.num_cameras]:
            cam = by_role.get(role)
            if cam is not None:
                out.append(cam)
        return tuple(out)

    def record_cameras(self) -> tuple[CameraSpec, ...]:
        """All configured cameras for demonstration recording."""
        return self.cameras

    def active_cameras(self) -> tuple[CameraSpec, ...]:
        """Alias for ``policy_cameras`` (deploy / train)."""
        return self.policy_cameras()

    def assert_cameras_pinned(self, *, for_recording: bool = False) -> None:
        if not self.require_camera_serials:
            return
        cams = self.record_cameras() if for_recording else self.policy_cameras()
        missing = [c.role for c in cams if not (c.serial or "").strip()]
        if missing:
            raise RuntimeError(
                f"Camera serial(s) not pinned for: {missing}. "
                "USB order is unstable — set cameras.*.serial in config. "
                "Identify with: python software/camera/live_view.py "
                "or python -m high_level_jobs.subtasks.insert_sensor --preview-cameras"
            )

    @staticmethod
    def from_mapping(
        data: Mapping[str, Any],
        base_dir: Path | None = None,
        *,
        arch: ArchName | None = None,
    ) -> InsertSensorConfig:
        base = base_dir or _repo_root()

        def _resolve(raw: str | None, default: str) -> Path:
            p = Path(str(raw or default))
            if not p.is_absolute():
                p = (base / p).resolve()
            return p

        resolved_arch = _resolve_arch(data, arch)
        policy = _parse_policy(data, resolved_arch)
        cameras = _parse_cameras(data)

        default_model = (
            "software/high_level_jobs/subtasks/insert_sensor/runs/diffusion/latest/policy.pt"
            if resolved_arch == "diffusion"
            else "software/high_level_jobs/subtasks/insert_sensor/runs/latest/policy.pt"
        )
        return InsertSensorConfig(
            dataset_dir=_resolve(
                data.get("dataset_dir"),
                "software/high_level_jobs/subtasks/insert_sensor/dataset/episodes",
            ),
            model_path=_resolve(data.get("model_path"), default_model),
            policy=policy,
            cameras=cameras,
            arch=resolved_arch,
            epochs=int(data.get("epochs", 80)),
            batch_size=int(data.get("batch_size", 16)),
            lr=float(data.get("lr", 3e-4)),
            val_frac=float(data.get("val_frac", 0.15)),
            seed=int(data.get("seed", 42)),
            num_workers=int(data.get("num_workers", 0)),
            record_hz=float(data.get("record_hz", 15.0)),
            deploy_hz=float(data.get("deploy_hz", 15.0)),
            action_scale=float(data.get("action_scale", 1.0)),
            deploy_speed_scale=float(data.get("deploy_speed_scale", 2.0)),
            gripper_close_threshold=float(data.get("gripper_close_threshold", 0.5)),
            require_camera_serials=bool(data.get("require_camera_serials", True)),
        )


def _resolve_arch(data: Mapping[str, Any], override: ArchName | None) -> ArchName:
    if override in ("act", "diffusion"):
        return override
    raw = str(data.get("arch", "act") or "act").strip().lower()
    if raw in ("diffusion", "diffusion_policy", "dp"):
        return "diffusion"
    if raw in ("act", "act_dinov2", "act_dinov2_v1"):
        return "act"
    # Heuristic: diffusion yaml without explicit arch
    if int(data.get("diffusion_steps", 0) or 0) > 0 and not data.get("style_dim"):
        return "diffusion"
    return "act"


def _parse_policy(data: Mapping[str, Any], arch: ArchName) -> Any:
    if arch == "diffusion":
        from .policies.diffusion import DiffusionPolicyConfig

        return DiffusionPolicyConfig.from_mapping(dict(data))
    from .model import InsertionPolicyConfig

    return InsertionPolicyConfig.from_mapping(dict(data))


def _cam_block(cams_raw: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    block = cams_raw.get(role, {})
    return block if isinstance(block, Mapping) else {}


def _exposure_us(block: Mapping[str, Any]) -> float | None:
    if "exposure_us" not in block or block.get("exposure_us") is None:
        return None
    return float(block["exposure_us"])


def _parse_cameras(data: Mapping[str, Any]) -> tuple[CameraSpec, ...]:
    cams_raw = data.get("cameras")
    legacy = str(data.get("record_camera_serial", "") or "").strip()
    if isinstance(cams_raw, Mapping):
        side_block = cams_raw.get(SIDE_ROLE, cams_raw.get("wrist", {}))
        if not isinstance(side_block, Mapping):
            side_block = {}
        oh_block = _cam_block(cams_raw, OVERHEAD_ROLE)
        oh_serial = str(oh_block.get("serial", "") or "").strip()
        if legacy:
            oh_serial = legacy
        side_serial = str(side_block.get("serial", "") or "").strip()

        cams: list[CameraSpec] = [
            CameraSpec(role=OVERHEAD_ROLE, serial=oh_serial, exposure_us=_exposure_us(oh_block)),
            CameraSpec(role=SIDE_ROLE, serial=side_serial, exposure_us=_exposure_us(side_block)),
        ]
        if SIDE2_ROLE in cams_raw:
            side2_block = _cam_block(cams_raw, SIDE2_ROLE)
            cams.append(
                CameraSpec(
                    role=SIDE2_ROLE,
                    serial=str(side2_block.get("serial", "") or "").strip(),
                    exposure_us=_exposure_us(side2_block),
                )
            )
        return tuple(cams)

    return (
        CameraSpec(role=OVERHEAD_ROLE, serial=legacy),
        CameraSpec(role=SIDE_ROLE, serial=""),
    )


def load_insert_sensor_config(
    path: Path | None = None,
    *,
    arch: ArchName | None = None,
) -> InsertSensorConfig:
    cfg_path = (path or default_config_path(arch=arch or "act")).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Insert sensor config not found: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    return InsertSensorConfig.from_mapping(data, base_dir=_repo_root(), arch=arch)
