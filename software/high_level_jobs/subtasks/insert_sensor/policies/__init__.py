"""Policy backends for insert_sensor (ACT lives in ``model.py``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal


ArchName = Literal["act", "diffusion"]


def detect_arch_from_checkpoint(path: Path) -> ArchName:
    import torch

    payload = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    tag = str((payload.get("cfg") or {}).get("arch", "") or "").lower()
    meta = str((payload.get("meta") or {}).get("arch", "") or "").lower()
    blob = f"{tag} {meta}"
    if "diffusion" in blob:
        return "diffusion"
    return "act"


def load_policy_for_arch(
    path: Path,
    *,
    arch: ArchName | None = None,
    device: str | None = None,
) -> tuple[Any, Any, dict[str, Any], ArchName]:
    """Return ``(model, policy_cfg, norm_stats, arch)``."""
    resolved = arch or detect_arch_from_checkpoint(path)
    if resolved == "diffusion":
        from .diffusion import load_policy

        model, cfg, stats = load_policy(path, device=device)
        return model, cfg, stats, "diffusion"
    from high_level_jobs.subtasks.insert_sensor.model import load_policy

    model, cfg, stats = load_policy(path, device=device)
    return model, cfg, stats, "act"
