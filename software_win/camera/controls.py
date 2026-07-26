"""Shared exposure / gain control surface for camera backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlLimits:
    """Slider limits for the live-view UI (exposure in milliseconds)."""

    exposure_ms_min: float = 0.1
    exposure_ms_max: float = 500.0
    gain_min: float = 0.0
    gain_max: float = 48.0
