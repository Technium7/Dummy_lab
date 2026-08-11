"""Temporal tip tracking — prefer local heatmap peak near last tip.

Near apply, TipNet's global argmax often snaps to the stationary epoxy blob.
Tracking keeps the estimate on the moving needle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from high_level_jobs.epoxy_app.vision.tip_model import TipNet


@dataclass
class TipTrackConfig:
    enabled: bool = True
    radius_px: float = 96.0
    sigma_px: float = 48.0
    max_jump_px: float = 140.0
    vel_ema: float = 0.35

    @classmethod
    def from_tip_cfg(cls, tip_cfg: dict[str, Any] | None) -> TipTrackConfig:
        tip_cfg = tip_cfg or {}
        return cls(
            enabled=bool(tip_cfg.get("track", True)),
            radius_px=float(tip_cfg.get("track_radius_px", 96)),
            sigma_px=float(tip_cfg.get("track_sigma_px", 48)),
            max_jump_px=float(tip_cfg.get("track_max_jump_px", 140)),
            vel_ema=float(tip_cfg.get("track_vel_ema", 0.35)),
        )


class TipTracker:
    """Stateful tip estimate with motion prior for sequential frames."""

    def __init__(self, cfg: TipTrackConfig | None = None) -> None:
        self.cfg = cfg or TipTrackConfig()
        self.prev_xy: tuple[float, float] | None = None
        self.vel_xy: tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        self.prev_xy = None
        self.vel_xy = (0.0, 0.0)

    def seed(self, x: float, y: float) -> None:
        self.prev_xy = (float(x), float(y))
        self.vel_xy = (0.0, 0.0)

    def prior_xy(self) -> tuple[float, float] | None:
        if self.prev_xy is None:
            return None
        return (self.prev_xy[0] + self.vel_xy[0], self.prev_xy[1] + self.vel_xy[1])

    @torch.inference_mode()
    def step(
        self,
        model: TipNet,
        image_t: torch.Tensor,
        *,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float, float]:
        """Run one tracked tip prediction. image_t: (1,C,H,W)."""
        cfg = self.cfg
        if not cfg.enabled or self.prev_xy is None:
            x, y, c = model.predict_tip(image_t, frame_w=frame_w, frame_h=frame_h)
            xf, yf, cf = float(x[0]), float(y[0]), float(c[0])
            self._commit(xf, yf)
            return xf, yf, cf

        prior = self.prior_xy()
        x, y, c = model.predict_tip(
            image_t,
            frame_w=frame_w,
            frame_h=frame_h,
            prior_xy=prior,
            track_radius_px=cfg.radius_px,
            track_sigma_px=cfg.sigma_px,
        )
        xf, yf, cf = float(x[0]), float(y[0]), float(c[0])

        # Safety: if local peak is absurdly far from last tip, fall back to hold
        dx = xf - self.prev_xy[0]
        dy = yf - self.prev_xy[1]
        if (dx * dx + dy * dy) ** 0.5 > cfg.max_jump_px:
            # Try global peak once; accept only if closer than max_jump
            gx, gy, gc = model.predict_tip(image_t, frame_w=frame_w, frame_h=frame_h)
            gxf, gyf, gcf = float(gx[0]), float(gy[0]), float(gc[0])
            gdx = gxf - self.prev_xy[0]
            gdy = gyf - self.prev_xy[1]
            if (gdx * gdx + gdy * gdy) ** 0.5 <= cfg.max_jump_px:
                xf, yf, cf = gxf, gyf, gcf
            else:
                xf, yf = self.prev_xy
                # keep last conf-ish signal low so crop gate can fail open if needed
                cf = min(cf, 0.2)

        self._commit(xf, yf)
        return xf, yf, cf

    def _commit(self, x: float, y: float) -> None:
        ema = float(self.cfg.vel_ema)
        if self.prev_xy is not None:
            dx = x - self.prev_xy[0]
            dy = y - self.prev_xy[1]
            self.vel_xy = (
                ema * dx + (1.0 - ema) * self.vel_xy[0],
                ema * dy + (1.0 - ema) * self.vel_xy[1],
            )
        else:
            self.vel_xy = (0.0, 0.0)
        self.prev_xy = (x, y)
