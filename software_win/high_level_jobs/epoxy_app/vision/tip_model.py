"""Mono tip heatmap detector (MobileNetV3-small stem)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from high_level_jobs.epoxy_app.vision.model import _make_encoder


class TipNet(nn.Module):
    """Full-frame mono → heatmap; tip = argmax peak."""

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        in_channels: int = 1,
        heatmap_size: int = 64,
    ) -> None:
        super().__init__()
        self.heatmap_size = int(heatmap_size)
        self.encoder, _feat_dim = _make_encoder(backbone, pretrained, in_channels)
        last_ch = self._infer_feature_channels()
        self.proj = nn.Sequential(
            nn.Conv2d(last_ch, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def _infer_feature_channels(self) -> int:
        try:
            last = self.encoder.features[-1]
            if hasattr(last, "out_channels"):
                return int(last.out_channels)
            for m in reversed(list(last.modules())):
                if isinstance(m, nn.Conv2d):
                    return int(m.out_channels)
        except Exception:
            pass
        return 576

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "features"):
            return self.encoder.features(x)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        heat = self.proj(feats)
        return F.interpolate(
            heat, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False
        )

    @torch.no_grad()
    def predict_tip(
        self,
        x: torch.Tensor,
        *,
        frame_w: int,
        frame_h: int,
        prior_xy: tuple[float, float] | None = None,
        track_radius_px: float | None = None,
        track_sigma_px: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_px, y_px, conf) batched; conf = peak sigmoid heatmap value.

        Optional ``prior_xy`` biases the peak toward the last tip (hard radius
        mask and/or soft Gaussian) so the stationary apply blob does not win.
        """
        logits = self.forward(x)
        score = torch.sigmoid(logits)
        b, _, hh, ww = score.shape
        if prior_xy is not None and (track_radius_px or track_sigma_px):
            device = score.device
            dtype = score.dtype
            yy = (torch.arange(hh, device=device, dtype=dtype) + 0.5) / hh * frame_h
            xx = (torch.arange(ww, device=device, dtype=dtype) + 0.5) / ww * frame_w
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
            px, py = float(prior_xy[0]), float(prior_xy[1])
            dist2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
            weight = torch.ones((1, 1, hh, ww), device=device, dtype=dtype)
            if track_radius_px is not None and track_radius_px > 0:
                weight = weight * (dist2 <= float(track_radius_px) ** 2).to(dtype)
            if track_sigma_px is not None and track_sigma_px > 0:
                weight = weight * torch.exp(-dist2 / (2.0 * float(track_sigma_px) ** 2))
            # Avoid all-zero mask (e.g. prior off-frame): fall back to global
            if float(weight.max()) > 0:
                score = score * weight
        flat = score.view(b, -1)
        conf, idx = flat.max(dim=1)
        ys = (idx // ww).float()
        xs = (idx % ww).float()
        x_px = (xs + 0.5) / ww * frame_w
        y_px = (ys + 0.5) / hh * frame_h
        return x_px, y_px, conf


def build_tip_model(cfg: dict[str, Any] | None = None, **overrides: Any) -> TipNet:
    m = (cfg or {}).get("model", {})
    tip = (cfg or {}).get("tip", {})
    kwargs = {
        "backbone": m.get("backbone", "mobilenet_v3_small"),
        "pretrained": bool(m.get("pretrained", True)),
        "in_channels": 1,
        "heatmap_size": int(tip.get("heatmap_size", 64)),
    }
    kwargs.update(overrides)
    return TipNet(**kwargs)


def load_tip_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    cfg: dict[str, Any] | None = None,
) -> tuple[TipNet, dict[str, Any]]:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", cfg)
    model = build_tip_model(model_cfg)
    state = ckpt.get("tip_state") or ckpt.get("model_state")
    if state is not None:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt
