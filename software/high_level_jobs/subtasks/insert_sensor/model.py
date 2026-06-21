"""Latent world-action insertion policy (BC + dynamics + temporal consistency)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from high_level_jobs.subtasks.grab_sensor.detection.grasp_point_net import (
    DEFAULT_DEPTH_MAX_M,
    DEFAULT_DEPTH_MIN_M,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

KEYPOINT_DIM = 8
STATE_DIM = 13
ACTION_DIM = 7


@dataclass(frozen=True)
class InsertionPolicyConfig:
    history_len: int = 8
    temporal: Literal["gru", "transformer"] = "gru"
    latent_dim: int = 256
    img_width: int = 160
    img_height: int = 120
    depth_min_m: float = DEFAULT_DEPTH_MIN_M
    depth_max_m: float = DEFAULT_DEPTH_MAX_M
    lambda_world: float = 0.5
    lambda_smooth: float = 0.05
    lambda_aux: float = 0.0

    @staticmethod
    def from_mapping(data: dict[str, Any]) -> InsertionPolicyConfig:
        temporal = str(data.get("temporal", "gru")).strip().lower()
        if temporal not in ("gru", "transformer"):
            raise ValueError(f"temporal must be gru or transformer, got {temporal!r}")
        return InsertionPolicyConfig(
            history_len=int(data.get("history_len", 8)),
            temporal=temporal,  # type: ignore[arg-type]
            latent_dim=int(data.get("latent_dim", 256)),
            img_width=int(data.get("img_width", 160)),
            img_height=int(data.get("img_height", 120)),
            depth_min_m=float(data.get("depth_min_m", DEFAULT_DEPTH_MIN_M)),
            depth_max_m=float(data.get("depth_max_m", DEFAULT_DEPTH_MAX_M)),
            lambda_world=float(data.get("lambda_world", 0.5)),
            lambda_smooth=float(data.get("lambda_smooth", 0.05)),
            lambda_aux=float(data.get("lambda_aux", 0.0)),
        )


def preprocess_rgbd_batch(
    color_rgb: np.ndarray,
    depth_m: np.ndarray,
    *,
    out_w: int,
    out_h: int,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    import cv2

    n = color_rgb.shape[0]
    out = np.zeros((n, 4, out_h, out_w), dtype=np.float32)
    span = max(depth_max_m - depth_min_m, 1e-6)
    for i in range(n):
        rgb = color_rgb[i]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        depth = depth_m[i].astype(np.float32)
        rgb_small = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        depth_small = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        rgb_f = rgb_small.astype(np.float32) / 255.0
        rgb_f = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD
        valid = depth_small > 1e-4
        depth_norm = np.zeros_like(depth_small, dtype=np.float32)
        depth_norm[valid] = np.clip((depth_small[valid] - depth_min_m) / span, 0.0, 1.0)
        out[i, 0:3] = np.transpose(rgb_f, (2, 0, 1))
        out[i, 3] = depth_norm
    return out


class RgbdEncoder(nn.Module):
    def __init__(self, *, out_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class MlpEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or max(out_dim * 2, 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceptionEncoder(nn.Module):
    def __init__(self, cfg: InsertionPolicyConfig) -> None:
        super().__init__()
        self.img_encoder = RgbdEncoder(out_dim=128)
        self.keypoint_encoder = MlpEncoder(KEYPOINT_DIM, 64)
        self.state_encoder = MlpEncoder(STATE_DIM, 64)
        self.fusion = nn.Sequential(
            nn.Linear(256, cfg.latent_dim),
            nn.ReLU(inplace=True),
        )

    def encode_step(
        self,
        rgbd: torch.Tensor,
        keypoint: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        f_img = self.img_encoder(rgbd)
        f_k = self.keypoint_encoder(keypoint)
        f_s = self.state_encoder(state)
        return self.fusion(torch.cat([f_img, f_k, f_s], dim=-1))

    def forward(
        self,
        rgbd: torch.Tensor,
        keypoint: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        b, t = rgbd.shape[:2]
        rgbd_flat = rgbd.reshape(b * t, *rgbd.shape[2:])
        kp_flat = keypoint.reshape(b * t, -1)
        st_flat = state.reshape(b * t, -1)
        x_flat = self.encode_step(rgbd_flat, kp_flat, st_flat)
        return x_flat.reshape(b, t, -1)


class TemporalEncoder(nn.Module):
    def __init__(self, cfg: InsertionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.temporal == "gru":
            self.core = nn.GRU(
                input_size=cfg.latent_dim,
                hidden_size=cfg.latent_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.latent_dim,
                nhead=4,
                dim_feedforward=cfg.latent_dim * 2,
                batch_first=True,
                dropout=0.1,
            )
            self.core = nn.TransformerEncoder(layer, num_layers=2)
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.history_len, cfg.latent_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.temporal == "gru":
            out, _ = self.core(x)
            return out
        pos = self.pos_embed[:, : x.shape[1], :]
        return self.core(x + pos)


class InsertionPolicy(nn.Module):
    def __init__(self, cfg: InsertionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.perception = PerceptionEncoder(cfg)
        self.temporal = TemporalEncoder(cfg)
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.latent_dim, ACTION_DIM),
        )
        self.world_head = nn.Sequential(
            nn.Linear(cfg.latent_dim + ACTION_DIM, cfg.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.aux_head = nn.Linear(cfg.latent_dim, KEYPOINT_DIM)

    def encode_temporal(
        self,
        rgbd: torch.Tensor,
        keypoint: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.perception(rgbd, keypoint, state)
        z = self.temporal(x)
        return x, z

    def forward(
        self,
        rgbd: torch.Tensor,
        keypoint: torch.Tensor,
        state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x, z = self.encode_temporal(rgbd, keypoint, state)
        action = self.policy_head(z)
        z_next_pred = self.world_head(torch.cat([z, action], dim=-1))
        aux_kp = self.aux_head(z)
        return {
            "x": x,
            "z": z,
            "action": action,
            "z_next_pred": z_next_pred,
            "aux_kp": aux_kp,
        }

    def predict_action(
        self,
        rgbd: torch.Tensor,
        keypoint: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, z = self.encode_temporal(rgbd, keypoint, state)
            return self.policy_head(z)


def compute_losses(
    model: InsertionPolicy,
    batch: dict[str, torch.Tensor],
    *,
    cfg: InsertionPolicyConfig,
) -> dict[str, torch.Tensor]:
    out = model(batch["rgbd"], batch["keypoint"], batch["state"])
    l_policy = F.mse_loss(out["action"], batch["action"])

    z = out["z"]
    x = out["x"]
    z_next_target = x[:, 1:, :].detach()
    z_curr = z[:, :-1, :]
    action_curr = out["action"][:, :-1, :]
    z_next_pred = model.world_head(torch.cat([z_curr, action_curr], dim=-1))
    if z_next_target.shape[1] > 0:
        l_world = F.mse_loss(z_next_pred, z_next_target)
    else:
        l_world = torch.zeros((), device=z.device)

    if z.shape[1] > 1:
        l_smooth = F.mse_loss(z[:, 1:, :], z[:, :-1, :])
    else:
        l_smooth = torch.zeros((), device=z.device)

    l_aux = torch.zeros((), device=z.device)
    if cfg.lambda_aux > 0.0:
        l_aux = F.mse_loss(out["aux_kp"], batch["keypoint"])

    total = (
        l_policy
        + cfg.lambda_world * l_world
        + cfg.lambda_smooth * l_smooth
        + cfg.lambda_aux * l_aux
    )
    return {
        "total": total,
        "policy": l_policy,
        "world": l_world,
        "smooth": l_smooth,
        "aux": l_aux,
    }


def save_policy(
    path: Path,
    model: InsertionPolicy,
    cfg: InsertionPolicyConfig,
    *,
    norm_stats: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": {
                "history_len": cfg.history_len,
                "temporal": cfg.temporal,
                "latent_dim": cfg.latent_dim,
                "img_width": cfg.img_width,
                "img_height": cfg.img_height,
                "depth_min_m": cfg.depth_min_m,
                "depth_max_m": cfg.depth_max_m,
                "lambda_world": cfg.lambda_world,
                "lambda_smooth": cfg.lambda_smooth,
                "lambda_aux": cfg.lambda_aux,
            },
            "norm_stats": norm_stats or {},
            "meta": extra or {},
        },
        path,
    )


def load_policy(
    path: Path,
    *,
    device: str | None = None,
) -> tuple[InsertionPolicy, InsertionPolicyConfig, dict[str, Any]]:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path.resolve(), map_location=torch_device, weights_only=False)
    cfg = InsertionPolicyConfig.from_mapping(payload.get("cfg", {}))
    model = InsertionPolicy(cfg).to(torch_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg, payload.get("norm_stats", {})
