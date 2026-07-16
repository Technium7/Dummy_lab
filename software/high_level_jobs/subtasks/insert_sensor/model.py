"""ACT (Action Chunking with Transformers) for insert_sensor.

Default fair-comparison vision matches diffusion:
  RGB-D CNN (per camera) + joints/gripper
  → Temporal Transformer → ACT CVAE → action chunk

Optional: ``vision_backbone: dinov2_*`` uses frozen DINOv2 RGB + depth CNN
(see ``config/default_act_dinov2.yaml``).
"""

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

STATE_DIM = 13
ACTION_DIM = 7

_DINO_HUB = "facebookresearch/dinov2"
_DINO_NAMES = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")


@dataclass(frozen=True)
class InsertionPolicyConfig:
    history_len: int = 8
    action_horizon: int = 8
    temporal: Literal["transformer", "gru"] = "transformer"
    latent_dim: int = 256
    img_width: int = 224
    img_height: int = 224
    depth_min_m: float = DEFAULT_DEPTH_MIN_M
    depth_max_m: float = DEFAULT_DEPTH_MAX_M
    num_cameras: int = 2
    use_depth: bool = True
    vision_backbone: str = "cnn"  # "cnn" (same as diffusion) or dinov2_*14
    freeze_backbone: bool = False
    # ACT CVAE
    style_dim: int = 32
    kl_weight: float = 10.0
    # legacy yaml keys ignored
    diffusion_steps: int = 0
    inference_steps: int = 0
    lambda_world: float = 0.0
    lambda_smooth: float = 0.0

    @property
    def uses_dinov2(self) -> bool:
        return self.vision_backbone.startswith("dinov2")

    @staticmethod
    def from_mapping(data: dict[str, Any]) -> InsertionPolicyConfig:
        temporal = str(data.get("temporal", "transformer")).strip().lower()
        if temporal not in ("gru", "transformer"):
            raise ValueError(f"temporal must be gru or transformer, got {temporal!r}")
        n_cam = int(data.get("num_cameras", 2))
        if n_cam < 1 or n_cam > 2:
            raise ValueError(f"num_cameras must be 1 or 2, got {n_cam}")
        backbone = str(data.get("vision_backbone", "cnn")).strip().lower()
        if backbone not in ("cnn", *_DINO_NAMES):
            raise ValueError(
                f"vision_backbone must be cnn or one of {_DINO_NAMES}, got {backbone!r}"
            )
        img_w = int(data.get("img_width", 224 if backbone.startswith("dinov2") else 160))
        img_h = int(data.get("img_height", 224 if backbone.startswith("dinov2") else 120))
        if backbone.startswith("dinov2") and (img_w % 14 != 0 or img_h % 14 != 0):
            raise ValueError(f"DINOv2 needs H/W multiples of 14, got {img_h}x{img_w}")
        freeze_default = backbone.startswith("dinov2")
        return InsertionPolicyConfig(
            history_len=int(data.get("history_len", 8)),
            action_horizon=int(data.get("action_horizon", 8)),
            temporal=temporal,  # type: ignore[arg-type]
            latent_dim=int(data.get("latent_dim", 256)),
            img_width=img_w,
            img_height=img_h,
            depth_min_m=float(data.get("depth_min_m", DEFAULT_DEPTH_MIN_M)),
            depth_max_m=float(data.get("depth_max_m", DEFAULT_DEPTH_MAX_M)),
            num_cameras=n_cam,
            use_depth=bool(data.get("use_depth", True)),
            vision_backbone=backbone,
            freeze_backbone=bool(data.get("freeze_backbone", freeze_default)),
            style_dim=int(data.get("style_dim", 32)),
            kl_weight=float(data.get("kl_weight", 10.0)),
            diffusion_steps=int(data.get("diffusion_steps", 0)),
            inference_steps=int(data.get("inference_steps", 0)),
            lambda_world=float(data.get("lambda_world", 0.0)),
            lambda_smooth=float(data.get("lambda_smooth", 0.0)),
        )


def load_dinov2(name: str = "dinov2_vits14") -> nn.Module:
    return torch.hub.load(_DINO_HUB, name, trust_repo=True)


def preprocess_rgbd_batch(
    color_rgb: np.ndarray,
    depth_m: np.ndarray,
    *,
    out_w: int,
    out_h: int,
    depth_min_m: float,
    depth_max_m: float,
    use_depth: bool = True,
) -> np.ndarray:
    import cv2

    n = color_rgb.shape[0]
    c_out = 4 if use_depth else 3
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
    span = max(depth_max_m - depth_min_m, 1e-6)
    for i in range(n):
        rgb = color_rgb[i]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        rgb_small = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb_f = rgb_small.astype(np.float32) / 255.0
        rgb_f = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD
        out[i, 0:3] = np.transpose(rgb_f, (2, 0, 1))
        if use_depth:
            depth = depth_m[i].astype(np.float32)
            depth_small = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            valid = depth_small > 1e-4
            depth_norm = np.zeros_like(depth_small, dtype=np.float32)
            depth_norm[valid] = np.clip((depth_small[valid] - depth_min_m) / span, 0.0, 1.0)
            out[i, 3] = depth_norm
    return out


class RgbdEncoder(nn.Module):
    def __init__(self, *, in_channels: int = 4, out_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
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


class VisionStateEncoder(nn.Module):
    """Per-timestep: N cameras RGB(+D) + joints/gripper → latent token."""

    def __init__(self, cfg: InsertionPolicyConfig) -> None:
        super().__init__()
        self.num_cameras = int(cfg.num_cameras)
        self.use_depth = bool(cfg.use_depth)
        self.uses_dinov2 = cfg.uses_dinov2
        self.freeze_backbone = bool(cfg.freeze_backbone)
        self.rgb_backbone: nn.Module | None = None
        self.rgb_proj: nn.Module | None = None
        self.depth_encoder: nn.Module | None = None
        self.img_encoders: nn.ModuleList | None = None

        if self.uses_dinov2:
            self.rgb_backbone = load_dinov2(cfg.vision_backbone)
            if self.freeze_backbone:
                for p in self.rgb_backbone.parameters():
                    p.requires_grad = False
                self.rgb_backbone.eval()
            emb = int(getattr(self.rgb_backbone, "embed_dim", 384))
            self.rgb_proj = nn.Sequential(
                nn.Linear(emb, 128),
                nn.ReLU(inplace=True),
            )
            if self.use_depth:
                self.depth_encoder = RgbdEncoder(in_channels=1, out_dim=64)
            cam_dim = 128 + (64 if self.use_depth else 0)
        else:
            in_ch = 4 if self.use_depth else 3
            self.img_encoders = nn.ModuleList(
                [RgbdEncoder(in_channels=in_ch, out_dim=128) for _ in range(self.num_cameras)]
            )
            cam_dim = 128

        self.state_encoder = MlpEncoder(STATE_DIM, 64)
        fuse_in = cam_dim * self.num_cameras + 64
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, cfg.latent_dim),
            nn.ReLU(inplace=True),
        )

    def train(self, mode: bool = True) -> VisionStateEncoder:
        super().train(mode)
        if self.rgb_backbone is not None and self.freeze_backbone:
            self.rgb_backbone.eval()
        return self

    def _encode_images(self, rgbd_flat: torch.Tensor) -> torch.Tensor:
        """rgbd_flat: (BT, N, C, H, W) → (BT, N*cam_dim)."""
        if self.uses_dinov2:
            assert self.rgb_backbone is not None and self.rgb_proj is not None
            bt, n = rgbd_flat.shape[:2]
            rgb = rgbd_flat[:, :, :3].reshape(bt * n, 3, *rgbd_flat.shape[-2:])
            if self.freeze_backbone:
                with torch.no_grad():
                    feat = self.rgb_backbone(rgb)
            else:
                feat = self.rgb_backbone(rgb)
            feat = self.rgb_proj(feat).reshape(bt, n, -1)
            if self.use_depth:
                assert self.depth_encoder is not None
                depth = rgbd_flat[:, :, 3:4].reshape(bt * n, 1, *rgbd_flat.shape[-2:])
                dfeat = self.depth_encoder(depth).reshape(bt, n, -1)
                return torch.cat([feat, dfeat], dim=-1).reshape(bt, -1)
            return feat.reshape(bt, -1)

        assert self.img_encoders is not None
        feats = [enc(rgbd_flat[:, i]) for i, enc in enumerate(self.img_encoders)]
        return torch.cat(feats, dim=-1)

    def forward(self, rgbd: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # rgbd: (B, T, N, C, H, W) or legacy (B, T, C, H, W)
        if rgbd.dim() == 5:
            rgbd = rgbd.unsqueeze(2)
        b, t = rgbd.shape[:2]
        rgbd_flat = rgbd.reshape(b * t, *rgbd.shape[2:])
        st_flat = state.reshape(b * t, -1)
        if rgbd_flat.dim() == 4:
            rgbd_flat = rgbd_flat.unsqueeze(1)
        v = self._encode_images(rgbd_flat)
        s = self.state_encoder(st_flat)
        return self.fusion(torch.cat([v, s], dim=-1)).reshape(b, t, -1)


class TemporalTransformer(nn.Module):
    """Observation history → memory tokens (B, T, D)."""

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
            self.pos_embed = None
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


class ActCvae(nn.Module):
    """
    ACT-style CVAE:
      train: q(z | obs_pool, actions) → KL(q||N(0,I)); decode actions from (memory, z)
      infer: z ~ N(0,I); decode
    """

    def __init__(self, cfg: InsertionPolicyConfig) -> None:
        super().__init__()
        d = cfg.latent_dim
        self.action_horizon = cfg.action_horizon
        self.style_dim = cfg.style_dim
        act_flat = cfg.action_horizon * ACTION_DIM

        self.enc = nn.Sequential(
            nn.Linear(d + act_flat, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(d, cfg.style_dim)
        self.logvar = nn.Linear(d, cfg.style_dim)

        self.action_queries = nn.Parameter(torch.zeros(1, cfg.action_horizon, d))
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        self.z_to_token = nn.Linear(cfg.style_dim, d)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=4,
            dim_feedforward=d * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.action_head = nn.Linear(d, ACTION_DIM)

    def encode(self, obs_pool: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obs_pool: (B, D), actions: (B, Ta, Da)
        flat = actions.reshape(actions.shape[0], -1)
        h = self.enc(torch.cat([obs_pool, flat], dim=-1))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, memory: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # memory: (B, T, D), z: (B, style)
        b = memory.shape[0]
        queries = self.action_queries.expand(b, -1, -1) + self.z_to_token(z).unsqueeze(1)
        hs = self.decoder(queries, memory)
        return self.action_head(hs)

    def forward(
        self,
        memory: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs_pool = memory.mean(dim=1)
        mu = logvar = None
        if actions is not None:
            # Encode whenever actions are given (train + val) so KL is available.
            mu, logvar = self.encode(obs_pool, actions)
            z = self.reparameterize(mu, logvar) if self.training else mu
        elif deterministic:
            z = torch.zeros(memory.shape[0], self.style_dim, device=memory.device, dtype=memory.dtype)
        else:
            z = torch.randn(memory.shape[0], self.style_dim, device=memory.device, dtype=memory.dtype)
        pred = self.decode(memory, z)
        out: dict[str, torch.Tensor] = {"action": pred}
        if mu is not None and logvar is not None:
            out["mu"] = mu
            out["logvar"] = logvar
        return out


class InsertionPolicy(nn.Module):
    """Vision → Temporal Transformer → ACT → action chunk."""

    def __init__(self, cfg: InsertionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_encoder = VisionStateEncoder(cfg)
        self.temporal = TemporalTransformer(cfg)
        self.act = ActCvae(cfg)

    def encode_memory(self, rgbd: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        tokens = self.obs_encoder(rgbd, state)
        return self.temporal(tokens)

    def forward(
        self,
        rgbd: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory = self.encode_memory(rgbd, state)
        return self.act(memory, actions=action, deterministic=False)

    @torch.no_grad()
    def predict_action(
        self,
        rgbd: torch.Tensor,
        state: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        self.eval()
        memory = self.encode_memory(rgbd, state)
        out = self.act(memory, actions=None, deterministic=deterministic)
        return out["action"]


def compute_losses(
    model: InsertionPolicy,
    batch: dict[str, torch.Tensor],
    *,
    cfg: InsertionPolicyConfig,
) -> dict[str, torch.Tensor]:
    out = model(batch["rgbd"], batch["state"], batch["action"])
    l_action = F.l1_loss(out["action"], batch["action"])
    mu = out["mu"]
    logvar = out["logvar"]
    # KL(q||N(0,I))
    l_kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = l_action + cfg.kl_weight * l_kl
    return {
        "total": total,
        "action": l_action,
        "kl": l_kl,
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
                "action_horizon": cfg.action_horizon,
                "temporal": cfg.temporal,
                "latent_dim": cfg.latent_dim,
                "img_width": cfg.img_width,
                "img_height": cfg.img_height,
                "depth_min_m": cfg.depth_min_m,
                "depth_max_m": cfg.depth_max_m,
                "num_cameras": cfg.num_cameras,
                "use_depth": cfg.use_depth,
                "vision_backbone": cfg.vision_backbone,
                "freeze_backbone": cfg.freeze_backbone,
                "style_dim": cfg.style_dim,
                "kl_weight": cfg.kl_weight,
                "arch": "act_dinov2_v1",
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
