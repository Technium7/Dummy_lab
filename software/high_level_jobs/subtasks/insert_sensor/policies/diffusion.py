"""Diffusion Policy for insert_sensor (no auxiliary heads).

Default: RGB-D CNN + optional dense table XYZ + joints/gripper → Temporal → DDPM.
Optional: ``vision_backbone: dinov2_*`` uses frozen DINOv2 RGB + depth/(XYZ) CNN
(see ``config/default_diffusion_dinov2.yaml``). Supports up to 3 cameras.
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
from high_level_jobs.subtasks.insert_sensor.model import load_dinov2

STATE_DIM = 13
ACTION_DIM = 7
RGBD_CHANNELS = 4
TABLE_XYZ_CHANNELS = 3
_DINO_NAMES = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")


@dataclass(frozen=True)
class DiffusionPolicyConfig:
    history_len: int = 8
    action_horizon: int = 8
    temporal: Literal["gru", "transformer"] = "transformer"
    latent_dim: int = 256
    img_width: int = 160
    img_height: int = 120
    depth_min_m: float = DEFAULT_DEPTH_MIN_M
    depth_max_m: float = DEFAULT_DEPTH_MAX_M
    num_cameras: int = 2
    use_depth: bool = True  # always True for this stack; kept for dataset API
    use_table_xyz: bool = False
    table_calib_path: str = ""
    table_xyz_xy_min_m: float = -0.40
    table_xyz_xy_max_m: float = 0.40
    table_xyz_z_min_m: float = -0.05
    table_xyz_z_max_m: float = 0.35
    vision_backbone: str = "cnn"  # "cnn" or dinov2_*14
    freeze_backbone: bool = False
    diffusion_steps: int = 50
    inference_steps: int = 16
    beta_start: float = 1e-4
    beta_end: float = 0.02

    @property
    def img_channels(self) -> int:
        return RGBD_CHANNELS + (TABLE_XYZ_CHANNELS if self.use_table_xyz else 0)

    @property
    def uses_dinov2(self) -> bool:
        return self.vision_backbone.startswith("dinov2")

    @staticmethod
    def from_mapping(data: dict[str, Any]) -> DiffusionPolicyConfig:
        temporal = str(data.get("temporal", "transformer")).strip().lower()
        if temporal not in ("gru", "transformer"):
            raise ValueError(f"temporal must be gru or transformer, got {temporal!r}")
        n_cam = int(data.get("num_cameras", 2))
        if n_cam < 1 or n_cam > 3:
            raise ValueError(f"num_cameras must be 1, 2, or 3, got {n_cam}")
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
        return DiffusionPolicyConfig(
            history_len=int(data.get("history_len", 8)),
            action_horizon=int(data.get("action_horizon", 8)),
            temporal=temporal,  # type: ignore[arg-type]
            latent_dim=int(data.get("latent_dim", 256)),
            img_width=img_w,
            img_height=img_h,
            depth_min_m=float(data.get("depth_min_m", DEFAULT_DEPTH_MIN_M)),
            depth_max_m=float(data.get("depth_max_m", DEFAULT_DEPTH_MAX_M)),
            num_cameras=n_cam,
            use_depth=True,
            use_table_xyz=bool(data.get("use_table_xyz", False)),
            table_calib_path=str(data.get("table_calib_path", "") or "").strip(),
            table_xyz_xy_min_m=float(data.get("table_xyz_xy_min_m", -0.40)),
            table_xyz_xy_max_m=float(data.get("table_xyz_xy_max_m", 0.40)),
            table_xyz_z_min_m=float(data.get("table_xyz_z_min_m", -0.05)),
            table_xyz_z_max_m=float(data.get("table_xyz_z_max_m", 0.35)),
            vision_backbone=backbone,
            freeze_backbone=bool(data.get("freeze_backbone", freeze_default)),
            diffusion_steps=int(data.get("diffusion_steps", 50)),
            inference_steps=int(data.get("inference_steps", 16)),
            beta_start=float(data.get("beta_start", 1e-4)),
            beta_end=float(data.get("beta_end", 0.02)),
        )


def preprocess_rgbd_batch(
    color_rgb: np.ndarray,
    depth_m: np.ndarray,
    *,
    out_w: int,
    out_h: int,
    depth_min_m: float,
    depth_max_m: float,
    use_depth: bool = True,
    use_table_xyz: bool = False,
    table_xyz_ctx: Any | None = None,
    is_overhead: bool = False,
    table_xyz_maps: np.ndarray | None = None,
) -> np.ndarray:
    """RGB-D preprocess; optional dense table XYZ → 7 channels when enabled."""
    del use_depth  # CNN stack always uses depth channel
    import cv2

    from high_level_jobs.subtasks.insert_sensor.table_xyz import normalize_table_xyz

    n = color_rgb.shape[0]
    n_ch = RGBD_CHANNELS + (TABLE_XYZ_CHANNELS if use_table_xyz else 0)
    out = np.zeros((n, n_ch, out_h, out_w), dtype=np.float32)
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
        if use_table_xyz:
            if is_overhead and table_xyz_maps is not None:
                out[i, 4:7] = normalize_table_xyz(
                    table_xyz_maps[i],
                    table_xyz_ctx.bounds,
                    out_w=out_w,
                    out_h=out_h,
                )
            elif is_overhead and table_xyz_ctx is not None:
                from high_level_jobs.subtasks.insert_sensor.table_xyz import depth_to_table_xyz

                xyz_i = depth_to_table_xyz(depth, table_xyz_ctx)
                out[i, 4:7] = normalize_table_xyz(
                    xyz_i,
                    table_xyz_ctx.bounds,
                    out_w=out_w,
                    out_h=out_h,
                )
            # else: leave XYZ zeros (side / side2)
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
    """Per-timestep: N cameras RGB-D (+ table XYZ) + joints/gripper → latent token."""

    def __init__(self, cfg: DiffusionPolicyConfig) -> None:
        super().__init__()
        self.num_cameras = int(cfg.num_cameras)
        self.use_table_xyz = bool(cfg.use_table_xyz)
        self.uses_dinov2 = cfg.uses_dinov2
        self.freeze_backbone = bool(cfg.freeze_backbone)
        self.rgb_backbone: nn.Module | None = None
        self.rgb_proj: nn.Module | None = None
        self.aux_encoder: nn.Module | None = None
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
            # Depth (+ optional table XYZ) as aux CNN; side/side2 XYZ are zero-padded.
            aux_ch = 1 + (TABLE_XYZ_CHANNELS if self.use_table_xyz else 0)
            self.aux_encoder = RgbdEncoder(in_channels=aux_ch, out_dim=64)
            cam_dim = 128 + 64
        else:
            in_ch = cfg.img_channels
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
            assert self.aux_encoder is not None
            bt, n = rgbd_flat.shape[:2]
            rgb = rgbd_flat[:, :, :3].reshape(bt * n, 3, *rgbd_flat.shape[-2:])
            if self.freeze_backbone:
                with torch.no_grad():
                    feat = self.rgb_backbone(rgb)
            else:
                feat = self.rgb_backbone(rgb)
            feat = self.rgb_proj(feat).reshape(bt, n, -1)
            # Channels 3: depth; 4:7 table XYZ when enabled
            aux = rgbd_flat[:, :, 3:].reshape(bt * n, -1, *rgbd_flat.shape[-2:])
            afeat = self.aux_encoder(aux).reshape(bt, n, -1)
            return torch.cat([feat, afeat], dim=-1).reshape(bt, -1)

        assert self.img_encoders is not None
        feats = [enc(rgbd_flat[:, i]) for i, enc in enumerate(self.img_encoders)]
        return torch.cat(feats, dim=-1)

    def forward(self, rgbd: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, cfg: DiffusionPolicyConfig) -> None:
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
            return out[:, -1, :]
        pos = self.pos_embed[:, : x.shape[1], :]
        out = self.core(x + pos)
        return out[:, -1, :]


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalActionUNet(nn.Module):
    """Lightweight MLP denoiser for action chunks (B, Ta, Da)."""

    def __init__(self, *, action_dim: int, horizon: int, cond_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        flat = horizon * action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.net = nn.Sequential(
            nn.Linear(flat + hidden + hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, flat),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        t_emb = self.time_mlp(timesteps)
        c_emb = self.cond_proj(cond)
        out = self.net(torch.cat([flat, t_emb, c_emb], dim=-1))
        return out.reshape(b, self.horizon, self.action_dim)


class DiffusionScheduler:
    """DDPM linear beta schedule."""

    def __init__(
        self,
        *,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        }

    def _to(self, device: torch.device) -> None:
        for k, v in self.register.items():
            self.register[k] = v.to(device)

    def add_noise(
        self, original: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        self._to(original.device)
        sqrt_alpha = self.register["sqrt_alphas_cumprod"][timesteps]
        sqrt_one_m = self.register["sqrt_one_minus_alphas_cumprod"][timesteps]
        while sqrt_alpha.dim() < original.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_m = sqrt_one_m.unsqueeze(-1)
        return sqrt_alpha * original + sqrt_one_m * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        self._to(sample.device)
        t = timestep
        beta = self.register["betas"][t]
        alpha = self.register["alphas"][t]
        alpha_bar = self.register["alphas_cumprod"][t]
        alpha_bar_prev = (
            self.register["alphas_cumprod"][t - 1]
            if t > 0
            else torch.tensor(1.0, device=sample.device)
        )
        pred_x0 = (sample - torch.sqrt(1.0 - alpha_bar) * model_output) / torch.sqrt(alpha_bar)
        pred_x0 = torch.clamp(pred_x0, -4.0, 4.0)
        coef1 = torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar)
        coef2 = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        mean = coef1 * pred_x0 + coef2 * sample
        if t > 0:
            noise = torch.randn_like(sample)
            sigma = torch.sqrt(beta)
            return mean + sigma * noise
        return mean


class DiffusionPolicy(nn.Module):
    """Diffusion Policy: temporal conditioning + action-chunk denoiser (no aux heads)."""

    def __init__(self, cfg: DiffusionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_encoder = VisionStateEncoder(cfg)
        self.temporal = TemporalTransformer(cfg)
        self.noise_pred = ConditionalActionUNet(
            action_dim=ACTION_DIM,
            horizon=cfg.action_horizon,
            cond_dim=cfg.latent_dim,
            hidden=cfg.latent_dim,
        )
        self.scheduler = DiffusionScheduler(
            num_train_timesteps=cfg.diffusion_steps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )

    def encode_obs(self, rgbd: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        tokens = self.obs_encoder(rgbd, state)
        return self.temporal(tokens)

    def forward(
        self,
        rgbd: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cond = self.encode_obs(rgbd, state)
        b = action.shape[0]
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0, self.cfg.diffusion_steps, (b,), device=action.device, dtype=torch.long
        )
        noisy = self.scheduler.add_noise(action, noise, timesteps)
        noise_pred = self.noise_pred(noisy, timesteps, cond)
        return {"noise": noise, "noise_pred": noise_pred, "cond": cond}

    @torch.no_grad()
    def sample_action(
        self,
        rgbd: torch.Tensor,
        state: torch.Tensor,
        *,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        cond = self.encode_obs(rgbd, state)
        b = rgbd.shape[0]
        steps = int(num_inference_steps or self.cfg.inference_steps)
        steps = max(1, min(steps, self.cfg.diffusion_steps))
        sample = torch.randn(
            b, self.cfg.action_horizon, ACTION_DIM, device=rgbd.device, dtype=rgbd.dtype
        )
        all_t = torch.linspace(
            self.cfg.diffusion_steps - 1, 0, steps, device=rgbd.device
        ).long()
        for t in all_t.tolist():
            t_batch = torch.full((b,), int(t), device=rgbd.device, dtype=torch.long)
            eps = self.noise_pred(sample, t_batch, cond)
            sample = self.scheduler.step(eps, int(t), sample)
        return sample

    def predict_action(
        self,
        rgbd: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.sample_action(rgbd, state)


def compute_losses(
    model: DiffusionPolicy,
    batch: dict[str, torch.Tensor],
    *,
    cfg: DiffusionPolicyConfig,
) -> dict[str, torch.Tensor]:
    del cfg
    out = model(batch["rgbd"], batch["state"], batch["action"])
    l_diff = F.mse_loss(out["noise_pred"], out["noise"])
    return {"total": l_diff, "diffusion": l_diff}


def save_policy(
    path: Path,
    model: DiffusionPolicy,
    cfg: DiffusionPolicyConfig,
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
                "use_depth": True,
                "use_table_xyz": cfg.use_table_xyz,
                "table_calib_path": cfg.table_calib_path,
                "table_xyz_xy_min_m": cfg.table_xyz_xy_min_m,
                "table_xyz_xy_max_m": cfg.table_xyz_xy_max_m,
                "table_xyz_z_min_m": cfg.table_xyz_z_min_m,
                "table_xyz_z_max_m": cfg.table_xyz_z_max_m,
                "vision_backbone": cfg.vision_backbone,
                "freeze_backbone": cfg.freeze_backbone,
                "diffusion_steps": cfg.diffusion_steps,
                "inference_steps": cfg.inference_steps,
                "beta_start": cfg.beta_start,
                "beta_end": cfg.beta_end,
                "arch": "diffusion_policy_v1",
            },
            "norm_stats": norm_stats or {},
            "meta": {"arch": "diffusion_policy_v1", **(extra or {})},
        },
        path,
    )


def load_policy(
    path: Path,
    *,
    device: str | None = None,
) -> tuple[DiffusionPolicy, DiffusionPolicyConfig, dict[str, Any]]:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path.resolve(), map_location=torch_device, weights_only=False)
    cfg = DiffusionPolicyConfig.from_mapping(payload.get("cfg", {}))
    model = DiffusionPolicy(cfg).to(torch_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg, payload.get("norm_stats", {})
