"""Cascade models: contact (mono) then phase (mono+|diff|) on contact only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

# Stage-2 classes among contact frames (SETTLED is labeled for contact; not a phase class)
PHASE_NAMES = ("TOUCHING", "FLOWING")
PHASE_TO_IDX = {n: i for i, n in enumerate(PHASE_NAMES)}
NUM_PHASE_CLASSES = len(PHASE_NAMES)


def _adapt_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv
    new = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        w = conv.weight.data  # (out, in_old, k, k)
        if in_channels == 1 and w.shape[1] == 3:
            # True mono stem: average RGB filters
            new.weight[:, 0] = w.mean(dim=1)
        elif in_channels == 2 and w.shape[1] == 3:
            # gray + |diff|: both start from luminance-averaged RGB kernels
            avg = w.mean(dim=1)
            new.weight[:, 0] = avg
            new.weight[:, 1] = avg
        elif in_channels <= w.shape[1]:
            new.weight[:, :in_channels] = w[:, :in_channels]
        else:
            reps = (in_channels + w.shape[1] - 1) // w.shape[1]
            tiled = w.repeat(1, reps, 1, 1)[:, :in_channels]
            new.weight.copy_(tiled * (w.shape[1] / float(in_channels)))
        if conv.bias is not None and new.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _make_encoder(backbone: str, pretrained: bool, in_channels: int) -> tuple[nn.Module, int]:
    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        net.features[0][0] = _adapt_conv(net.features[0][0], in_channels)
        feat_dim = net.classifier[0].in_features
        net.classifier = nn.Identity()
        return net, feat_dim
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.conv1 = _adapt_conv(net.conv1, in_channels)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim
    raise ValueError(f"Unsupported backbone: {backbone}")


class ContactNet(nn.Module):
    """Stage 1: mono gray → contact logit (no temporal channels)."""

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        in_channels: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder, feat_dim = _make_encoder(backbone, pretrained, in_channels)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return self.head(feats).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class PhaseNet(nn.Module):
    """Stage 2: mono+|diff| → TOUCHING / FLOWING (contact frames only)."""

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        in_channels: int = 2,
        num_classes: int = NUM_PHASE_CLASSES,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder, feat_dim = _make_encoder(backbone, pretrained, in_channels)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return self.head(feats)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


class CascadeEpoxyModel(nn.Module):
    """Contact then phase. Phase is only meaningful when contact is true."""

    def __init__(self, contact: ContactNet, phase: PhaseNet) -> None:
        super().__init__()
        self.contact = contact
        self.phase = phase

    def predict(
        self,
        rgb: torch.Tensor,
        six: torch.Tensor | None = None,
        contact_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        gray: (B,1,H,W) for contact
        temporal: (B,2,H,W) gray+|diff| for phase
        """
        contact_p = self.contact.predict_proba(rgb)
        B = contact_p.shape[0]
        phase_probs = torch.zeros(B, NUM_PHASE_CLASSES, device=contact_p.device)
        phase_idx = torch.zeros(B, dtype=torch.long, device=contact_p.device)
        if six is not None:
            phase_probs = self.phase.predict_proba(six)
            phase_idx = phase_probs.argmax(dim=-1)
        return {
            "contact_probability": contact_p,
            "phase_probs": phase_probs,
            "phase_index": phase_idx,
            "phase_names": PHASE_NAMES,
        }


def build_contact_model(cfg: dict[str, Any] | None = None, **overrides: Any) -> ContactNet:
    m = (cfg or {}).get("model", {})
    kwargs = {
        "backbone": m.get("backbone", "mobilenet_v3_small"),
        "pretrained": bool(m.get("pretrained", True)),
        "in_channels": int(m.get("contact_in_channels", 1)),
    }
    kwargs.update(overrides)
    return ContactNet(**kwargs)


def build_phase_model(cfg: dict[str, Any] | None = None, **overrides: Any) -> PhaseNet:
    m = (cfg or {}).get("model", {})
    kwargs = {
        "backbone": m.get("backbone", "mobilenet_v3_small"),
        "pretrained": bool(m.get("pretrained", True)),
        "in_channels": int(m.get("phase_in_channels", 2)),
        "num_classes": NUM_PHASE_CLASSES,
    }
    kwargs.update(overrides)
    return PhaseNet(**kwargs)


def build_cascade(cfg: dict[str, Any] | None = None) -> CascadeEpoxyModel:
    return CascadeEpoxyModel(build_contact_model(cfg), build_phase_model(cfg))


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    cfg: dict[str, Any] | None = None,
) -> tuple[CascadeEpoxyModel, dict[str, Any]]:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", cfg)
    model = build_cascade(model_cfg)
    if "contact_state" in ckpt and "phase_state" in ckpt:
        model.contact.load_state_dict(ckpt["contact_state"])
        model.phase.load_state_dict(ckpt["phase_state"])
    elif "model_state" in ckpt:
        # Legacy dual-head: refuse silently partial — rebuild fresh heads
        pass
    model.to(device)
    model.eval()
    return model, ckpt


# Back-compat aliases used by older imports
DualHeadMobileNet = CascadeEpoxyModel
build_model = build_cascade
