"""Cascade nets: proposal, class net, sensor grasp+axis, center keypoint."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

from .dataset_io import CLASS_NAMES, CLASS_TO_IDX

NUM_CLASSES = len(CLASS_NAMES)
CROP_SIZE = 128
FULL_SIZE = 160  # downsampled full-frame for proposal


def _adapt_stem(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
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
        w = conv.weight.data
        if in_channels == 1 and w.shape[1] == 3:
            new.weight[:, 0] = w.mean(dim=1)
        elif in_channels <= w.shape[1]:
            new.weight[:, :in_channels] = w[:, :in_channels]
        else:
            reps = (in_channels + w.shape[1] - 1) // w.shape[1]
            tiled = w.repeat(1, reps, 1, 1)[:, :in_channels]
            new.weight.copy_(tiled * (w.shape[1] / float(in_channels)))
        if conv.bias is not None and new.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _mobilenet_encoder(in_channels: int = 3, pretrained: bool = True) -> tuple[nn.Module, int]:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    net.features[0][0] = _adapt_stem(net.features[0][0], in_channels)
    feat_dim = net.classifier[0].in_features
    net.classifier = nn.Identity()
    return net, feat_dim


class TinyCNN(nn.Module):
    """Small CNN for crops / full-frame maps."""

    def __init__(self, in_channels: int = 3, out_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, out_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.out_dim = out_dim * 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class ProposalDetector(nn.Module):
    """Full-frame center heatmap + coarse box size (normalized)."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(64, 1, 1)
        self.size_head = nn.Conv2d(64, 2, 1)  # log(w), log(h) relative to FULL_SIZE

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.heatmap(feat), self.size_head(feat)


class ClassNet(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder, feat_dim = _mobilenet_encoder(in_channels, pretrained)
        self.head = nn.Linear(feat_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class SensorHead(nn.Module):
    """Grasp UV + holder orientation from RGB+depth+edge.

    Input CHW: [R, G, B, depth_norm, edge].
    Output: (u, v, cos θ, sin θ) — contact in crop UV [0,1], holder axis as unit dir.
    """

    def __init__(self, in_channels: int = 5) -> None:
        super().__init__()
        if in_channels < 5:
            raise ValueError("SensorHead expects RGB+depth+edge (5 channels)")
        self.backbone = TinyCNN(4)
        self.edge_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        fuse_dim = self.backbone.out_dim + 64 * 16
        self.head = nn.Sequential(
            nn.Linear(fuse_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # contact (u,v) + (cos θ, sin θ)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb_d = x[:, :4]
        edge = x[:, 4:5]
        feat = torch.cat(
            [self.backbone(rgb_d), self.edge_layer(edge).flatten(1)],
            dim=1,
        )
        return self.head(feat)


class CenterHead(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.backbone = TinyCNN(in_channels)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def axis_deg_from_kpts(kpts: list) -> float:
    """Backward-compatible wrapper → sensor geometry (holder axis)."""
    from .sensor_geometry import axis_deg_from_sensor_kpts

    return axis_deg_from_sensor_kpts(kpts)


def class_index(name: str) -> int:
    return CLASS_TO_IDX[str(name).lower()]


HeadKind = Literal["sensor", "center"]
