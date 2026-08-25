"""
Passive 3D liveness detection via monocular depth estimation.

A spoof attack (printed photo, phone screen, paper mask) lacks real 3D facial
geometry. By estimating a per-pixel depth map from the RGB selfie and jointly
training an anti-spoof classifier, we use geometric inconsistency as a liveness
signal that texture-based methods miss.

Architecture:
  - ResNet-50 encoder (shared between depth and liveness heads)
  - FPN-style decoder for dense depth map output (H x W x 1)
  - Liveness classification head on global pooled features
  - BerHu loss for depth, BCE for liveness, contrastive loss across spoof types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights
from typing import NamedTuple


class LivenessOutput(NamedTuple):
    liveness_score: Tensor   # (B,) in [0,1], 1 = live
    depth_map: Tensor        # (B, 1, H, W) pseudo-depth
    embedding: Tensor        # (B, 512) shared features


# ──────────────────────────────────────────────────────────────────────────────
# FPN Depth Decoder
# ──────────────────────────────────────────────────────────────────────────────

class DepthDecoder(nn.Module):
    """
    Feature Pyramid Network decoder for dense depth estimation.
    Takes multi-scale ResNet features and produces a (B, 1, H, W) depth map.
    """

    def __init__(self, encoder_channels: list[int] = [64, 256, 512, 1024, 2048]):
        super().__init__()
        # Lateral 1x1 convolutions for each encoder level
        self.lat5 = nn.Conv2d(encoder_channels[4], 256, 1)
        self.lat4 = nn.Conv2d(encoder_channels[3], 256, 1)
        self.lat3 = nn.Conv2d(encoder_channels[2], 256, 1)
        self.lat2 = nn.Conv2d(encoder_channels[1], 256, 1)

        # Top-down 3x3 convolutions
        self.td4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.td3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.td2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())

        # Final depth head
        self.depth_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),  # depth in [0, 1]
        )

    def forward(self, features: dict[str, Tensor], target_size: tuple[int, int]) -> Tensor:
        c2, c3, c4, c5 = features["c2"], features["c3"], features["c4"], features["c5"]

        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.td4(p4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.td3(p3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.td2(p2)

        depth = self.depth_head(p2)
        depth = F.interpolate(depth, size=target_size, mode="bilinear", align_corners=False)
        return depth


# ──────────────────────────────────────────────────────────────────────────────
# Liveness Classification Head
# ──────────────────────────────────────────────────────────────────────────────

class LivenessHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        pooled = self.pool(features)
        embedding = pooled.flatten(1)
        logit = self.mlp(pooled).squeeze(1)
        return logit, embedding


# ──────────────────────────────────────────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────────────────────────────────────────

class DepthLivenessModel(nn.Module):
    """
    Joint depth estimation + liveness classification model.

    Given a single RGB face crop (3 x 224 x 224), outputs:
      - liveness_score: probability the face is live (not spoofed)
      - depth_map: per-pixel monocular depth estimate (1 x 224 x 224)
      - embedding: 512-d shared representation for downstream fusion
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Strip off final pool + fc, keep multi-scale feature maps
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # (B, 256, 56, 56) for 224 input
        self.layer2 = backbone.layer2   # (B, 512, 28, 28)
        self.layer3 = backbone.layer3   # (B, 1024, 14, 14)
        self.layer4 = backbone.layer4   # (B, 2048, 7, 7)

        self.depth_decoder = DepthDecoder(encoder_channels=[64, 256, 512, 1024, 2048])
        self.liveness_head = LivenessHead(in_dim=2048, hidden_dim=512, dropout=dropout)

        # Project pooled 2048-d to 512-d embedding for fusion module
        self.embedding_proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
        )

    def _extract_features(self, x: Tensor) -> dict[str, Tensor]:
        x = self.stem(x)       # (B, 64, 56, 56)
        c2 = self.layer1(x)   # (B, 256, 56, 56)
        c3 = self.layer2(c2)  # (B, 512, 28, 28)
        c4 = self.layer3(c3)  # (B, 1024, 14, 14)
        c5 = self.layer4(c4)  # (B, 2048, 7, 7)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}

    def forward(self, images: Tensor) -> LivenessOutput:
        B, C, H, W = images.shape
        features = self._extract_features(images)

        # Depth decoder
        depth_map = self.depth_decoder(features, target_size=(H, W))

        # Liveness head
        liveness_logit, pooled = self.liveness_head(features["c5"])
        liveness_score = torch.sigmoid(liveness_logit)

        # Shared embedding for fusion
        embedding = self.embedding_proj(pooled)

        return LivenessOutput(
            liveness_score=liveness_score,
            depth_map=depth_map,
            embedding=embedding,
        )


# ──────────────────────────────────────────────────────────────────────────────
# BerHu depth loss (better than L2 for depth — robust to outliers)
# ──────────────────────────────────────────────────────────────────────────────

class BerHuLoss(nn.Module):
    """
    Reverse Huber (BerHu) loss for depth estimation.
    L1 when |diff| <= c, L2/c otherwise.
    c = 0.2 * max(|diff|) per batch.
    """

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        diff = torch.abs(pred - target)
        if mask is not None:
            diff = diff[mask]
        c = 0.2 * diff.max().detach()
        berhu = torch.where(diff <= c, diff, (diff.pow(2) + c.pow(2)) / (2 * c))
        return berhu.mean()


class LivenessLoss(nn.Module):
    """Combined loss: liveness BCE + depth BerHu + contrastive."""

    def __init__(
        self,
        liveness_weight: float = 1.0,
        depth_weight: float = 0.5,
        contrastive_weight: float = 0.1,
    ):
        super().__init__()
        self.w_live = liveness_weight
        self.w_depth = depth_weight
        self.w_contrast = contrastive_weight
        self.berhu = BerHuLoss()

    def forward(
        self,
        output: LivenessOutput,
        liveness_labels: Tensor,
        depth_targets: Tensor | None = None,
    ) -> dict[str, Tensor]:
        liveness_loss = F.binary_cross_entropy(
            output.liveness_score, liveness_labels.float()
        )

        depth_loss = torch.tensor(0.0, device=liveness_labels.device)
        if depth_targets is not None:
            depth_loss = self.berhu(output.depth_map, depth_targets)

        # Contrastive: live embeddings should cluster, spoof should be far
        live_mask = liveness_labels == 1
        spoof_mask = liveness_labels == 0
        contrast_loss = torch.tensor(0.0, device=liveness_labels.device)
        if live_mask.sum() > 1 and spoof_mask.sum() > 0:
            live_embs = F.normalize(output.embedding[live_mask], dim=1)
            spoof_embs = F.normalize(output.embedding[spoof_mask], dim=1)
            pos_sim = (live_embs @ live_embs.T).mean()
            neg_sim = (live_embs @ spoof_embs.T).mean()
            contrast_loss = F.relu(neg_sim - pos_sim + 0.5)

        total = (
            self.w_live * liveness_loss
            + self.w_depth * depth_loss
            + self.w_contrast * contrast_loss
        )

        return {
            "total": total,
            "liveness": liveness_loss,
            "depth": depth_loss,
            "contrastive": contrast_loss,
        }
