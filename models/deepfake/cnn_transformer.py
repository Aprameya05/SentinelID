"""
Deepfake detection via CNN-Transformer hybrid with frequency-domain analysis.

Two parallel branches:
  1. Spatial branch: EfficientNet-B4 on the RGB image
  2. Frequency branch: FFT magnitude spectrum as auxiliary input channel

A cross-attention transformer fuses them. The intuition: real faces captured
by a camera sensor have a different spectral signature than GAN/diffusion
outputs because of the camera's optical transfer function, sensor noise, and
JPEG compression pipeline. These artifacts survive face-swapping but not the
regeneration process.

Reference: "Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware
Clues" (Li et al., ECCV 2021)
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class DeepfakeOutput(NamedTuple):
    fake_score: Tensor       # (B,) in [0,1], 1 = fake
    spatial_feat: Tensor     # (B, 512)
    frequency_feat: Tensor   # (B, 256)
    fused_feat: Tensor       # (B, 512) for fusion module


# ──────────────────────────────────────────────────────────────────────────────
# Frequency feature extractor
# ──────────────────────────────────────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Extract DCT or FFT magnitude spectrum features.

    We compute the 2D FFT of each channel independently, take the log-magnitude
    spectrum, and pass it through a lightweight CNN. The azimuthal power spectrum
    reveals GAN periodic artifacts that are invisible in the spatial domain.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 16, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    @staticmethod
    def compute_fft_magnitude(x: Tensor) -> Tensor:
        """
        (B, C, H, W) -> (B, C, H, W) log-magnitude FFT spectrum.
        Shifted so DC is at center.
        """
        fft = torch.fft.fft2(x.float(), norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted) + 1e-8
        log_magnitude = torch.log(magnitude)
        # Normalise to [0, 1] per sample
        lo = log_magnitude.flatten(2).min(dim=-1).values[..., None, None]
        hi = log_magnitude.flatten(2).max(dim=-1).values[..., None, None]
        return (log_magnitude - lo) / (hi - lo + 1e-8)

    def forward(self, x: Tensor) -> Tensor:
        freq_map = self.compute_fft_magnitude(x)
        feat = self.cnn(freq_map)
        feat = feat.flatten(1)
        return self.proj(feat)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-attention fusion transformer
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between spatial and frequency features.
    Spatial features attend to frequency cues and vice-versa.
    """

    def __init__(self, spatial_dim: int = 512, freq_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.proj_s = nn.Linear(spatial_dim, 512)
        self.proj_f = nn.Linear(freq_dim, 512)

        self.attn_s2f = nn.MultiheadAttention(512, num_heads, batch_first=True, dropout=0.1)
        self.attn_f2s = nn.MultiheadAttention(512, num_heads, batch_first=True, dropout=0.1)

        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.ffn = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
        )
        self.norm3 = nn.LayerNorm(512)

    def forward(self, spatial: Tensor, freq: Tensor) -> Tensor:
        # (B, 512) -> (B, 1, 512) for attention API
        s = self.proj_s(spatial).unsqueeze(1)
        f = self.proj_f(freq).unsqueeze(1)

        # Spatial attends to frequency
        s_attn, _ = self.attn_s2f(query=s, key=f, value=f)
        s = self.norm1(s + s_attn)

        # Frequency attends to spatial
        f_attn, _ = self.attn_f2s(query=f, key=s, value=s)
        f = self.norm2(f + f_attn)

        # Concatenate and project
        fused = torch.cat([s.squeeze(1), f.squeeze(1)], dim=-1)
        out = self.ffn(fused)
        return self.norm3(out)


# ──────────────────────────────────────────────────────────────────────────────
# Full deepfake detector
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    CNN-Transformer deepfake detector operating in spatial and frequency domains.

    Input: (B, 3, 224, 224) face crop
    Output: DeepfakeOutput (fake_score, spatial_feat, frequency_feat, fused_feat)
    """

    def __init__(self, pretrained: bool = True, spatial_out_dim: int = 512):
        super().__init__()

        # Spatial branch: EfficientNet-B4
        if HAS_TIMM:
            self.spatial_backbone = timm.create_model(
                "efficientnet_b4",
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool="avg",
            )
            spatial_feat_dim = self.spatial_backbone.num_features  # 1792
        else:
            # Fallback: ResNet-50
            from torchvision.models import ResNet50_Weights, resnet50
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
            self.spatial_backbone = nn.Sequential(*list(backbone.children())[:-1])
            spatial_feat_dim = 2048

        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_feat_dim, spatial_out_dim),
            nn.LayerNorm(spatial_out_dim),
        )

        # Frequency branch
        self.freq_branch = FrequencyBranch(out_dim=256)

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            spatial_dim=spatial_out_dim,
            freq_dim=256,
            num_heads=8,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Returns fake_score logit (B,) — apply sigmoid for probability."""
        # Spatial branch
        spatial_raw = self.spatial_backbone(images)
        if spatial_raw.dim() > 2:
            spatial_raw = spatial_raw.flatten(1)
        spatial_feat = self.spatial_proj(spatial_raw)

        # Frequency branch
        freq_feat = self.freq_branch(images)

        # Cross-attention fusion
        fused = self.fusion(spatial_feat, freq_feat)

        # Logit (B,)
        logit = self.classifier(fused).squeeze(1)
        return logit


class DeepfakeLoss(nn.Module):
    """Focal loss for imbalanced real/fake datasets."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        logits: (B,) raw model output (pre-sigmoid)
        labels: (B,) float 0/1
        Returns scalar focal loss.
        """
        labels = labels.float()
        p = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        pt = torch.where(labels == 1, p, 1 - p)
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        focal = alpha_t * (1 - pt).pow(self.gamma) * bce

        return focal.mean()
