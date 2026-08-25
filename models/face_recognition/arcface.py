"""
ArcFace face recognition model.

ArcFace: Additive Angular Margin Loss for Deep Face Recognition
Deng et al., CVPR 2019 — https://arxiv.org/abs/1801.07698

Architecture:
  - ResNet-100 backbone (iResNet variant with BN after conv)
  - 512-d L2-normalised embedding
  - Additive angular margin loss during training
  - FAISS flat-IP index for deduplication at inference
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# iResNet blocks (identity-mapped residual as in InsightFace)
# ──────────────────────────────────────────────────────────────────────────────

class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, eps=2e-5)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=2e-5),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    """Identity-mapped ResNet for face recognition (InsightFace variant)."""

    def __init__(self, layers: list[int], dropout: float = 0.0, embedding_dim: int = 512):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=2e-5)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(64,  layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=2e-5)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim, eps=2e-5)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(IBasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet100(embedding_dim: int = 512, dropout: float = 0.0) -> IResNet:
    return IResNet([3, 13, 30, 3], dropout=dropout, embedding_dim=embedding_dim)


def iresnet50(embedding_dim: int = 512, dropout: float = 0.0) -> IResNet:
    return IResNet([3, 4, 14, 3], dropout=dropout, embedding_dim=embedding_dim)


# ──────────────────────────────────────────────────────────────────────────────
# ArcFace loss
# ──────────────────────────────────────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    Additive angular margin loss.

    For class c and embedding x (unit vector):
        logit_c = s * cos(theta_c + m)
        logit_others = s * cos(theta_k)

    s: scale (typically 64)
    m: margin (typically 0.5 radians)
    """

    def __init__(self, num_classes: int, embedding_dim: int = 512,
                 scale: float = 64.0, margin: float = 0.5,
                 easy_margin: bool = False):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)   # cos(pi - m)
        self.mm = math.sin(math.pi - margin) * margin  # sin(pi - m) * m

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        # Normalise embeddings and weight matrix
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # cos(theta) for all classes
        cosine = F.linear(emb_norm, w_norm)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))

        # cos(theta + m) = cos*cos_m - sin*sin_m
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Clamp to handle numerical instability at theta near pi
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot to select target class logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return F.cross_entropy(output, labels)


# ──────────────────────────────────────────────────────────────────────────────
# Full model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ArcFaceModel(nn.Module):
    """
    Complete face recognition model.

    Training: backbone + ArcFace loss head.
    Inference: backbone only, outputs L2-normalised 512-d embedding.
    """

    def __init__(
        self,
        num_classes: int = 93_431,  # MS-Celeb-1M cleaned
        embedding_dim: int = 512,
        backbone: str = "iresnet100",
        scale: float = 64.0,
        margin: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if backbone == "iresnet100":
            self.backbone = iresnet100(embedding_dim=embedding_dim, dropout=dropout)
        elif backbone == "iresnet50":
            self.backbone = iresnet50(embedding_dim=embedding_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.head = ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            scale=scale,
            margin=margin,
        )
        self.embedding_dim = embedding_dim

    def forward(self, images: Tensor, labels: Tensor | None = None) -> Tensor:
        embeddings = self.backbone(images)
        if labels is not None:
            return self.head(embeddings, labels)
        # Inference: return L2-normalised embedding
        return F.normalize(embeddings, p=2, dim=1)

    @torch.inference_mode()
    def embed(self, images: Tensor) -> Tensor:
        """Extract normalised embeddings (inference only)."""
        return F.normalize(self.backbone(images), p=2, dim=1)

    def similarity(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        """Cosine similarity between two embedding batches."""
        return (emb_a * emb_b).sum(dim=1)

    def verify(self, emb_a: Tensor, emb_b: Tensor, threshold: float = 0.45) -> Tensor:
        """Binary same-person decision."""
        return self.similarity(emb_a, emb_b) > threshold


# ──────────────────────────────────────────────────────────────────────────────
# FAISS deduplication index
# ──────────────────────────────────────────────────────────────────────────────

class FaceDeduplicationIndex:
    """
    FAISS flat inner-product index for face deduplication.
    Normalised embeddings turn cosine similarity into inner product.
    """

    def __init__(self, embedding_dim: int = 512, threshold: float = 0.45):
        import faiss
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.threshold = threshold
        self.id_map: list[str] = []

    def add(self, embedding: np.ndarray, identity_id: str):
        assert embedding.shape == (512,), "Embedding must be (512,)"
        self.index.add(embedding[None].astype(np.float32))
        self.id_map.append(identity_id)

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        distances, indices = self.index.search(
            embedding[None].astype(np.float32), top_k
        )
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.id_map[idx], float(dist)))
        return results

    def is_duplicate(self, embedding: np.ndarray) -> tuple[bool, str | None, float]:
        if self.index.ntotal == 0:
            return False, None, 0.0
        results = self.search(embedding, top_k=1)
        if results and results[0][1] >= self.threshold:
            return True, results[0][0], results[0][1]
        return False, None, 0.0

    def __len__(self) -> int:
        return self.index.ntotal
