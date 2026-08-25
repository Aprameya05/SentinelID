"""
Knowledge distillation: teacher ensemble -> MobileNetV3-Large student.

The student covers modules 01 (liveness) + 03 (face recognition) + 04 (behavioral).
These three are the most latency-critical for an on-device pipeline.
Modules 02 (deepfake) and 05 (document) remain server-side as they require
heavier computation and are less time-sensitive.

Distillation strategy:
  - Intermediate feature matching (FitNets-style) on layer 3 and layer 4
  - Output KL divergence between teacher soft-labels and student logits
  - Hard-label cross-entropy for the primary tasks
  - Temperature annealing: T=4 early, T=2 mid, T=1 final

Target: MobileNetV3-Large at INT8 < 200ms on Snapdragon 778G
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from typing import NamedTuple


class StudentOutput(NamedTuple):
    liveness_score: Tensor   # (B,) P(live)
    embedding: Tensor        # (B, 256) face embedding (smaller than teacher's 512)
    au_signal: Tensor        # (B,) behavioral liveness signal
    logits_liveness: Tensor  # (B,) raw logit for distillation
    logits_au: Tensor        # (B,) raw logit for behavioral


# ──────────────────────────────────────────────────────────────────────────────
# Student network (MobileNetV3-Large + multi-head)
# ──────────────────────────────────────────────────────────────────────────────

class SentinelEdgeModel(nn.Module):
    """
    Lightweight student model for on-device inference.
    Single backbone, three output heads.
    Total params: ~5.4M (MobileNetV3-Large base)
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        base = mobilenet_v3_large(weights=weights)

        # Feature extractor (everything up to the classifier)
        self.features = base.features
        self.avgpool = base.avgpool
        feat_dim = 960  # MobileNetV3-Large output channels

        # Project to shared embedding
        self.shared_proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Liveness head
        self.liveness_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Hardswish(),
            nn.Linear(64, 1),
        )

        # Behavioral (AU liveness signal) head
        self.au_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Hardswish(),
            nn.Linear(64, 1),
        )

    def forward(self, images: Tensor) -> StudentOutput:
        feat = self.features(images)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)

        embedding = self.shared_proj(feat)

        logits_liveness = self.liveness_head(embedding).squeeze(1)
        logits_au = self.au_head(embedding).squeeze(1)

        return StudentOutput(
            liveness_score=torch.sigmoid(logits_liveness),
            embedding=F.normalize(embedding, p=2, dim=1),
            au_signal=torch.sigmoid(logits_au),
            logits_liveness=logits_liveness,
            logits_au=logits_au,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Distillation loss
# ──────────────────────────────────────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    Combined distillation loss:
      - KL divergence between teacher soft labels and student logits (temperature T)
      - Hard label cross-entropy
      - Embedding feature matching (L2 distance between projected embeddings)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,   # weight for KL (distillation) vs hard label
        beta: float = 0.3,    # weight for embedding matching
    ):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = beta

        # Project teacher 512-d embedding to student 256-d for matching
        self.emb_matcher = nn.Linear(512, 256, bias=False)

    def set_temperature(self, T: float):
        """Anneal temperature during training."""
        self.T = T

    def forward(
        self,
        student: StudentOutput,
        teacher_liveness_logit: Tensor,
        teacher_embedding: Tensor,
        hard_labels: Tensor,
    ) -> dict[str, Tensor]:

        # KL divergence (soft labels from teacher at temperature T)
        T = self.T
        soft_teacher = F.sigmoid(teacher_liveness_logit / T)
        soft_student = F.sigmoid(student.logits_liveness / T)

        # Binary KL: sum(p*log(p/q) + (1-p)*log((1-p)/(1-q)))
        kl_loss = (
            soft_teacher * torch.log(soft_teacher / (soft_student + 1e-8) + 1e-8)
            + (1 - soft_teacher) * torch.log((1 - soft_teacher) / (1 - soft_student + 1e-8) + 1e-8)
        ).mean() * (T ** 2)

        # Hard label loss
        hard_loss = F.binary_cross_entropy(
            student.liveness_score, hard_labels.float()
        )

        # Embedding matching (FitNets-style)
        teacher_proj = self.emb_matcher(teacher_embedding)
        emb_loss = F.mse_loss(student.embedding, teacher_proj.detach())

        total = self.alpha * kl_loss + (1 - self.alpha) * hard_loss + self.beta * emb_loss

        return {
            "total": total,
            "kl": kl_loss,
            "hard": hard_loss,
            "embedding": emb_loss,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Export utilities
# ──────────────────────────────────────────────────────────────────────────────

class EdgeDistiller:
    """Orchestrates training and export of the edge model."""

    def __init__(self, student: SentinelEdgeModel):
        self.student = student

    def export_onnx(self, path: str, input_size: tuple = (1, 3, 224, 224)):
        self.student.eval()
        dummy = torch.randn(input_size)
        torch.onnx.export(
            self.student,
            dummy,
            path,
            input_names=["image"],
            output_names=["liveness_score", "embedding", "au_signal", "logits_liveness", "logits_au"],
            dynamic_axes={"image": {0: "batch"}},
            opset_version=17,
        )
        print(f"ONNX model exported to {path}")

    def export_tflite(self, onnx_path: str, tflite_path: str, quantize: bool = True):
        """
        Convert ONNX -> TFLite with optional INT8 quantization.
        Requires onnx2tf or ai-edge-torch.
        """
        try:
            import onnx2tf
            onnx2tf.convert(
                input_onnx_file_path=onnx_path,
                output_folder_path=tflite_path.rsplit("/", 1)[0],
                non_verbose=True,
                quant_type="per-tensor" if quantize else None,
            )
            print(f"TFLite model exported to {tflite_path}")
        except ImportError:
            print("onnx2tf not installed. Run: pip install onnx2tf")

    def benchmark(self, num_runs: int = 100) -> dict[str, float]:
        """Measure latency on CPU (proxy for mobile inference)."""
        import time
        self.student.eval()
        dummy = torch.randn(1, 3, 224, 224)
        times = []
        with torch.inference_mode():
            for _ in range(10):  # warmup
                self.student(dummy)
            for _ in range(num_runs):
                t0 = time.perf_counter()
                self.student(dummy)
                times.append((time.perf_counter() - t0) * 1000)
        return {
            "mean_ms": float(torch.tensor(times).mean()),
            "p95_ms": float(torch.tensor(times).quantile(0.95)),
            "p99_ms": float(torch.tensor(times).quantile(0.99)),
        }
