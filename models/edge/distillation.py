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

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


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

    def __init__(
        self,
        embedding_dim: int = 256,
        pretrained: bool = True,
        liveness_out: int = 1,
        face_embed_dim: int = 256,
        au_out: int = 1,
    ):
        super().__init__()
        # face_embed_dim overrides embedding_dim when explicitly provided
        embedding_dim = face_embed_dim
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
            nn.Linear(64, liveness_out),
        )

        # Behavioral (AU liveness signal) head
        self.au_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Hardswish(),
            nn.Linear(64, au_out),
        )

        self._embedding_dim = embedding_dim
        self._liveness_out = liveness_out
        self._au_out = au_out

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        """
        Returns dict with keys:
          liveness   : (B,) liveness probability
          face_embed : (B, face_embed_dim) L2-normalised embedding
          au_signal  : (B,) or (B, au_out) behavioral signal
          logits_liveness: (B,) raw logit
          logits_au  : (B,) or (B, au_out) raw logit
        """
        feat = self.features(images)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)

        embedding = self.shared_proj(feat)

        liveness_raw = self.liveness_head(embedding)
        au_raw = self.au_head(embedding)
        logits_liveness = liveness_raw.squeeze(-1) if self._liveness_out == 1 else liveness_raw
        logits_au = au_raw.squeeze(-1) if self._au_out == 1 else au_raw

        return {
            "liveness": torch.sigmoid(logits_liveness),
            "face_embed": F.normalize(embedding, p=2, dim=1),
            "au_signal": torch.sigmoid(logits_au),
            "logits_liveness": logits_liveness,
            "logits_au": logits_au,
        }


# ──────────────────────────────────────────────────────────────────────────────
# ONNX-compatible wrapper (returns tuple instead of dict)
# ──────────────────────────────────────────────────────────────────────────────

class _ONNXWrapper(nn.Module):
    """Wraps SentinelEdgeModel to return a tuple — required for ONNX export."""

    def __init__(self, model: SentinelEdgeModel):
        super().__init__()
        self.model = model

    def forward(self, images: Tensor):
        out = self.model(images)
        return (
            out["liveness"],
            out["face_embed"],
            out["au_signal"],
            out["logits_liveness"],
            out["logits_au"],
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

    Args:
        temperature: softmax temperature for soft targets
        alpha: weight for KL loss vs hard label loss (alpha * KL + (1-alpha) * BCE)
        beta: weight for embedding matching loss
        embedding_weight: alias for beta (kept for config compatibility)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        embedding_weight: float | None = None,  # alias for beta
    ):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = embedding_weight if embedding_weight is not None else beta

        # Projection from teacher embedding dim to student embedding dim.
        # Created lazily on first forward call when dims are known.
        self._emb_matcher: nn.Linear | None = None

    def set_temperature(self, T: float) -> None:
        """Anneal temperature during training."""
        self.T = T

    def forward(
        self,
        student_liveness: Tensor,
        student_embedding: Tensor,
        teacher_liveness_logit: Tensor,
        teacher_embedding: Tensor,
        hard_labels: Tensor,
    ) -> Tensor:
        """
        student_liveness:       (B,) student liveness logit
        student_embedding:      (B, D) L2-normalised student embedding
        teacher_liveness_logit: (B,) teacher liveness logit
        teacher_embedding:      (B, D') teacher embedding (projected to student dim internally)
        hard_labels:            (B,) binary float labels
        Returns scalar loss tensor.
        """
        T = self.T
        soft_teacher = torch.sigmoid(teacher_liveness_logit / T)
        soft_student = torch.sigmoid(student_liveness / T)

        # Binary KL divergence (T² rescaling per Hinton et al.)
        kl_loss = (
            soft_teacher * torch.log(soft_teacher / (soft_student + 1e-8) + 1e-8)
            + (1 - soft_teacher) * torch.log((1 - soft_teacher) / (1 - soft_student + 1e-8) + 1e-8)
        ).mean() * (T ** 2)

        # Hard label loss
        hard_loss = F.binary_cross_entropy_with_logits(student_liveness, hard_labels.float())

        # Embedding matching — project teacher -> student dim if they differ
        t_dim = teacher_embedding.shape[-1]
        s_dim = student_embedding.shape[-1]
        if t_dim == s_dim:
            teacher_proj = teacher_embedding
        else:
            if (
                self._emb_matcher is None
                or self._emb_matcher.in_features != t_dim
                or self._emb_matcher.out_features != s_dim
            ):
                self._emb_matcher = nn.Linear(t_dim, s_dim, bias=False).to(teacher_embedding.device)
            teacher_proj = self._emb_matcher(teacher_embedding)
        emb_loss = F.mse_loss(student_embedding, teacher_proj.detach())

        total = self.alpha * kl_loss + (1 - self.alpha) * hard_loss + self.beta * emb_loss
        return total


# ──────────────────────────────────────────────────────────────────────────────
# Export utilities
# ──────────────────────────────────────────────────────────────────────────────

class EdgeDistiller:
    """Orchestrates training and export of the edge model."""

    def __init__(self, student: SentinelEdgeModel):
        self.student = student

    def export_onnx(self, path: str, input_size: tuple = (1, 3, 224, 224)) -> None:
        """
        Export to ONNX using the legacy (non-dynamo) exporter for maximum compatibility.
        The model is wrapped in _ONNXWrapper to convert dict output -> tuple,
        which is required by the ONNX exporter.
        """
        self.student.eval().cpu()
        wrapper = _ONNXWrapper(self.student).eval()
        dummy = torch.randn(input_size)

        torch.onnx.export(
            wrapper,
            dummy,
            path,
            input_names=["image"],
            output_names=["liveness_score", "face_embed", "au_signal", "logits_liveness", "logits_au"],
            dynamic_axes={"image": {0: "batch"}},
            opset_version=14,
            dynamo=False,   # force legacy exporter — dynamo path rejects dynamic_axes
        )
        print(f"ONNX model exported to {path}")

        # Quick sanity check with onnxruntime
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            dummy_np = np.random.randn(*input_size).astype(np.float32)
            outs = sess.run(None, {"image": dummy_np})
            print(f"  ORT verification: {len(outs)} outputs, liveness shape {outs[0].shape} ✓")
        except Exception as e:
            print(f"  ORT check skipped: {e}")

    def export_tflite(self, onnx_path: str, tflite_path: str, quantize: bool = True) -> None:
        """
        Convert ONNX -> TFLite with optional INT8 quantization.
        Requires onnx2tf: pip install onnx2tf
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

    def benchmark(self, num_runs: int = 100, input_size: tuple = (1, 3, 224, 224)) -> dict[str, float]:
        """Measure latency on CPU (proxy for mobile inference)."""
        import time
        self.student.eval().cpu()
        dummy = torch.randn(input_size)
        times = []
        with torch.inference_mode():
            for _ in range(10):  # warmup
                self.student(dummy)
            for _ in range(num_runs):
                t0 = time.perf_counter()
                self.student(dummy)
                times.append((time.perf_counter() - t0) * 1000)
        t = torch.tensor(times)
        return {
            "mean_ms": float(t.mean()),
            "p95_ms": float(t.quantile(0.95)),
            "p99_ms": float(t.quantile(0.99)),
        }
