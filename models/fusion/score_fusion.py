"""
Multimodal score fusion with Platt calibration.

Takes calibrated probability outputs from all upstream modules and produces
a single trust score + structured decision. A lightweight MLP learns the
optimal combination; Platt scaling ensures each module output is a true
probability before fusion.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch import Tensor


@dataclass
class VerificationResult:
    trust_score: float
    decision: str                   # "ACCEPT" | "REVIEW" | "REJECT"
    explanations: dict[str, float]  # per-module contribution
    raw_scores: dict[str, float]    # uncalibrated module outputs
    accept_threshold: float = 0.75
    review_threshold: float = 0.45


@dataclass
class FusionInput:
    liveness_score: float | Tensor      # P(live)
    deepfake_score: float | Tensor      # P(fake) -> we invert to P(genuine)
    face_match_score: float | Tensor    # cosine similarity [0, 1]
    au_liveness_signal: float | Tensor  # behavioral liveness P(live)
    document_score: float | Tensor      # P(document_genuine)
    voice_score: Optional[float | Tensor] = None  # P(same_speaker), optional


class PlattCalibrator:
    """
    Per-module Platt (logistic) calibration.
    Fits a logistic regression on held-out validation scores -> labels.
    """

    def __init__(self):
        self.model = LogisticRegression(C=1.0)
        self._fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        scores_2d = scores.reshape(-1, 1)
        self.model.fit(scores_2d, labels)
        self._fitted = True

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Alias for calibrate_batch — maps raw scores to calibrated probabilities."""
        return self.calibrate_batch(scores)

    def calibrate(self, score: float) -> float:
        if not self._fitted:
            return score
        return self.model.predict_proba([[score]])[0, 1]

    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return scores
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


class ScoreFusionModel(nn.Module):
    """
    MLP fusion of calibrated module scores.

    Input: vector of shape (B, n_modules) — one calibrated probability per module.
    Output: (trust_score, logit) where trust_score is sigmoid output in [0,1]
            and logit is the pre-sigmoid value.

    n_modules: number of score inputs (default 5: liveness, deepfake, face, au, doc)
    """

    def __init__(
        self,
        n_modules: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        accept_threshold: float = 0.75,
        review_threshold: float = 0.45,
    ):
        super().__init__()
        self.n_modules = n_modules
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold

        self.mlp = nn.Sequential(
            nn.Linear(n_modules, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Learnable per-module attention weights (for interpretability)
        self.module_attn = nn.Parameter(torch.ones(n_modules) / n_modules)

        # Calibrators (fitted separately on validation data)
        self.calibrators: dict[str, PlattCalibrator] = {}

    def _build_score_vector(self, fusion_input: "FusionInput") -> Tensor:
        def t(x):
            if isinstance(x, Tensor):
                return x.float().flatten()[:1]
            return torch.tensor([float(x)])

        scores = [
            t(fusion_input.liveness_score),
            1.0 - t(fusion_input.deepfake_score),
            t(fusion_input.face_match_score),
            t(fusion_input.au_liveness_signal),
            t(fusion_input.document_score),
        ]
        if self.n_modules > 5 and fusion_input.voice_score is not None:
            scores.append(t(fusion_input.voice_score))
        elif self.n_modules > 5:
            scores.append(torch.tensor([0.5]))

        return torch.cat(scores).unsqueeze(0)

    def forward(self, score_vector: Tensor) -> tuple[Tensor, Tensor]:
        """
        score_vector: (B, n_modules) calibrated scores.
        Returns: (trust_score, logit)
          trust_score: (B,) in [0, 1]
          logit:       (B,) pre-sigmoid
        """
        attn = torch.softmax(self.module_attn, dim=0)
        weighted = score_vector * attn
        logit = self.mlp(weighted).squeeze(-1)      # (B,)
        trust_score = torch.sigmoid(logit)          # (B,)
        return trust_score, logit

    @torch.inference_mode()
    def decide(self, fusion_input: "FusionInput") -> VerificationResult:
        """Full verification decision from a FusionInput."""
        score_vec = self._build_score_vector(fusion_input)
        trust_score, _ = self.forward(score_vec)
        trust_score = trust_score.item()

        if trust_score >= self.accept_threshold:
            decision = "ACCEPT"
        elif trust_score >= self.review_threshold:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        # Per-module contributions via attention weights
        attn = torch.softmax(self.module_attn, dim=0).detach().cpu().numpy()
        module_names = ["liveness", "anti_deepfake", "face_match", "behavioral", "document"]
        if self.n_modules > 5:
            module_names.append("voice")

        scores_np = score_vec.squeeze(0).cpu().numpy()
        contributions = {
            name: float(attn[i] * scores_np[i])
            for i, name in enumerate(module_names)
        }

        raw = {
            "liveness": float(fusion_input.liveness_score if not isinstance(fusion_input.liveness_score, Tensor)
                              else fusion_input.liveness_score.item()),
            "deepfake_fake_prob": float(fusion_input.deepfake_score if not isinstance(fusion_input.deepfake_score, Tensor)
                                        else fusion_input.deepfake_score.item()),
            "face_match": float(fusion_input.face_match_score if not isinstance(fusion_input.face_match_score, Tensor)
                                else fusion_input.face_match_score.item()),
            "behavioral": float(fusion_input.au_liveness_signal if not isinstance(fusion_input.au_liveness_signal, Tensor)
                                else fusion_input.au_liveness_signal.item()),
            "document": float(fusion_input.document_score if not isinstance(fusion_input.document_score, Tensor)
                              else fusion_input.document_score.item()),
        }

        return VerificationResult(
            trust_score=trust_score,
            decision=decision,
            explanations=contributions,
            raw_scores=raw,
        )


class FusionLoss(nn.Module):
    """BCE on logits + confidence penalty encouraging decisive predictions."""

    def __init__(self, confidence_penalty: float = 0.1):
        super().__init__()
        self.confidence_penalty = confidence_penalty

    def forward(self, logit: Tensor, labels: Tensor, trust_score: Tensor) -> Tensor:
        """
        logit:       (B,) pre-sigmoid logits
        labels:      (B,) binary float labels
        trust_score: (B,) sigmoid(logit) — passed in to avoid recomputing
        """
        bce = F.binary_cross_entropy_with_logits(logit, labels.float())
        # Penalty when model is uncertain (trust_score near 0.5)
        confidence = (trust_score - 0.5).abs().mean()
        penalty = F.relu(0.3 - confidence)
        return bce + self.confidence_penalty * penalty
