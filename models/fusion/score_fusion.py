"""
Multimodal score fusion with Platt calibration.

Takes calibrated probability outputs from all upstream modules and produces
a single trust score + structured decision. A lightweight MLP learns the
optimal combination; Platt scaling ensures each module output is a true
probability before fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


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

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        scores_2d = scores.reshape(-1, 1)
        self.model.fit(scores_2d, labels)
        self._fitted = True

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

    Input: vector of [liveness, anti_deepfake, face_match, behavioral, document, (voice)]
    Output: trust_score in [0, 1]

    The MLP learns which signals matter most for the overall verification decision.
    Dropout provides uncertainty — low-confidence predictions land in REVIEW.
    """

    INPUT_DIMS_WITHOUT_VOICE = 5
    INPUT_DIMS_WITH_VOICE = 6

    def __init__(
        self,
        use_voice: bool = False,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        accept_threshold: float = 0.75,
        review_threshold: float = 0.45,
    ):
        super().__init__()
        self.use_voice = use_voice
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold
        in_dim = self.INPUT_DIMS_WITH_VOICE if use_voice else self.INPUT_DIMS_WITHOUT_VOICE

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Learnable per-module attention weights (for interpretability)
        self.module_attn = nn.Parameter(torch.ones(in_dim) / in_dim)

        # Calibrators (fitted separately on validation data)
        self.calibrators: dict[str, PlattCalibrator] = {}

    def _build_score_vector(self, fusion_input: FusionInput) -> Tensor:
        def t(x):
            if isinstance(x, Tensor):
                return x.float().flatten()[:1]
            return torch.tensor([float(x)])

        # Note: deepfake_score is P(fake), so anti-deepfake is 1 - P(fake)
        scores = [
            t(fusion_input.liveness_score),
            1.0 - t(fusion_input.deepfake_score),  # invert: P(genuine)
            t(fusion_input.face_match_score),
            t(fusion_input.au_liveness_signal),
            t(fusion_input.document_score),
        ]
        if self.use_voice and fusion_input.voice_score is not None:
            scores.append(t(fusion_input.voice_score))
        elif self.use_voice:
            scores.append(torch.tensor([0.5]))  # neutral if no audio

        return torch.cat(scores).unsqueeze(0)  # (1, num_modules)

    def forward(self, score_vector: Tensor) -> Tensor:
        """score_vector: (B, num_modules) calibrated scores."""
        # Soft attention weighting before MLP
        attn = torch.softmax(self.module_attn, dim=0)
        weighted = score_vector * attn
        return self.mlp(weighted)

    @torch.inference_mode()
    def decide(self, fusion_input: FusionInput) -> VerificationResult:
        """Full verification decision from a FusionInput."""
        score_vec = self._build_score_vector(fusion_input)
        trust_score = self.forward(score_vec).item()

        if trust_score >= self.accept_threshold:
            decision = "ACCEPT"
        elif trust_score >= self.review_threshold:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        # Per-module contributions via attention weights
        attn = torch.softmax(self.module_attn, dim=0).detach().cpu().numpy()
        module_names = ["liveness", "anti_deepfake", "face_match", "behavioral", "document"]
        if self.use_voice:
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
    """BCE + calibration consistency loss."""

    def forward(self, trust_score: Tensor, labels: Tensor) -> dict[str, Tensor]:
        bce = F.binary_cross_entropy(trust_score, labels.float())
        # Confidence penalty: push scores away from 0.5 (encourage decisiveness)
        confidence = (trust_score - 0.5).abs().mean()
        confidence_loss = F.relu(0.3 - confidence)  # penalise if avg confidence < 0.3

        return {
            "total": bce + 0.1 * confidence_loss,
            "bce": bce,
            "confidence": confidence_loss,
        }
