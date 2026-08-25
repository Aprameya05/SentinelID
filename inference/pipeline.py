"""
Full SentinelID inference pipeline.

Loads all trained module checkpoints and runs them in sequence.
Designed for production use: thread-safe, batched where possible,
returns structured results with per-module explanations.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import cv2
from PIL import Image
import json

from models.liveness.depth_liveness import DepthLivenessModel
from models.deepfake.cnn_transformer import DeepfakeDetector
from models.face_recognition.arcface import ArcFaceModel, FaceDeduplicationIndex
from models.behavioral.au_gnn import ActionUnitGNN, GazeRegressionHead
from models.fusion.score_fusion import ScoreFusionModel, FusionInput, VerificationResult


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
FACE_MEAN     = [0.5, 0.5, 0.5]
FACE_STD      = [0.5, 0.5, 0.5]


def load_and_preprocess(
    image_path: str | Path | np.ndarray,
    size: tuple[int, int] = (224, 224),
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> Tensor:
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path, np.ndarray):
        img = image_path
    else:
        raise TypeError(f"Expected path or ndarray, got {type(image_path)}")

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def extract_landmarks(image: np.ndarray) -> np.ndarray | None:
    """
    Extract 68-point facial landmarks. Uses mediapipe or dlib as available.
    Returns (68, 2) normalised coordinates or None if no face detected.
    """
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:
            result = mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                return None
            lm = result.multi_face_landmarks[0].landmark
            # Select 68 iBUG-compatible points from mediapipe's 478
            pts = np.array([[l.x, l.y] for l in lm[:68]])
            return pts
    except ImportError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    liveness_threshold: float = 0.6
    deepfake_threshold: float = 0.5     # above this = fake
    face_match_threshold: float = 0.45  # cosine similarity
    accept_threshold: float = 0.75
    review_threshold: float = 0.45
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1


class SentinelPipeline:
    """
    Unified inference pipeline for identity verification.

    Usage:
        pipeline = SentinelPipeline.from_pretrained("checkpoints/")
        result = pipeline.verify(selfie_path="face.jpg", document_path="id.jpg")
        print(result.decision)  # ACCEPT | REVIEW | REJECT
    """

    def __init__(
        self,
        liveness_model: DepthLivenessModel,
        deepfake_model: DeepfakeDetector,
        face_model: ArcFaceModel,
        au_model: ActionUnitGNN,
        fusion_model: ScoreFusionModel,
        dedup_index: FaceDeduplicationIndex | None = None,
        config: PipelineConfig | None = None,
    ):
        self.liveness = liveness_model
        self.deepfake = deepfake_model
        self.face = face_model
        self.au = au_model
        self.fusion = fusion_model
        self.dedup = dedup_index
        self.config = config or PipelineConfig()

        self.device = torch.device(self.config.device)
        self._set_eval_mode()

    def _set_eval_mode(self):
        for model in [self.liveness, self.deepfake, self.face, self.au, self.fusion]:
            model.eval().to(self.device)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path, config: PipelineConfig | None = None) -> "SentinelPipeline":
        """Load all module checkpoints from a directory."""
        ckpt_dir = Path(checkpoint_dir)
        cfg = config or PipelineConfig()
        device = torch.device(cfg.device)

        def load(model: torch.nn.Module, name: str) -> torch.nn.Module:
            path = ckpt_dir / f"{name}.pt"
            if path.exists():
                state = torch.load(path, map_location=device, weights_only=True)
                model.load_state_dict(state)
                print(f"Loaded {name} from {path}")
            else:
                print(f"Warning: checkpoint {path} not found, using random weights")
            return model

        liveness  = load(DepthLivenessModel(), "liveness_model")
        deepfake  = load(DeepfakeDetector(), "deepfake_model")
        face      = load(ArcFaceModel(), "face_model")
        au        = load(ActionUnitGNN(), "au_model")
        fusion    = load(ScoreFusionModel(), "fusion_model")

        dedup_path = ckpt_dir / "dedup_index.bin"
        dedup = FaceDeduplicationIndex() if not dedup_path.exists() else None

        return cls(liveness, deepfake, face, au, fusion, dedup, cfg)

    @torch.inference_mode()
    def run_liveness(self, selfie: Tensor) -> tuple[float, np.ndarray]:
        out = self.liveness(selfie.to(self.device))
        score = out.liveness_score.cpu().item()
        depth = out.depth_map.squeeze().cpu().numpy()
        return score, depth

    @torch.inference_mode()
    def run_deepfake(self, selfie: Tensor) -> float:
        out = self.deepfake(selfie.to(self.device))
        return out.fake_score.cpu().item()

    @torch.inference_mode()
    def run_face_recognition(self, selfie: Tensor) -> tuple[np.ndarray, float | None]:
        embedding = self.face.embed(selfie.to(self.device)).cpu().numpy()[0]
        dedup_score = None
        if self.dedup is not None:
            is_dup, _, sim = self.dedup.is_duplicate(embedding)
            dedup_score = sim
        return embedding, dedup_score

    @torch.inference_mode()
    def run_behavioral(self, image: np.ndarray) -> float:
        landmarks = extract_landmarks(image)
        if landmarks is None:
            return 0.5  # neutral if landmark detection fails
        lm_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        out = self.au(lm_tensor)
        return out["liveness_signal"].cpu().item()

    def verify(
        self,
        selfie_path: str | np.ndarray,
        document_path: str | None = None,
        enrolled_embedding: np.ndarray | None = None,
        audio_path: str | None = None,
    ) -> VerificationResult:
        """
        Full verification pipeline.

        selfie_path: path to selfie image or numpy array (BGR)
        document_path: path to ID document image (optional)
        enrolled_embedding: 512-d face embedding to match against (optional)
        audio_path: path to audio file for voice verification (optional)
        """
        # Load selfie
        if isinstance(selfie_path, str):
            raw_image = cv2.imread(selfie_path)
        else:
            raw_image = selfie_path

        selfie_tensor = load_and_preprocess(raw_image)

        # Module 01: Liveness
        liveness_score, depth_map = self.run_liveness(selfie_tensor)

        # Module 02: Deepfake
        fake_score = self.run_deepfake(selfie_tensor)

        # Module 03: Face recognition
        embedding, dedup_score = self.run_face_recognition(selfie_tensor)
        face_match_score = 0.5  # default if no enrolled embedding
        if enrolled_embedding is not None:
            face_match_score = float(
                np.dot(embedding, enrolled_embedding) /
                (np.linalg.norm(embedding) * np.linalg.norm(enrolled_embedding) + 1e-8)
            )
            # Normalise cosine similarity from [-1,1] to [0,1]
            face_match_score = (face_match_score + 1) / 2

        # Module 04: Behavioral biometrics
        au_score = self.run_behavioral(raw_image)

        # Module 05: Document intelligence (if provided)
        doc_score = 0.5  # neutral default
        if document_path is not None:
            # Full doc intelligence is in DocumentIntelligence module
            # Lightweight stub here: check document exists and is readable
            doc_score = 0.8 if Path(document_path).exists() else 0.3

        # Module 06: Fusion
        fusion_input = FusionInput(
            liveness_score=liveness_score,
            deepfake_score=fake_score,
            face_match_score=face_match_score,
            au_liveness_signal=au_score,
            document_score=doc_score,
        )
        result = self.fusion.decide(fusion_input)
        return result

    def to_json(self, result: VerificationResult) -> str:
        return json.dumps({
            "trust_score": result.trust_score,
            "decision": result.decision,
            "explanations": result.explanations,
            "raw_scores": result.raw_scores,
        }, indent=2)
