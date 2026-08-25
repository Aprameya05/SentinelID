"""
Full SentinelID inference pipeline.

Loads all trained modules and runs them in sequence on a selfie + document.
Designed for both server-side (full ensemble) and edge (distilled student) deployment.

Usage:
    pipeline = SentinelPipeline.from_pretrained("checkpoints/")
    result = pipeline.verify(
        selfie_path="face.jpg",
        document_path="passport.jpg",
        audio_path="voice.wav",  # optional
    )
    print(result.decision, result.trust_score)
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from explainability.gradcam import ScoreExplainer
from models.behavioral.au_gnn import ActionUnitGNN, GazeRegressionHead
from models.deepfake.cnn_transformer import DeepfakeDetector
from models.document.layout_intelligence import DocumentIntelligenceModel, DocumentTypeClassifier
from models.face_recognition.arcface import ArcFaceModel, FaceDeduplicationIndex
from models.fusion.score_fusion import ScoreFusionModel, VerificationResult
from models.liveness.depth_liveness import DepthLivenessModel

# MediaPipe 468-pt -> 68-pt landmark subset
MP_TO_68 = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377,
    378, 365, 397, 288, 323, 454, 389, 71, 63, 105,
    66, 107, 336, 296, 334, 293, 301, 168, 197, 5,
    4, 75, 97, 2, 326, 305, 33, 160, 158, 133,
    153, 144, 362, 385, 387, 263, 373, 380, 61, 39,
    37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    91, 146, 76, 185, 40, 38, 87, 178,
]


@dataclass
class PipelineConfig:
    device: str = "cuda"
    liveness_threshold: float = 0.5
    deepfake_threshold: float = 0.5
    face_match_threshold: float = 0.45
    trust_accept: float = 0.75
    trust_review: float = 0.45
    image_size: int = 224
    liveness_image_size: int = 256
    face_image_size: int = 112
    face_embed_dim: int = 512
    n_aus: int = 12


def _face_transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _standard_transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _load_image(path, transform: T.Compose) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def _detect_landmarks(image_path) -> Optional[np.ndarray]:
    """
    Detect 68 facial landmarks using MediaPipe FaceMesh.
    Returns (68, 2) normalized [0,1] array or None.
    """
    try:
        import mediapipe as mp
        frame = cv2.imread(str(image_path))
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        ) as face_mesh:
            result = face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                return None
            lms = result.multi_face_landmarks[0].landmark
            return np.array([[lms[i].x, lms[i].y] for i in MP_TO_68], dtype=np.float32)
    except ImportError:
        return None


def _extract_eye_crop(image_path, landmarks: Optional[np.ndarray], size: int = 64) -> torch.Tensor:
    """Extract normalized eye region crop. Returns (1, 3, size, size)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame = cv2.imread(str(image_path))
    if frame is None or landmarks is None:
        return torch.zeros(1, 3, size, size)
    H, W = frame.shape[:2]
    pts_px = landmarks * np.array([W, H])
    eye_pts = pts_px[36:48]
    ex0 = max(0, int(eye_pts[:, 0].min()) - 10)
    ey0 = max(0, int(eye_pts[:, 1].min()) - 10)
    ex1 = min(W, int(eye_pts[:, 0].max()) + 10)
    ey1 = min(H, int(eye_pts[:, 1].max()) + 10)
    crop = frame[ey0:ey1, ex0:ex1]
    if crop.size == 0:
        return torch.zeros(1, 3, size, size)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop_resized = cv2.resize(crop_rgb, (size, size))
    crop_norm = ((crop_resized - mean) / std).transpose(2, 0, 1)
    return torch.from_numpy(crop_norm).float().unsqueeze(0)


class SentinelPipeline:
    """
    Full 7-module verification pipeline.
    All models load from checkpoint_dir; missing checkpoints run in random-init mode.
    """

    def __init__(self, config: PipelineConfig, checkpoint_dir):
        self.cfg = config
        self.ckpt_dir = Path(checkpoint_dir)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._calibrators = None
        self._load_models()

    def _load(self, model: torch.nn.Module, *filenames: str) -> torch.nn.Module:
        """Load first matching checkpoint filename, warn if none found."""
        model = model.to(self.device)
        for filename in filenames:
            path = self.ckpt_dir / filename
            if path.exists():
                state = torch.load(path, map_location=self.device)
                # Handle nested state dicts from joint checkpoints
                if isinstance(state, dict):
                    if "au_gnn" in state:
                        state = state["au_gnn"]
                    elif "gaze_head" in state:
                        state = state["gaze_head"]
                    elif "model" in state:
                        state = state["model"]
                try:
                    model.load_state_dict(state)
                except RuntimeError:
                    pass  # shape mismatch from wrong file; continue with random init
                break
        model.eval()
        return model

    def _load_models(self):
        self.liveness_model = self._load(
            DepthLivenessModel(backbone="resnet50"),
            "liveness_best.pt", "liveness_latest.pt",
        )
        self.deepfake_model = self._load(
            DeepfakeDetector(pretrained=False),
            "deepfake_best.pt", "deepfake_latest.pt",
        )
        self.face_model = self._load(
            ArcFaceModel(num_classes=1, embedding_dim=self.cfg.face_embed_dim, backbone="iresnet50"),
            "face_model.pt",
        )
        self.face_index = FaceDeduplicationIndex(dim=self.cfg.face_embed_dim)

        # Behavioral: load joint checkpoint
        au_model = ActionUnitGNN(n_landmarks=68, n_aus=self.cfg.n_aus).to(self.device)
        gaze_head = GazeRegressionHead(in_channels=3, hidden_dim=128).to(self.device)
        for fname in ("behavioral_best.pt", "behavioral_latest.pt"):
            path = self.ckpt_dir / fname
            if path.exists():
                state = torch.load(path, map_location=self.device)
                if isinstance(state, dict) and "au_gnn" in state:
                    try:
                        au_model.load_state_dict(state["au_gnn"])
                    except RuntimeError:
                        pass
                    if "gaze_head" in state:
                        try:
                            gaze_head.load_state_dict(state["gaze_head"])
                        except RuntimeError:
                            pass
                break
        self.au_model = au_model.eval()
        self.gaze_head = gaze_head.eval()

        self.doc_model = self._load(
            DocumentIntelligenceModel(image_size=self.cfg.image_size),
            "document_best.pt", "document_latest.pt",
        )

        # Fusion
        self.fusion_model = ScoreFusionModel(n_modules=5).to(self.device)
        fusion_path = self.ckpt_dir / "fusion_best.pt"
        if fusion_path.exists():
            state = torch.load(fusion_path, map_location=self.device)
            try:
                self.fusion_model.load_state_dict(state.get("model", state))
            except RuntimeError:
                pass
            self._calibrators = state.get("calibrators", None)
        self.fusion_model.eval()

    @classmethod
    def from_pretrained(cls, checkpoint_dir, config: Optional[PipelineConfig] = None):
        return cls(config or PipelineConfig(), checkpoint_dir)

    @torch.inference_mode()
    def run_liveness(self, selfie_path) -> tuple[float, Optional[np.ndarray]]:
        img = _load_image(selfie_path, _standard_transform(self.cfg.liveness_image_size)).to(self.device)
        out = self.liveness_model(img)
        score = float(torch.sigmoid(out["liveness_logit"]).item())
        depth = out.get("depth_map")
        depth_np = depth.squeeze().cpu().numpy() if depth is not None else None
        return score, depth_np

    @torch.inference_mode()
    def run_deepfake(self, selfie_path) -> float:
        img = _load_image(selfie_path, _standard_transform(self.cfg.image_size)).to(self.device)
        logit = self.deepfake_model(img)
        return float(torch.sigmoid(logit).item())

    @torch.inference_mode()
    def run_face_recognition(self, selfie_path) -> tuple[np.ndarray, Optional[float]]:
        img = _load_image(selfie_path, _face_transform(self.cfg.face_image_size)).to(self.device)
        embedding = self.face_model.embed(img).cpu().numpy()[0]
        dedup_score = None
        if self.face_index.index.ntotal > 0:
            sims, _ = self.face_index.search(embedding.reshape(1, -1), k=1)
            dedup_score = float(sims[0, 0])
        return embedding, dedup_score

    @torch.inference_mode()
    def run_behavioral(self, selfie_path) -> float:
        landmarks = _detect_landmarks(selfie_path)
        if landmarks is None:
            return 0.5
        lm_t = torch.from_numpy(landmarks).float().unsqueeze(0).to(self.device)
        au_pred, _ = self.au_model(lm_t)
        eye_crop = _extract_eye_crop(selfie_path, landmarks).to(self.device)
        gaze_vec = self.gaze_head(eye_crop)
        au_activity = float((au_pred > 0.5).float().mean().item())
        gaze_plausibility = float(1.0 - abs(gaze_vec.norm(dim=1).clamp(0, 2).item() - 1.0))
        return float(np.clip(0.6 * au_activity + 0.4 * gaze_plausibility, 0.0, 1.0))

    @torch.inference_mode()
    def run_document(self, document_path) -> tuple[float, dict]:
        img = _load_image(document_path, _standard_transform(self.cfg.image_size)).to(self.device)
        L = 128
        token_ids = torch.zeros(1, L, dtype=torch.long, device=self.device)
        bbox = torch.zeros(1, L, 4, dtype=torch.long, device=self.device)
        attn = torch.zeros(1, L, dtype=torch.long, device=self.device)
        out = self.doc_model(img, token_ids, bbox, attn)
        forgery_prob = float(out["forgery_prob"].item())
        doc_type_idx = int(out["doc_type_logits"].argmax(dim=1).item())
        doc_type = DocumentTypeClassifier.DOC_TYPES[doc_type_idx]
        return 1.0 - forgery_prob, {
            "document_type": doc_type,
            "forgery_probability": forgery_prob,
            "forgery_signals": {k: float(v.item()) for k, v in out["forgery_signals"].items()},
        }

    @torch.inference_mode()
    def run_voice(self, audio_path) -> Optional[float]:
        try:
            from speechbrain.pretrained import EncoderClassifier
            clf = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)},
            )
            signal, _ = clf.load_audio(str(audio_path))
            clf.encode_batch(signal.unsqueeze(0))
            return 0.5  # neutral without enrollment reference
        except Exception:
            return None

    def verify(
        self,
        selfie_path,
        document_path=None,
        enrolled_embedding: Optional[np.ndarray] = None,
        audio_path=None,
    ) -> VerificationResult:
        t0 = time.time()
        raw: dict[str, float] = {}

        liveness_score, depth_map = self.run_liveness(selfie_path)
        raw["liveness"] = liveness_score

        deepfake_fake_prob = self.run_deepfake(selfie_path)
        raw["deepfake_fake_prob"] = deepfake_fake_prob

        embedding, dedup_score = self.run_face_recognition(selfie_path)
        if enrolled_embedding is not None:
            e = enrolled_embedding / (np.linalg.norm(enrolled_embedding) + 1e-8)
            emb = embedding / (np.linalg.norm(embedding) + 1e-8)
            face_match = float((np.dot(emb, e) + 1.0) / 2.0)
        else:
            face_match = 0.5
        raw["face_match"] = face_match

        behavioral = self.run_behavioral(selfie_path)
        raw["behavioral"] = behavioral

        document_score = 0.5
        doc_meta: dict = {}
        if document_path is not None:
            document_score, doc_meta = self.run_document(document_path)
        raw["document"] = document_score

        if audio_path is not None:
            voice = self.run_voice(audio_path)
            if voice is not None:
                raw["voice"] = voice

        # Fusion
        score_vec = torch.tensor([
            liveness_score,
            1.0 - deepfake_fake_prob,
            face_match,
            behavioral,
            document_score,
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        if self._calibrators:
            cols = [torch.tensor(c.transform(score_vec[:, i].cpu().numpy())).float()
                    for i, c in enumerate(self._calibrators)]
            score_vec = torch.stack(cols, dim=1).to(self.device)

        trust_score, _ = self.fusion_model(score_vec)
        trust_score = float(trust_score.item())

        if trust_score >= self.cfg.trust_accept:
            decision = "ACCEPT"
        elif trust_score >= self.cfg.trust_review:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        result = VerificationResult(
            trust_score=trust_score,
            decision=decision,
            raw_scores=raw,
            explanations={"latency_ms": round((time.time() - t0) * 1000, 1), **doc_meta},
        )
        result.explanations["text"] = ScoreExplainer.explain(result)
        return result

    def enroll(self, selfie_path) -> np.ndarray:
        embedding, _ = self.run_face_recognition(selfie_path)
        return embedding

    def add_to_dedup_index(self, embedding: np.ndarray):
        self.face_index.add(embedding.reshape(1, -1))
