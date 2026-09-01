"""
inference/pipeline.py — SentinelID Production Inference Pipeline

Loads all trained modules and runs them in sequence on a selfie + optional document.
Designed for both server-side (full ensemble) and edge (distilled student) deployment.

Usage:
    pipeline = SentinelPipeline.from_pretrained("checkpoints/")
    result = pipeline.verify(selfie_path="face.jpg", document_path="passport.jpg")
    print(result.decision, result.trust_score)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
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

log = logging.getLogger("sentinelid.pipeline")

# MediaPipe 468-pt → 68-pt landmark subset
MP_TO_68 = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377,
    378, 365, 397, 288, 323, 454, 389, 71, 63, 105,
    66, 107, 336, 296, 334, 293, 301, 168, 197, 5,
    4, 75, 97, 2, 326, 305, 33, 160, 158, 133,
    153, 144, 362, 385, 387, 263, 373, 380, 61, 39,
    37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    91, 146, 76, 185, 40, 38, 87, 178,
]

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    device: str = "cuda"
    # Thresholds
    liveness_threshold: float = 0.5
    deepfake_threshold: float = 0.5
    face_match_threshold: float = 0.45
    trust_accept: float = 0.75
    trust_review: float = 0.45
    # Image sizes
    image_size: int = 224
    liveness_image_size: int = 256
    face_image_size: int = 112
    # Model dims
    face_embed_dim: int = 512
    n_aus: int = 12
    # Runtime
    fp16: bool = False                       # half-precision forward pass on GPU
    warmup_iters: int = 3                    # warm-up runs on load (reduces first-call latency)
    module_timeout_s: float = 5.0            # per-module soft timeout (logged, not enforced as hard kill)

    def resolve_device(self) -> torch.device:
        if self.device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable — falling back to CPU.")
            return torch.device("cpu")
        return torch.device(self.device)


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

_TRANSFORM_CACHE: dict[tuple, T.Compose] = {}


def _face_transform(size: int) -> T.Compose:
    key = ("face", size)
    if key not in _TRANSFORM_CACHE:
        _TRANSFORM_CACHE[key] = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return _TRANSFORM_CACHE[key]


def _standard_transform(size: int) -> T.Compose:
    key = ("std", size)
    if key not in _TRANSFORM_CACHE:
        _TRANSFORM_CACHE[key] = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _TRANSFORM_CACHE[key]


def _load_image(path, transform: T.Compose, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Landmark / gaze helpers
# ──────────────────────────────────────────────────────────────────────────────

def _detect_landmarks(image_path) -> Optional[np.ndarray]:
    """
    Detect 68 facial landmarks via MediaPipe FaceMesh.
    Returns (68, 2) normalized [0,1] float32 array, or None on failure.
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
    except Exception as exc:
        log.debug("Landmark detection failed: %s", exc)
        return None


def _extract_eye_crop(image_path, landmarks: Optional[np.ndarray], size: int = 64) -> torch.Tensor:
    """Return (1, 3, size, size) normalized eye-region crop or zeros on failure."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if landmarks is None:
        return torch.zeros(1, 3, size, size)
    frame = cv2.imread(str(image_path))
    if frame is None:
        return torch.zeros(1, 3, size, size)
    H, W = frame.shape[:2]
    pts_px  = landmarks * np.array([W, H], dtype=np.float32)
    eye_pts = pts_px[36:48]
    ex0 = max(0, int(eye_pts[:, 0].min()) - 10)
    ey0 = max(0, int(eye_pts[:, 1].min()) - 10)
    ex1 = min(W, int(eye_pts[:, 0].max()) + 10)
    ey1 = min(H, int(eye_pts[:, 1].max()) + 10)
    crop = frame[ey0:ey1, ex0:ex1]
    if crop.size == 0:
        return torch.zeros(1, 3, size, size)
    crop_rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop_resized = cv2.resize(crop_rgb, (size, size))
    crop_norm    = ((crop_resized - mean) / std).transpose(2, 0, 1)
    return torch.from_numpy(crop_norm).float().unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class SentinelPipeline:
    """
    Full 7-module verification pipeline.

    Thread-safe: each module inference is protected by its own lock so multiple
    FastAPI async workers can call the pipeline concurrently on different inputs
    without sharing GPU state.

    Missing checkpoints: models load in random-init mode with a logged warning
    rather than crashing — the pipeline degrades gracefully.
    """

    def __init__(self, config: PipelineConfig, checkpoint_dir):
        self.cfg        = config
        self.ckpt_dir   = Path(checkpoint_dir)
        self.device     = config.resolve_device()
        self._calibrators = None

        # Per-module inference locks (prevents concurrent CUDA kernel stomping)
        self._lock_liveness   = threading.Lock()
        self._lock_deepfake   = threading.Lock()
        self._lock_face       = threading.Lock()
        self._lock_behavioral = threading.Lock()
        self._lock_document   = threading.Lock()
        self._lock_fusion     = threading.Lock()

        self._load_models()

        if config.warmup_iters > 0:
            self._warmup()

    # ── checkpoint loading ──────────────────────────────────────────────────

    def _load(self, model: torch.nn.Module, *filenames: str) -> torch.nn.Module:
        """Move model to device and load first matching checkpoint."""
        model = model.to(self.device)
        for filename in filenames:
            path = self.ckpt_dir / filename
            if path.exists():
                try:
                    # weights_only=True prevents arbitrary pickle execution
                    state = torch.load(path, map_location=self.device, weights_only=True)
                except TypeError:
                    # older torch without weights_only kwarg
                    state = torch.load(path, map_location=self.device)

                if isinstance(state, dict):
                    for key in ("au_gnn", "gaze_head", "model"):
                        if key in state:
                            state = state[key]
                            break
                try:
                    model.load_state_dict(state, strict=False)
                    log.info("Loaded %s from %s", type(model).__name__, filename)
                except RuntimeError as exc:
                    log.warning("Partial load of %s: %s", filename, exc)
                break
        else:
            log.warning("No checkpoint found for %s (tried: %s) — using random init.",
                        type(model).__name__, filenames)
        model.eval()
        if self.cfg.fp16 and self.device.type == "cuda":
            model = model.half()
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

        # Behavioral: joint checkpoint with sub-keys
        au_model   = ActionUnitGNN(n_landmarks=68, n_aus=self.cfg.n_aus).to(self.device)
        gaze_head  = GazeRegressionHead(in_channels=3, hidden_dim=128).to(self.device)
        loaded_beh = False
        for fname in ("behavioral_best.pt", "behavioral_latest.pt"):
            path = self.ckpt_dir / fname
            if path.exists():
                try:
                    try:
                        state = torch.load(path, map_location=self.device, weights_only=True)
                    except TypeError:
                        state = torch.load(path, map_location=self.device)
                    if isinstance(state, dict) and "au_gnn" in state:
                        try:
                            au_model.load_state_dict(state["au_gnn"], strict=False)
                        except RuntimeError as e:
                            log.warning("AU-GNN partial load: %s", e)
                        if "gaze_head" in state:
                            try:
                                gaze_head.load_state_dict(state["gaze_head"], strict=False)
                            except RuntimeError as e:
                                log.warning("GazeHead partial load: %s", e)
                        loaded_beh = True
                        log.info("Loaded behavioral checkpoint from %s", fname)
                except Exception as exc:
                    log.warning("Could not load behavioral checkpoint %s: %s", fname, exc)
                break
        if not loaded_beh:
            log.warning("No behavioral checkpoint found — using random init.")
        self.au_model   = au_model.eval()
        self.gaze_head  = gaze_head.eval()

        # Document
        self.doc_model = self._load(
            DocumentIntelligenceModel(image_size=self.cfg.image_size),
            "document_best.pt", "document_latest.pt",
        )

        # Fusion
        self.fusion_model = ScoreFusionModel(n_modules=5).to(self.device)
        fusion_path = self.ckpt_dir / "fusion_best.pt"
        if fusion_path.exists():
            try:
                try:
                    state = torch.load(fusion_path, map_location=self.device, weights_only=True)
                except TypeError:
                    state = torch.load(fusion_path, map_location=self.device)
                self.fusion_model.load_state_dict(state.get("model", state), strict=False)
                self._calibrators = state.get("calibrators", None)
                log.info("Loaded fusion model from fusion_best.pt")
            except Exception as exc:
                log.warning("Could not load fusion checkpoint: %s", exc)
        self.fusion_model.eval()

    # ── warm-up ────────────────────────────────────────────────────────────

    def _warmup(self):
        """Run dummy forward passes to JIT-compile kernels and prime CUDA caches."""
        log.info("Warming up models (%d iters)…", self.cfg.warmup_iters)
        dummy_std  = torch.zeros(1, 3, self.cfg.image_size, self.cfg.image_size, device=self.device)
        dummy_live = torch.zeros(1, 3, self.cfg.liveness_image_size, self.cfg.liveness_image_size, device=self.device)
        dummy_face = torch.zeros(1, 3, self.cfg.face_image_size, self.cfg.face_image_size, device=self.device)
        dummy_lm   = torch.zeros(1, 68, 2, device=self.device)
        dummy_L    = 128

        with torch.inference_mode():
            for _ in range(self.cfg.warmup_iters):
                try:
                    self.liveness_model(dummy_live)
                    self.deepfake_model(dummy_std)
                    self.face_model.embed(dummy_face)
                    self.au_model(dummy_lm)
                    token_ids = torch.zeros(1, dummy_L, dtype=torch.long, device=self.device)
                    bbox      = torch.zeros(1, dummy_L, 4, dtype=torch.long, device=self.device)
                    attn      = torch.zeros(1, dummy_L, dtype=torch.long, device=self.device)
                    self.doc_model(dummy_std, token_ids, bbox, attn)
                    score_vec = torch.ones(1, 5, device=self.device) * 0.5
                    self.fusion_model(score_vec)
                except Exception as exc:
                    log.debug("Warm-up step failed (non-fatal): %s", exc)
                    break

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        log.info("Warm-up complete.")

    # ── module runners ─────────────────────────────────────────────────────

    @torch.inference_mode()
    def run_liveness(self, selfie_path) -> tuple[float, Optional[np.ndarray]]:
        img = _load_image(selfie_path, _standard_transform(self.cfg.liveness_image_size), self.device)
        if self.cfg.fp16 and self.device.type == "cuda":
            img = img.half()
        t0 = time.perf_counter()
        with self._lock_liveness:
            out = self.liveness_model(img)
        elapsed = time.perf_counter() - t0
        if elapsed > self.cfg.module_timeout_s:
            log.warning("Liveness inference took %.2fs (threshold %.1fs)", elapsed, self.cfg.module_timeout_s)
        score   = float(torch.sigmoid(out["liveness_logit"]).item())
        depth   = out.get("depth_map")
        depth_np = depth.squeeze().cpu().float().numpy() if depth is not None else None
        return score, depth_np

    @torch.inference_mode()
    def run_deepfake(self, selfie_path) -> float:
        img = _load_image(selfie_path, _standard_transform(self.cfg.image_size), self.device)
        if self.cfg.fp16 and self.device.type == "cuda":
            img = img.half()
        with self._lock_deepfake:
            logit = self.deepfake_model(img)
        return float(torch.sigmoid(logit).item())

    @torch.inference_mode()
    def run_face_recognition(self, selfie_path) -> tuple[np.ndarray, Optional[float]]:
        img = _load_image(selfie_path, _face_transform(self.cfg.face_image_size), self.device)
        if self.cfg.fp16 and self.device.type == "cuda":
            img = img.half()
        with self._lock_face:
            embedding = self.face_model.embed(img).cpu().float().numpy()[0]
        dedup_score = None
        if self.face_index.index.ntotal > 0:
            sims, _ = self.face_index.search(embedding.reshape(1, -1), k=1)
            dedup_score = float(sims[0, 0])
        return embedding, dedup_score

    @torch.inference_mode()
    def run_behavioral(self, selfie_path) -> float:
        """
        Returns behavioral plausibility in [0, 1].
        Falls back to 0.5 (neutral) if MediaPipe is unavailable or face not detected.
        """
        try:
            landmarks = _detect_landmarks(selfie_path)
            if landmarks is None:
                return 0.5
            lm_t     = torch.from_numpy(landmarks).float().unsqueeze(0).to(self.device)
            eye_crop = _extract_eye_crop(selfie_path, landmarks).to(self.device)
            with self._lock_behavioral:
                au_pred, _ = self.au_model(lm_t)
                gaze_vec   = self.gaze_head(eye_crop)
            au_activity    = float((au_pred > 0.5).float().mean().item())
            # Gaze plausibility: unit vector is ideal (norm ≈ 1); penalise extremes
            gaze_plausibility = float(
                1.0 - abs(gaze_vec.norm(dim=1).clamp(0.0, 2.0).item() - 1.0)
            )
            return float(np.clip(0.6 * au_activity + 0.4 * gaze_plausibility, 0.0, 1.0))
        except Exception as exc:
            log.warning("Behavioral module failed (%s) — returning neutral 0.5", exc)
            return 0.5

    @torch.inference_mode()
    def run_document(self, document_path) -> tuple[float, dict]:
        img = _load_image(document_path, _standard_transform(self.cfg.image_size), self.device)
        if self.cfg.fp16 and self.device.type == "cuda":
            img = img.half()
        L         = 128
        token_ids = torch.zeros(1, L, dtype=torch.long, device=self.device)
        bbox      = torch.zeros(1, L, 4, dtype=torch.long, device=self.device)
        attn      = torch.zeros(1, L, dtype=torch.long, device=self.device)
        with self._lock_document:
            out = self.doc_model(img, token_ids, bbox, attn)
        forgery_prob = float(out["forgery_prob"].item())
        doc_type_idx = int(out["doc_type_logits"].argmax(dim=1).item())
        doc_type     = DocumentTypeClassifier.DOC_TYPES[doc_type_idx]
        return 1.0 - forgery_prob, {
            "document_type":     doc_type,
            "forgery_probability": round(forgery_prob, 4),
            "forgery_signals":   {k: round(float(v.item()), 4) for k, v in out["forgery_signals"].items()},
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
            return 0.5  # neutral — no enrolled reference to compare against
        except Exception as exc:
            log.debug("Voice module skipped: %s", exc)
            return None

    # ── high-level API ──────────────────────────────────────────────────────

    def verify(
        self,
        selfie_path,
        document_path=None,
        enrolled_embedding: Optional[np.ndarray] = None,
        audio_path=None,
    ) -> VerificationResult:
        """
        Run the full SentinelID pipeline.

        Args:
            selfie_path:        Path to a frontal face image (JPEG/PNG).
            document_path:      Optional path to an ID document image.
            enrolled_embedding: Optional 512-d float32 embedding for 1:1 face match.
            audio_path:         Optional path to a voice audio clip.

        Returns:
            VerificationResult with trust_score, decision, raw_scores, explanations.
        """
        t0   = time.time()
        raw: dict[str, float] = {}

        # M1 — Liveness
        liveness_score, _depth_map = self.run_liveness(selfie_path)
        raw["liveness"] = liveness_score

        # M2 — Deepfake
        deepfake_fake_prob = self.run_deepfake(selfie_path)
        raw["deepfake_fake_prob"] = deepfake_fake_prob

        # M3 — Face recognition / 1:1 match
        embedding, dedup_score = self.run_face_recognition(selfie_path)
        if enrolled_embedding is not None:
            e   = enrolled_embedding / (np.linalg.norm(enrolled_embedding) + 1e-8)
            emb = embedding          / (np.linalg.norm(embedding)          + 1e-8)
            # Cosine similarity → [0, 1]
            face_match = float((np.dot(emb, e) + 1.0) / 2.0)
        else:
            face_match = 0.5          # neutral when no enrolled reference
        raw["face_match"] = face_match
        if dedup_score is not None:
            raw["dedup_score"] = dedup_score

        # M4 — Behavioral
        behavioral = self.run_behavioral(selfie_path)
        raw["behavioral"] = behavioral

        # M5 — Document (optional)
        document_score = 0.5
        doc_meta: dict = {}
        if document_path is not None:
            try:
                document_score, doc_meta = self.run_document(document_path)
            except Exception as exc:
                log.warning("Document module failed (%s) — using neutral 0.5", exc)
        raw["document"] = document_score

        # M6 (optional) — Voice
        if audio_path is not None:
            voice = self.run_voice(audio_path)
            if voice is not None:
                raw["voice"] = voice

        # M6/M7 — Fusion
        score_vec = torch.tensor([
            liveness_score,
            1.0 - deepfake_fake_prob,
            face_match,
            behavioral,
            document_score,
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        if self._calibrators:
            try:
                cols = [
                    torch.tensor(c.transform(score_vec[:, i].cpu().numpy()), dtype=torch.float32)
                    for i, c in enumerate(self._calibrators)
                ]
                score_vec = torch.stack(cols, dim=1).to(self.device)
            except Exception as exc:
                log.warning("Calibration failed (%s) — using raw scores.", exc)

        with self._lock_fusion:
            trust_score, _ = self.fusion_model(score_vec)
        trust_score = float(trust_score.item())

        # Decision
        if trust_score >= self.cfg.trust_accept:
            decision = "ACCEPT"
        elif trust_score >= self.cfg.trust_review:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        explanations = {
            "latency_ms": round((time.time() - t0) * 1000, 1),
            **doc_meta,
        }
        result = VerificationResult(
            trust_score=trust_score,
            decision=decision,
            raw_scores=raw,
            explanations=explanations,
        )
        try:
            result.explanations["text"] = ScoreExplainer.explain(result)
        except Exception:
            pass
        return result

    def enroll(self, selfie_path) -> np.ndarray:
        """Extract and return a 512-d face embedding for enrollment storage."""
        embedding, _ = self.run_face_recognition(selfie_path)
        return embedding

    def add_to_dedup_index(self, embedding: np.ndarray):
        """Add an embedding to the in-memory deduplication FAISS index."""
        self.face_index.add(embedding.reshape(1, -1))

    # ── factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, checkpoint_dir, config: Optional[PipelineConfig] = None) -> "SentinelPipeline":
        return cls(config or PipelineConfig(), checkpoint_dir)

    # ── utilities ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SentinelPipeline("
            f"device={self.device}, "
            f"fp16={self.cfg.fp16}, "
            f"ckpt_dir={self.ckpt_dir})"
        )

    def memory_stats(self) -> dict:
        """Return current GPU memory usage (MB). Returns {} on CPU."""
        if self.device.type != "cuda":
            return {}
        return {
            "allocated_mb":  round(torch.cuda.memory_allocated(self.device) / 1e6, 1),
            "reserved_mb":   round(torch.cuda.memory_reserved(self.device)   / 1e6, 1),
        }

    def clear_gpu_cache(self):
        """Release cached but unused GPU memory."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
