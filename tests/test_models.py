"""
Smoke tests for all 7 model modules.
Runs forward passes with random input on CPU — no GPU or dataset required.
Tests catch import errors, shape mismatches, and obvious logic bugs.
"""

import pytest
import torch
import numpy as np


DEVICE = torch.device("cpu")


class TestArcFace:
    def test_forward_training(self):
        from models.face_recognition.arcface import ArcFaceModel
        model = ArcFaceModel(num_classes=100, embedding_dim=512, backbone="iresnet50").to(DEVICE)
        imgs = torch.randn(4, 3, 112, 112)
        labels = torch.randint(0, 100, (4,))
        loss = model(imgs, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_embed(self):
        from models.face_recognition.arcface import ArcFaceModel
        model = ArcFaceModel(num_classes=100, embedding_dim=512).to(DEVICE).eval()
        imgs = torch.randn(2, 3, 112, 112)
        emb = model.embed(imgs)
        assert emb.shape == (2, 512)
        # Embeddings should be L2-normalized
        norms = emb.norm(dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_faiss_index(self):
        from models.face_recognition.arcface import FaceDeduplicationIndex
        idx = FaceDeduplicationIndex(dim=128)
        vecs = np.random.randn(10, 128).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        idx.add(vecs)
        assert idx.index.ntotal == 10
        sims, ids = idx.search(vecs[:2], k=3)
        assert sims.shape == (2, 3)


class TestDepthLiveness:
    def test_forward(self):
        from models.liveness.depth_liveness import DepthLivenessModel
        model = DepthLivenessModel(backbone="resnet50").to(DEVICE).eval()
        imgs = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model(imgs)
        assert "liveness_logit" in out
        assert "depth_map" in out
        assert out["liveness_logit"].shape == (2,)
        assert out["depth_map"].shape[0] == 2

    def test_berhu_loss(self):
        from models.liveness.depth_liveness import BerHuLoss
        loss_fn = BerHuLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)
        mask = torch.ones_like(target)
        loss = loss_fn(pred, target, mask)
        assert loss.item() >= 0

    def test_liveness_loss(self):
        from models.liveness.depth_liveness import LivenessLoss
        criterion = LivenessLoss()
        logit = torch.randn(4)
        depth_pred = torch.randn(4, 1, 64, 64)
        depth_gt = torch.randn(4, 1, 64, 64)
        labels = torch.randint(0, 2, (4,)).float()
        has_depth = torch.ones(4)
        loss = criterion(logit, depth_pred, depth_gt, labels, has_depth)
        assert loss.item() > 0


class TestDeepfakeDetector:
    def test_forward(self):
        from models.deepfake.cnn_transformer import DeepfakeDetector
        model = DeepfakeDetector(pretrained=False).to(DEVICE).eval()
        imgs = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(imgs)
        assert logits.shape == (2,)

    def test_focal_loss(self):
        from models.deepfake.cnn_transformer import DeepfakeLoss
        criterion = DeepfakeLoss()
        logits = torch.randn(8)
        labels = torch.randint(0, 2, (8,)).float()
        loss = criterion(logits, labels)
        assert loss.item() >= 0

    def test_frequency_branch(self):
        from models.deepfake.cnn_transformer import FrequencyBranch
        branch = FrequencyBranch(out_dim=256).to(DEVICE).eval()
        imgs = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = branch(imgs)
        assert out.shape == (2, 256)


class TestAUGNN:
    def test_forward(self):
        from models.behavioral.au_gnn import ActionUnitGNN
        model = ActionUnitGNN(n_landmarks=68, n_aus=12).to(DEVICE).eval()
        landmarks = torch.randn(3, 68, 2)
        with torch.no_grad():
            au_pred, attn = model(landmarks)
        assert au_pred.shape == (3, 12)
        assert au_pred.min() >= 0  # outputs are after sigmoid/relu

    def test_gaze_head(self):
        from models.behavioral.au_gnn import GazeRegressionHead
        head = GazeRegressionHead(in_channels=3, hidden_dim=64).to(DEVICE).eval()
        eye_crops = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            gaze = head(eye_crops)
        assert gaze.shape == (2, 3)
        # Output should be unit vector
        norms = gaze.norm(dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_au_loss(self):
        from models.behavioral.au_gnn import AULoss
        weights = torch.ones(12)
        criterion = AULoss(au_weights=weights)
        pred = torch.rand(4, 12) * 5
        target = torch.rand(4, 12) * 5
        loss = criterion(pred, target)
        assert loss.item() >= 0


class TestDocumentIntelligence:
    def test_forward(self):
        from models.document.layout_intelligence import DocumentIntelligenceModel
        model = DocumentIntelligenceModel(
            vocab_size=100, hidden_size=64, num_layers=2, num_heads=4, ff_dim=128,
            image_size=32, patch_size=8, max_seq_len=16,
        ).to(DEVICE).eval()
        image = torch.randn(2, 3, 32, 32)
        token_ids = torch.randint(0, 100, (2, 16))
        bbox = torch.randint(0, 1000, (2, 16, 4))
        attn_mask = torch.ones(2, 16, dtype=torch.long)
        with torch.no_grad():
            out = model(image, token_ids, bbox, attn_mask)
        assert "doc_type_logits" in out
        assert "forgery_prob" in out
        assert out["doc_type_logits"].shape == (2, 8)
        assert out["forgery_prob"].shape == (2,)
        assert out["forgery_prob"].min() >= 0
        assert out["forgery_prob"].max() <= 1

    def test_document_loss(self):
        from models.document.layout_intelligence import DocumentIntelligenceModel, DocumentLoss
        model = DocumentIntelligenceModel(
            vocab_size=100, hidden_size=64, num_layers=2, num_heads=4, ff_dim=128,
            image_size=32, patch_size=8, max_seq_len=16,
        ).to(DEVICE)
        criterion = DocumentLoss()
        image = torch.randn(2, 3, 32, 32)
        token_ids = torch.zeros(2, 16, dtype=torch.long)
        bbox = torch.zeros(2, 16, 4, dtype=torch.long)
        attn = torch.ones(2, 16, dtype=torch.long)
        out = model(image, token_ids, bbox, attn)
        loss = criterion(
            out,
            doc_type_labels=torch.randint(0, 8, (2,)),
            field_labels={},
            forgery_labels=torch.randint(0, 2, (2,)).float(),
        )
        assert loss.item() > 0


class TestScoreFusion:
    def test_forward(self):
        from models.fusion.score_fusion import ScoreFusionModel
        model = ScoreFusionModel(n_modules=5).to(DEVICE).eval()
        score_vec = torch.rand(4, 5)
        with torch.no_grad():
            trust_score, logit = model(score_vec)
        assert trust_score.shape == (4,)
        assert trust_score.min() >= 0
        assert trust_score.max() <= 1

    def test_platt_calibrator(self):
        from models.fusion.score_fusion import PlattCalibrator
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, 200)
        labels = (scores > 0.5).astype(int)
        cal = PlattCalibrator()
        cal.fit(scores, labels)
        out = cal.transform(scores)
        assert out.shape == scores.shape
        assert out.min() >= 0
        assert out.max() <= 1

    def test_fusion_loss(self):
        from models.fusion.score_fusion import FusionLoss
        criterion = FusionLoss(confidence_penalty=0.1)
        logit = torch.randn(8)
        trust = torch.sigmoid(logit)
        labels = torch.randint(0, 2, (8,)).float()
        loss = criterion(logit, labels, trust)
        assert loss.item() >= 0


class TestEdgeDistillation:
    def test_forward(self):
        from models.edge.distillation import SentinelEdgeModel
        model = SentinelEdgeModel(liveness_out=1, face_embed_dim=64, au_out=12).to(DEVICE).eval()
        imgs = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(imgs)
        assert "liveness" in out
        assert "face_embed" in out
        assert "au_signal" in out
        assert out["liveness"].shape == (2,)
        assert out["face_embed"].shape == (2, 64)

    def test_distillation_loss(self):
        from models.edge.distillation import DistillationLoss
        criterion = DistillationLoss(temperature=4.0, alpha=0.7)
        student_liveness = torch.randn(4)
        student_embed = torch.randn(4, 64)
        teacher_liveness = torch.randn(4)
        teacher_embed = torch.randn(4, 64)
        hard_labels = torch.randint(0, 2, (4,)).float()
        loss = criterion(student_liveness, student_embed, teacher_liveness, teacher_embed, hard_labels)
        assert loss.item() >= 0
