"""
Document intelligence module for SentinelID.

Combines LayoutLMv3-style multimodal encoding with TrOCR text extraction
and a forgery detection head. Handles passports, driver's licenses, ID cards,
and other identity documents supported by MIDV-500.

Architecture:
    Image patches (ViT) + OCR token embeddings + 2D spatial position embeddings
    -> 12-layer transformer encoder
    -> Document type classifier
    -> Field extraction head (name, DOB, ID number, expiry)
    -> Forgery detection head (printer artifacts, splicing, font anomalies)
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
# Spatial 2D position encoding
# ──────────────────────────────────────────────────────────────────────────────

class SpatialPositionEmbedding(nn.Module):
    """
    Encodes bounding box coordinates as continuous position embeddings.
    Each token gets (x0, y0, x1, y1, w, h) -> 128-d via linear projection.
    Coordinates are normalized to [0, 1000] as in LayoutLM convention.
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.x_embed = nn.Embedding(1001, hidden_size // 6)
        self.y_embed = nn.Embedding(1001, hidden_size // 6)
        self.w_embed = nn.Embedding(1001, hidden_size // 6)
        self.h_embed = nn.Embedding(1001, hidden_size // 6)
        self.proj = nn.Linear((hidden_size // 6) * 4, hidden_size)

    def forward(self, bbox: Tensor) -> Tensor:
        """
        bbox: (B, L, 4) in normalized [0, 1000] coords (x0, y0, x1, y1)
        Returns: (B, L, hidden_size)
        """
        bbox = bbox.long().clamp(0, 1000)
        x0 = self.x_embed(bbox[:, :, 0])
        y0 = self.y_embed(bbox[:, :, 1])
        x1 = self.x_embed(bbox[:, :, 2])
        y1 = self.y_embed(bbox[:, :, 3])
        spatial = torch.cat([x0, y0, x1, y1], dim=-1)
        return self.proj(spatial)


# ──────────────────────────────────────────────────────────────────────────────
# Vision patch encoder (ViT-style image tokenizer)
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits document image into 16x16 patches and projects each to hidden_size.
    For a 224x224 input: 196 patches.
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16, hidden_size: int = 768):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, hidden_size) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, 3, H, W) -> (B, N+1, hidden_size)"""
        B = x.shape[0]
        x = self.proj(x)                         # (B, C, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)         # (B, N, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B, N+1, C)
        x = x + self.pos_embed
        return self.norm(x)


# ──────────────────────────────────────────────────────────────────────────────
# Multimodal transformer encoder
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalTransformerLayer(nn.Module):
    """Single transformer layer with pre-norm and standard MHA."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 12, ff_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + x
        x = x + self.ff(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Output heads
# ──────────────────────────────────────────────────────────────────────────────

class DocumentTypeClassifier(nn.Module):
    """Predicts document type: passport, license, national ID, etc."""

    DOC_TYPES = [
        "passport", "drivers_license", "national_id", "residence_permit",
        "voter_id", "pan_card", "aadhaar", "other",
    ]

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(self.DOC_TYPES)),
        )

    def forward(self, cls_token: Tensor) -> Tensor:
        """cls_token: (B, hidden_size) -> (B, n_doc_types)"""
        return self.head(cls_token)


@dataclass
class ExtractedFields:
    full_name: str = ""
    date_of_birth: str = ""
    document_number: str = ""
    expiry_date: str = ""
    nationality: str = ""
    mrz_line1: str = ""
    mrz_line2: str = ""
    confidence: dict = field(default_factory=dict)


class FieldExtractionHead(nn.Module):
    """
    Token-level classification for field boundaries (BIO tagging).
    Separate classifier for each field type.
    """

    FIELDS = ["name", "dob", "doc_number", "expiry", "nationality", "mrz"]

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        # BIO: B-field, I-field, O -> 3 labels per field
        self.heads = nn.ModuleDict({
            f: nn.Linear(hidden_size, 3) for f in self.FIELDS
        })

    def forward(self, sequence_output: Tensor) -> dict[str, Tensor]:
        """
        sequence_output: (B, L, hidden_size)
        Returns dict: field -> (B, L, 3) logits
        """
        return {f: head(sequence_output) for f, head in self.heads.items()}


class ForgeryDetectionHead(nn.Module):
    """
    Detects document forgery signals:
        - Printer dot pattern anomalies (inkjet vs laser vs genuine press)
        - Copy-move splicing artifacts
        - Font inconsistencies
        - Tampered MRZ check digits

    Output: probability of forgery + per-signal scores.
    """

    FORGERY_SIGNALS = [
        "printer_artifact",
        "splicing",
        "font_anomaly",
        "mrz_tamper",
        "metadata_mismatch",
    ]

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.signal_heads = nn.ModuleDict({
            s: nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
            for s in self.FORGERY_SIGNALS
        })
        self.overall = nn.Sequential(
            nn.Linear(hidden_size + len(self.FORGERY_SIGNALS), 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, cls_token: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        cls_token: (B, hidden_size)
        Returns: (forgery_prob: (B,), signal_scores: {name: (B,)})
        """
        signal_logits = {s: head(cls_token).squeeze(-1) for s, head in self.signal_heads.items()}
        signal_stack = torch.stack(list(signal_logits.values()), dim=1)  # (B, n_signals)
        combined = torch.cat([cls_token, signal_stack], dim=1)
        forgery_prob = torch.sigmoid(self.overall(combined).squeeze(-1))
        signal_probs = {s: torch.sigmoid(v) for s, v in signal_logits.items()}
        return forgery_prob, signal_probs


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class DocumentIntelligenceModel(nn.Module):
    """
    Full LayoutLMv3-style document intelligence model.

    Inputs:
        image: (B, 3, 224, 224) document image
        token_ids: (B, L) tokenizer output (WordPiece or BPE)
        bbox: (B, L, 4) normalized bounding boxes per token
        attention_mask: (B, L) optional padding mask

    The model concatenates image patch tokens with OCR text tokens,
    adds spatial position embeddings, and passes through 12 transformer
    layers to produce CLS output and per-token representations.
    """

    def __init__(
        self,
        vocab_size: int = 30522,      # BERT WordPiece vocab
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        image_size: int = 224,
        patch_size: int = 16,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Text encoder
        self.token_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.text_pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.spatial_pos_embed = SpatialPositionEmbedding(hidden_size)

        # Image encoder
        self.patch_embed = PatchEmbedding(image_size, patch_size, hidden_size)

        # Modality type embeddings (0=text, 1=image)
        self.modality_embed = nn.Embedding(2, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            MultimodalTransformerLayer(hidden_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output heads
        self.doc_type_head = DocumentTypeClassifier(hidden_size)
        self.field_head = FieldExtractionHead(hidden_size)
        self.forgery_head = ForgeryDetectionHead(hidden_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        image: Tensor,                               # (B, 3, H, W)
        token_ids: Tensor,                           # (B, L_text)
        bbox: Tensor,                                # (B, L_text, 4)
        attention_mask: Optional[Tensor] = None,    # (B, L_text)
    ) -> dict[str, Tensor | dict]:
        B, L_text = token_ids.shape

        # --- Text branch ---
        positions = torch.arange(L_text, device=token_ids.device).unsqueeze(0)
        text_emb = (
            self.token_embed(token_ids)
            + self.text_pos_embed(positions)
            + self.spatial_pos_embed(bbox)
            + self.modality_embed(torch.zeros(B, L_text, dtype=torch.long, device=token_ids.device))
        )

        # --- Image branch ---
        img_tokens = self.patch_embed(image)           # (B, N_patch+1, C)
        N_img = img_tokens.shape[1]
        img_tokens = img_tokens + self.modality_embed(
            torch.ones(B, N_img, dtype=torch.long, device=image.device)
        )

        # --- Concatenate ---
        x = torch.cat([img_tokens, text_emb], dim=1)  # (B, N_img + L_text, C)
        x = self.norm(self.dropout(x))

        # Attention mask: MHA expects (N_total, N_total) or None.
        # Simplify to None (all tokens attend to each other).
        attn_mask = None

        # --- Transformer ---
        for layer in self.layers:
            x = layer(x, attn_mask)

        cls_token = x[:, 0]                   # image CLS
        text_output = x[:, N_img:]            # (B, L_text, C)

        # --- Heads ---
        doc_type_logits = self.doc_type_head(cls_token)
        field_logits = self.field_head(text_output)
        forgery_prob, forgery_signals = self.forgery_head(cls_token)

        return {
            "doc_type_logits": doc_type_logits,       # (B, 8)
            "field_logits": field_logits,              # {field: (B, L, 3)}
            "forgery_prob": forgery_prob,              # (B,)
            "forgery_signals": forgery_signals,        # {signal: (B,)}
            "cls_embedding": cls_token,                # (B, 768) for downstream fusion
        }

    @torch.inference_mode()
    def predict(
        self,
        image: Tensor,
        token_ids: Tensor,
        bbox: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> dict:
        """Returns human-readable prediction dict."""
        out = self.forward(image, token_ids, bbox, attention_mask)
        doc_type_idx = out["doc_type_logits"].argmax(dim=1).item()
        doc_type = DocumentTypeClassifier.DOC_TYPES[doc_type_idx]
        forgery_score = float(out["forgery_prob"][0])

        return {
            "document_type": doc_type,
            "forgery_score": forgery_score,
            "is_forged": forgery_score > 0.5,
            "forgery_signals": {
                k: float(v[0]) for k, v in out["forgery_signals"].items()
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class DocumentLoss(nn.Module):
    """
    Combined loss for document intelligence training.

    doc_type_loss: cross-entropy
    field_loss: BIO token-level cross-entropy (summed over fields)
    forgery_loss: focal loss
    """

    def __init__(
        self,
        field_weight: float = 1.0,
        doc_type_weight: float = 0.5,
        forgery_weight: float = 1.5,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.field_w = field_weight
        self.doc_type_w = doc_type_weight
        self.forgery_w = forgery_weight
        self.gamma = focal_gamma

    def focal_bce(self, pred: Tensor, target: Tensor) -> Tensor:
        bce = F.binary_cross_entropy(pred.float(), target.float(), reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        return ((1 - p_t) ** self.gamma * bce).mean()

    def forward(
        self,
        outputs: dict,
        doc_type_labels: Tensor,       # (B,) int
        field_labels: dict[str, Tensor],  # {field: (B, L)} BIO int
        forgery_labels: Tensor,        # (B,) float in {0, 1}
    ) -> Tensor:
        # Document type loss
        doc_loss = F.cross_entropy(outputs["doc_type_logits"], doc_type_labels)

        # Field extraction loss
        field_loss = torch.tensor(0.0, device=doc_type_labels.device)
        for f, logits in outputs["field_logits"].items():
            if f in field_labels:
                B, L, _ = logits.shape
                field_loss = field_loss + F.cross_entropy(
                    logits.view(B * L, 3),
                    field_labels[f].view(B * L),
                    ignore_index=-1,
                )

        # Forgery loss (focal)
        forgery_loss = self.focal_bce(outputs["forgery_prob"], forgery_labels.float())

        total = (
            self.doc_type_w * doc_loss
            + self.field_w * field_loss
            + self.forgery_w * forgery_loss
        )
        return total
