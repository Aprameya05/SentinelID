"""
Behavioral biometrics: Facial Action Unit detection via Graph Neural Network.

Facial Action Units (FACS) describe the muscular basis of facial expressions.
Real faces exhibit involuntary micro-movements correlated with gaze and breathing.
A static photo or pre-recorded replay cannot reproduce these stochastic patterns.

Architecture:
  - 68-point landmark detector (dlib or mediapipe)
  - GNN where nodes = landmarks, edges = anatomical adjacency
  - AU intensity regression head per AU
  - Gaze direction regression from eye-region crops
  - Liveness signal: temporal variance of AU activations

Reference: "Graph-based AU Detection" (Li et al., ICCV 2019)
           "Learning Dynamic Graph Representation for Facial AU" (Luo et al., AAAI 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple

# ──────────────────────────────────────────────────────────────────────────────
# Anatomical landmark adjacency (68-point iBUG schema)
# ──────────────────────────────────────────────────────────────────────────────

# Groups of landmarks: jaw (0-16), eyebrows (17-26), nose (27-35),
# eyes (36-47), mouth (48-67)
LANDMARK_EDGES: list[tuple[int, int]] = (
    # Jaw contour
    [(i, i+1) for i in range(16)] +
    # Left eyebrow
    [(17, 18), (18, 19), (19, 20), (20, 21)] +
    # Right eyebrow
    [(22, 23), (23, 24), (24, 25), (25, 26)] +
    # Nose bridge
    [(27, 28), (28, 29), (29, 30)] +
    # Nose bottom
    [(31, 32), (32, 33), (33, 34), (34, 35)] +
    # Left eye
    [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)] +
    # Right eye
    [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)] +
    # Outer mouth
    [(48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
     (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48)] +
    # Inner mouth
    [(60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)] +
    # Cross-region connections (muscle groups)
    [(17, 36), (26, 45), (0, 17), (16, 26)]
)

# AU indices present in DISFA: 1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43
DISFA_AUS = [1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43]
AU_IDX_MAP = {au: i for i, au in enumerate(DISFA_AUS)}


def build_adjacency(num_nodes: int = 68) -> tuple[Tensor, Tensor]:
    """Build edge_index and edge_attr tensors from LANDMARK_EDGES."""
    edges = LANDMARK_EDGES + [(b, a) for a, b in LANDMARK_EDGES]  # undirected
    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index


# ──────────────────────────────────────────────────────────────────────────────
# Graph convolution (simple message passing)
# ──────────────────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """
    Single graph convolution: aggregates neighbour features and applies
    a linear transformation. Uses degree-normalised adjacency.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.self_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        x: (B, N, in_dim) node features
        edge_index: (2, E) edges
        """
        B, N, _ = x.shape
        src, dst = edge_index[0], edge_index[1]

        # Aggregate: mean of neighbour features
        msg = x[:, src, :]                            # (B, E, in_dim)
        agg = torch.zeros(B, N, x.shape[-1], device=x.device)
        agg.scatter_add_(1, dst.view(1, -1, 1).expand(B, -1, x.shape[-1]), msg)

        # Count neighbours for normalisation
        count = torch.zeros(N, device=x.device)
        count.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
        count = count.clamp(min=1).view(1, N, 1)

        agg = agg / count

        out = self.linear(agg) + self.self_linear(x)
        return F.gelu(self.norm(out))


class ActionUnitGNN(nn.Module):
    """
    Graph neural network for facial action unit detection.

    Input:
      landmarks: (B, 68, 2) normalized (x,y) landmark coordinates
      image_features: (B, 68, C) per-landmark visual features (optional patch embeds)

    Output:
      au_intensities: (B, num_aus) intensity in [0, 5] per AU
      liveness_signal: (B,) behavioral liveness score
      gaze: (B, 3) gaze direction unit vector (optional)
    """

    # FACS AUs we predict (DISFA subset)
    AUS = DISFA_AUS

    def __init__(
        self,
        in_dim: int = 2,            # (x, y) coords; set to 2+C if using visual features
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_aus: int = len(DISFA_AUS),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_aus = num_aus

        # Register fixed adjacency
        self.register_buffer("edge_index", build_adjacency(68))

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Global graph readout (attention pooling)
        self.attn_pool = nn.Linear(hidden_dim, 1)

        # AU regression head
        self.au_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_aus),
            nn.ReLU(),  # AU intensities >= 0
        )

        # Liveness signal head (behavioral consistency score)
        self.liveness_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, landmarks: Tensor) -> dict[str, Tensor]:
        """
        landmarks: (B, 68, 2) in normalised image coordinates
        """
        B = landmarks.shape[0]

        # Encode nodes
        x = self.node_encoder(landmarks)    # (B, 68, hidden_dim)

        # Graph convolutions
        edge_index = self.edge_index
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = self.dropout(x)

        # Attention pooling for graph-level representation
        attn_weights = F.softmax(self.attn_pool(x), dim=1)  # (B, 68, 1)
        graph_repr = (attn_weights * x).sum(dim=1)           # (B, hidden_dim)

        # AU intensities (0-5 scale as in DISFA)
        au_intensities = self.au_head(graph_repr) * 5.0

        # Behavioral liveness signal
        liveness_signal = self.liveness_head(graph_repr).squeeze(1)

        return {
            "au_intensities": au_intensities,    # (B, 12)
            "liveness_signal": liveness_signal,  # (B,)
            "graph_repr": graph_repr,            # (B, hidden_dim) for fusion
        }


class GazeRegressionHead(nn.Module):
    """
    Predict 3D gaze direction from eye-region crop.
    Input: (B, 3, 64, 64) eye crop (left or right eye)
    Output: (B, 3) unit gaze vector
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, eye_crop: Tensor) -> Tensor:
        feat = self.backbone(eye_crop)
        gaze = self.head(feat)
        return F.normalize(gaze, dim=1)  # unit vector


class AULoss(nn.Module):
    """Smooth L1 for AU intensity regression with per-AU weighting."""

    def __init__(self, au_weights: Tensor | None = None):
        super().__init__()
        self.register_buffer("weights", au_weights if au_weights is not None
                             else torch.ones(len(DISFA_AUS)))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        per_au = F.smooth_l1_loss(pred, target, reduction="none")  # (B, num_aus)
        weighted = (per_au * self.weights).mean()
        return weighted
