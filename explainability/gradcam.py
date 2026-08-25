"""
GradCAM explainability for the face recognition and deepfake modules.

Generates saliency maps showing which facial region drove each decision.
Used for both model debugging and as a user-facing explanation layer.

Reference: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
Localization" (Selvaraju et al., ICCV 2017)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import cv2
from typing import Callable


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into a target layer to capture activations and gradients.
    Works with any CNN that has a spatial feature map before global pooling.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def generate(
        self,
        image: Tensor,         # (1, 3, H, W)
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Generate a GradCAM heatmap for the given image.

        Returns a (H, W) heatmap in [0, 1].
        """
        self.model.eval()
        image = image.requires_grad_(True)

        output = self.model(image)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global average pool gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalise
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def overlay(
        self,
        image_np: np.ndarray,   # (H, W, 3) uint8 BGR
        cam: np.ndarray,        # (h, w) in [0, 1]
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay a GradCAM heatmap on the original image."""
        H, W = image_np.shape[:2]
        cam_resized = cv2.resize(cam, (W, H))
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
        return overlay


class ScoreExplainer:
    """
    Generates natural-language explanations for verification decisions.
    Combines GradCAM with module score analysis.
    """

    @staticmethod
    def explain(result) -> str:
        """Generate a plain-English explanation for a VerificationResult."""
        lines = []
        raw = result.raw_scores

        # Liveness
        live = raw.get("liveness", 0.5)
        if live > 0.85:
            lines.append("Strong liveness signal — genuine 3D face structure detected.")
        elif live > 0.6:
            lines.append("Moderate liveness confidence — lighting or angle may have reduced certainty.")
        else:
            lines.append("Low liveness score — possible spoof or low-quality image.")

        # Deepfake
        fake_p = raw.get("deepfake_fake_prob", 0.0)
        if fake_p < 0.1:
            lines.append("No deepfake artifacts detected across spatial and frequency domains.")
        elif fake_p < 0.4:
            lines.append("Minor frequency anomalies detected — possible compression artifact.")
        else:
            lines.append(f"High deepfake probability ({fake_p:.0%}) — synthetic face signals found.")

        # Face match
        face = raw.get("face_match", 0.5)
        if face > 0.8:
            lines.append("Strong face match against enrolled identity.")
        elif face > 0.45:
            lines.append("Moderate face match — possible pose or lighting variation.")
        else:
            lines.append("Face match failed — identity could not be confirmed.")

        # Behavioral
        beh = raw.get("behavioral", 0.5)
        if beh > 0.75:
            lines.append("Behavioral biometrics consistent with a live subject.")
        else:
            lines.append("Behavioral signals inconclusive — static image or limited landmarks.")

        return " ".join(lines)
