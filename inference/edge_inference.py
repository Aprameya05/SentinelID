"""
inference/edge_inference.py — ONNX Runtime edge inference for SentinelID.

Runs the distilled MobileNetV3 student model via ONNX Runtime, which is what
actually runs on Android / iOS. Achieves < 200 ms on Snapdragon 778G (INT8).

Usage:
    from inference.edge_inference import EdgeInference
    engine = EdgeInference("checkpoints/edge_model.onnx")
    result = engine.run("selfie.jpg")
    print(result)

Benchmark:
    python inference/edge_inference.py --model checkpoints/edge_model.onnx --benchmark
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Output structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeResult:
    liveness_score: float        # P(live) in [0, 1]
    face_embedding: np.ndarray   # (256,) L2-normalised
    au_signal: float             # behavioral liveness in [0, 1]
    is_live: bool                # liveness_score >= threshold
    latency_ms: float            # end-to-end preprocessing + inference

    def __repr__(self) -> str:
        status = "✓ LIVE" if self.is_live else "✗ SPOOF"
        return (
            f"EdgeResult({status} | liveness={self.liveness_score:.3f} | "
            f"au={self.au_signal:.3f} | latency={self.latency_ms:.1f}ms)"
        )

    def to_dict(self) -> dict:
        return {
            "liveness_score": float(self.liveness_score),
            "face_embedding": self.face_embedding.tolist(),
            "au_signal": float(self.au_signal),
            "is_live": self.is_live,
            "latency_ms": float(self.latency_ms),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing — matches SentinelEdgeModel training normalization
# ──────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(image: Image.Image | np.ndarray | str | Path, size: int = 224) -> np.ndarray:
    """
    Load + normalize an image to (1, 3, H, W) float32 NCHW tensor.
    Accepts PIL Image, numpy HWC uint8, or a file path.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)


def _preprocess_batch(
    images: list[Image.Image | np.ndarray | str | Path],
    size: int = 224,
) -> np.ndarray:
    """Preprocess a list of images into (B, 3, H, W)."""
    return np.concatenate([_preprocess(img, size) for img in images], axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class EdgeInference:
    """
    ONNX Runtime inference engine for SentinelID edge model.

    Provider priority: CUDA → DirectML → CPU (auto-detected at init).
    Identical results to PyTorch forward pass (within float32 tolerance).
    """

    def __init__(
        self,
        onnx_path: str | Path,
        liveness_threshold: float = 0.5,
        image_size: int = 224,
        provider: str = "auto",
    ):
        import onnxruntime as ort

        self.path = str(onnx_path)
        self.threshold = liveness_threshold
        self.image_size = image_size

        # Provider selection
        available = ort.get_available_providers()
        if provider == "auto":
            for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]:
                if p in available:
                    provider = p
                    break
        self._provider = provider

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        self._sess = ort.InferenceSession(
            self.path,
            sess_options=sess_opts,
            providers=[provider],
        )
        self._input_name = self._sess.get_inputs()[0].name

        # Warm up
        dummy = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
        for _ in range(3):
            self._sess.run(None, {self._input_name: dummy})

        print(f"EdgeInference loaded: {Path(onnx_path).name} | provider={self._provider}")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, image: Image.Image | np.ndarray | str | Path) -> EdgeResult:
        """Run inference on a single image. Returns EdgeResult."""
        t0 = time.perf_counter()
        tensor = _preprocess(image, self.image_size)
        outs = self._sess.run(None, {self._input_name: tensor})
        latency = (time.perf_counter() - t0) * 1000

        liveness_score = float(outs[0][0])
        face_embed = outs[1][0]  # (256,)
        au_signal = float(outs[2][0])

        return EdgeResult(
            liveness_score=liveness_score,
            face_embedding=face_embed,
            au_signal=au_signal,
            is_live=liveness_score >= self.threshold,
            latency_ms=latency,
        )

    def run_batch(
        self,
        images: list[Image.Image | np.ndarray | str | Path],
        batch_size: int = 16,
    ) -> list[EdgeResult]:
        """Run inference on a list of images in mini-batches."""
        results = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            t0 = time.perf_counter()
            tensor = _preprocess_batch(batch_imgs, self.image_size)
            outs = self._sess.run(None, {self._input_name: tensor})
            latency = (time.perf_counter() - t0) * 1000 / len(batch_imgs)

            for i in range(len(batch_imgs)):
                results.append(EdgeResult(
                    liveness_score=float(outs[0][i]),
                    face_embedding=outs[1][i],
                    au_signal=float(outs[2][i]),
                    is_live=float(outs[0][i]) >= self.threshold,
                    latency_ms=latency,
                ))
        return results

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two face embeddings."""
        a = emb_a / (np.linalg.norm(emb_a) + 1e-8)
        b = emb_b / (np.linalg.norm(emb_b) + 1e-8)
        return float(np.dot(a, b))

    def verify_pair(
        self,
        image_a: Image.Image | np.ndarray | str | Path,
        image_b: Image.Image | np.ndarray | str | Path,
        face_threshold: float = 0.45,
    ) -> dict:
        """
        1:1 face verification.
        Returns liveness for image_a + cosine similarity + match decision.
        """
        r_a = self.run(image_a)
        r_b = self.run(image_b)
        sim = self.cosine_similarity(r_a.face_embedding, r_b.face_embedding)
        return {
            "liveness_a": r_a.liveness_score,
            "liveness_b": r_b.liveness_score,
            "face_similarity": sim,
            "face_match": sim >= face_threshold,
            "live_a": r_a.is_live,
            "live_b": r_b.is_live,
            "decision": "ACCEPT" if (r_a.is_live and sim >= face_threshold) else "REJECT",
            "latency_ms": r_a.latency_ms + r_b.latency_ms,
        }

    def benchmark(self, n_runs: int = 200, batch_sizes: tuple = (1, 4, 8)) -> dict:
        """
        Latency benchmark across batch sizes.
        Returns mean/p95/p99 per batch size.
        """
        results = {}
        for bs in batch_sizes:
            dummy = np.random.randn(bs, 3, self.image_size, self.image_size).astype(np.float32)
            times = []
            for _ in range(10):  # warmup
                self._sess.run(None, {self._input_name: dummy})
            for _ in range(n_runs):
                t0 = time.perf_counter()
                self._sess.run(None, {self._input_name: dummy})
                times.append((time.perf_counter() - t0) * 1000)
            t = np.array(times)
            results[f"batch_{bs}"] = {
                "mean_ms": float(t.mean()),
                "p95_ms":  float(np.percentile(t, 95)),
                "p99_ms":  float(np.percentile(t, 99)),
                "per_image_ms": float(t.mean() / bs),
            }
        return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SentinelID edge inference")
    parser.add_argument("--model", required=True, help="Path to edge_model.onnx")
    parser.add_argument("--image", default=None, help="Path to a face image to run")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    engine = EdgeInference(args.model, liveness_threshold=args.threshold)

    if args.image:
        result = engine.run(args.image)
        print(result)

    if args.benchmark:
        print("\nRunning latency benchmark (200 runs each)…")
        stats = engine.benchmark()
        print(f"\n{'Batch':<10} {'Mean':>10} {'P95':>10} {'P99':>10} {'Per-img':>12}")
        print("─" * 55)
        for label, s in stats.items():
            print(f"{label:<10} {s['mean_ms']:>9.1f}ms {s['p95_ms']:>9.1f}ms "
                  f"{s['p99_ms']:>9.1f}ms {s['per_image_ms']:>10.1f}ms")


if __name__ == "__main__":
    main()
