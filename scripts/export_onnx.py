"""
SentinelID — ONNX Export + Latency Benchmark
Exports the M7 edge-distilled MobileNetV3-Small model to ONNX and INT8,
then benchmarks inference latency on CPU.

Usage:
    python scripts/export_onnx.py --checkpoint models/distilled_best.pt --output models/m7_edge

Requirements:
    pip install torch torchvision onnx onnxruntime numpy
"""
import argparse
import time
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import onnx
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Model definition — must match train_distillation.py
# ---------------------------------------------------------------------------

def build_student(num_classes: int = 2) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(checkpoint: pathlib.Path, output_stem: pathlib.Path):
    print(f"Loading checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    model = build_student()

    # Handle both raw state dict and wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = output_stem.with_suffix(".onnx")

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["face_image"],
        output_names=["logits"],
        dynamic_axes={"face_image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"Exported ONNX model: {onnx_path}")

    # Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX graph validated.")

    size_kb = onnx_path.stat().st_size / 1024
    print(f"Model size: {size_kb:.1f} KB")

    return onnx_path


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(onnx_path: pathlib.Path, n_runs: int = 200, warmup: int = 20):
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_name = sess.get_inputs()[0].name

    print(f"\nBenchmarking {onnx_path.name} — {n_runs} runs (CPU)")

    # Warmup
    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    lats = sorted(latencies)
    print(f"  Mean:   {np.mean(lats):.2f} ms")
    print(f"  Median: {np.median(lats):.2f} ms")
    print(f"  P95:    {np.percentile(lats, 95):.2f} ms")
    print(f"  P99:    {np.percentile(lats, 99):.2f} ms")
    print(f"  Min:    {lats[0]:.2f} ms")
    print(f"  Max:    {lats[-1]:.2f} ms")
    return latencies


# ---------------------------------------------------------------------------
# INT8 quantization
# ---------------------------------------------------------------------------

def quantize_int8(onnx_path: pathlib.Path) -> pathlib.Path:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    q_path = onnx_path.parent / (onnx_path.stem + "_int8.onnx")
    quantize_dynamic(
        str(onnx_path),
        str(q_path),
        weight_type=QuantType.QUInt8,
    )
    print(f"\nINT8 quantized model: {q_path}")
    q_size = q_path.stat().st_size / 1024
    fp32_size = onnx_path.stat().st_size / 1024
    print(f"  FP32 size: {fp32_size:.1f} KB")
    print(f"  INT8 size: {q_size:.1f} KB  ({100*(1-q_size/fp32_size):.1f}% reduction)")
    return q_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        default=pathlib.Path("models/distilled_best.pt"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("models/m7_edge"))
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Generating dummy model for export demo...")
        model = build_student()
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)

    onnx_path = export_onnx(args.checkpoint, args.output)
    benchmark(onnx_path, n_runs=args.runs)

    if not args.skip_quantize:
        try:
            q_path = quantize_int8(onnx_path)
            print("\nINT8 benchmark:")
            benchmark(q_path, n_runs=args.runs)
        except ImportError:
            print("onnxruntime.quantization not available, skipping INT8.")


if __name__ == "__main__":
    main()
