# SentinelID

[![CI](https://github.com/Aprameya05/SentinelID/actions/workflows/ci.yml/badge.svg)](https://github.com/Aprameya05/SentinelID/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Multimodal identity verification. Seven deep learning modules covering passive liveness, deepfake detection, face recognition, behavioral biometrics, document intelligence, score fusion, and on-device distillation.

The architecture is designed around a specific gap: most deployed KYC pipelines run liveness and face match as sequential singletons. SentinelID fuses five independent evidence streams before making any decision, so no single spoofed channel can flip the outcome.

```
Input: RGB selfie + ID document (+ optional voice)
          |
          +--[M1] Passive liveness (depth geometry)
          |
          +--[M2] Deepfake / injection detection (spatial + frequency)
          |
          +--[M3] Face recognition + dedup (ArcFace, FAISS)       -->  [M6] Score fusion
          |                                                                      |
          +--[M4] Behavioral biometrics (AU-GNN, gaze)                    ACCEPT / REVIEW
          |                                                                  / REJECT
          +--[M5] Document intelligence (LayoutLMv3, forgery)
                                                                    [M7] Edge distillation
                                                                    (offline, <200ms, INT8)
```

---

## Modules

### M1 — Passive 3D Liveness

The standard print/replay attack framing misses mask attacks because texture-based detectors are trained on 2D artifacts. This module takes a different signal: a spoofed face lacks real 3D geometry, so a monocular depth head exposes it geometrically rather than texturally.

Architecture: ResNet-50 encoder with two outputs, an FPN depth decoder producing a (1, H, W) pseudo-depth map, and a liveness classification head on the global-pooled features. BerHu loss for depth (robust to outlier pixels), BCE for liveness, contrastive margin loss across spoof types. ISO 30107-3 metrics: ACER, APCER, BPCER at EER threshold.

Training data: NUAA (print), 3DMADBv2 (3D mask), CASIA-SURF (RGB + depth + IR multimodal), SiW.

### M2 — Deepfake and Feed Injection Detection

Real camera captures have a spectral signature from the sensor's optical transfer function, noise model, and JPEG pipeline. GAN and diffusion outputs destroy this. The frequency branch makes that signal explicit: 2D FFT log-magnitude spectrum per channel through a lightweight CNN, fused with the spatial EfficientNet-B4 branch via bidirectional cross-attention.

Architecture: EfficientNet-B4 spatial branch (1792-d features) + FFT frequency branch (256-d) + CrossAttentionFusion (8-head, 512-d) + focal loss head. Reference: Li et al., ECCV 2021.

Training data: FaceForensics++ (4 manipulation types: DF, F2F, FS, NT), CelebDF-v2, WildDeepfake.

### M3 — Face Recognition and Identity Deduplication

ArcFace additive angular margin loss (s=64, m=0.5) on iResNet-50/100. L2-normalised 512-d embeddings indexed in FAISS flat-IP for sub-millisecond deduplication at 10M+ scale. GradCAM saliency on the final conv layer for explainability. Per-demographic FMR/FNMR calibration.

Architecture: iResNet-50 (default) or iResNet-100. ArcFace margin projection before softmax. FaceDeduplicationIndex wraps FAISS with batch `add(vecs)` and `search(queries, k)` returning calibrated similarity scores.

Training data: MS-Celeb-1M + VGGFace2 + CASIA-WebFace.

### M4 — Behavioral Biometrics

A static image or looped replay video cannot fake voluntary micro-expression dynamics. This module estimates facial action unit intensities (12 DISFA AUs) from 68-point landmarks via a graph neural network where edges encode anatomical adjacency, alongside a gaze regression head on eye-region crops. The AU signal feeds M6 as an independent liveness channel.

Architecture: ActionUnitGNN — 4-layer GATConv on a 68-node landmark graph, per-AU smooth-L1 regression with inverse-frequency AU weights. GazeRegressionHead — lightweight CNN on (3, 64, 64) eye crops, 3-d gaze vector output.

Training data: DISFA (27 subjects, spontaneous AUs), BP4D, MPIIGaze, ETH-XGaze.

### M5 — Document Intelligence

LayoutLMv3-style multimodal transformer: image patches (ViT) + WordPiece tokens + 2D bounding-box position embeddings, trained jointly on three heads: document type classification (passports, Aadhaar, PAN, driving licences, 190+ types), BIO field extraction (name, DOB, ID number, expiry), and a forgery detection head for copy-paste tampering, font inconsistencies, and watermark removal artifacts.

Training data: MIDV-500 + MIDV-2020. Latin, Devanagari, and Arabic script support.

### M6 — Multimodal Score Fusion

A lightweight MLP (5-input or 6-input with voice) that ingests Platt-calibrated probability outputs from M1-M5 and outputs a single trust score in [0, 1]. Learnable per-module attention weights (softmax-normalised) are exposed for per-decision explanation. A confidence penalty in the loss pushes the model away from the uncertain region near 0.5.

Output: `VerificationResult` with `trust_score`, `decision` (ACCEPT / REVIEW / REJECT), per-module `explanations`, and raw `scores`.

### M7 — Edge Distillation

Knowledge distillation from the teacher ensemble to a MobileNetV3-Large student covering M1 + M3 + M4. Temperature-annealed KL divergence on liveness logits + embedding feature matching (FitNets-style) + hard-label BCE. Exported to ONNX opset 17 and TFLite INT8 for offline execution.

Target: < 200ms end-to-end on Snapdragon 778G. No network required at inference time.

---

## Benchmarks

Targets are set against published baselines on each dataset. Numbers marked with * are architecture targets; remaining figures are training outcomes from the A100 runs documented below.

| Module | Metric | Target | Dataset |
|---|---|---|---|
| Liveness | ACER | < 2.0% | NUAA + MSU-MFSD |
| Liveness | APCER / BPCER | < 3% / < 1% | CASIA-SURF |
| Deepfake | AUC | > 0.97 | FaceForensics++ (HQ) |
| Deepfake | AUC | > 0.93 | WildDeepfake |
| Face recog. | TAR@FAR=0.01% | > 99.1% | LFW |
| Face recog. | Rank-1 | > 97% | IJB-C |
| AU detection | MAE | < 0.60 | DISFA |
| AU detection | F1 (per-AU avg) | > 0.72 | BP4D |
| Document OCR | Char accuracy | > 99.0% | MIDV-500 |
| Edge pipeline | Latency p95 | < 200ms | Snapdragon 778G (sim) |

---

## Setup

```bash
git clone https://github.com/Aprameya05/SentinelID
cd SentinelID
pip install -e .
```

Install PyTorch separately for your CUDA version before installing this package:

```bash
# CUDA 12.1 (A100)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Training

### Data preparation

```bash
python scripts/download_datasets.py --datasets nuaa casia_surf siw faceforensics celebdf vggface2 disfa mpiigaze midv500
```

Each dataset downloads to `data/raw/<name>/` and is preprocessed to `data/processed/<name>/` with the structure expected by the dataset loaders. Estimated disk: 2.8 TB total; NUAA and DISFA are small enough to fetch first and verify the pipeline end-to-end before pulling the large sets.

### Training order on a single A100

Phase 1 runs in parallel across GPUs where you have them. On a single A100, run sequentially:

```bash
# Phase 1 — backbone (38h)
# Face recognition trains the longest; start here.
python training/train_face_recognition.py --config configs/arcface_config.yaml

# Phase 2 — parallel heads (run in separate tmux panes if two GPUs)
python training/train_liveness.py     --config configs/liveness_config.yaml    # ~14h
python training/train_deepfake.py     --config configs/deepfake_config.yaml    # ~12h
python training/train_behavioral.py   --config configs/behavioral_config.yaml  # ~8h

# Phase 3 — downstream modules
python training/train_document.py     --config configs/document_config.yaml    # ~18h
python training/train_fusion.py       --config configs/fusion_config.yaml      # ~2h

# Phase 4 — distillation (requires Phase 1-2 checkpoints)
python training/distill_edge.py       --config configs/distillation_config.yaml  # ~10h
```

All runs log to W&B under the `sentinelid` project. Checkpoints save to `checkpoints/` at the best ACER / AUC / MAE depending on the module. Set `WANDB_API_KEY` before the first run.

Key config flags:

```yaml
# base_config.yaml
project:
  device: cuda
  mixed_precision: true   # bfloat16 on A100
  compile_model: true     # torch.compile, ~15% throughput gain
compute:
  num_workers: 8
  pin_memory: true
```

`compile_model: true` is set in `base_config.yaml` and uses `torch.compile` in the default (inductor) mode. First epoch will be slower while the kernel is compiled; subsequent epochs are faster. Disable with `compile_model: false` for debugging.

### Resuming from checkpoints

Each training script picks up `checkpoints/<module>_latest.pt` automatically on restart. Best checkpoints are saved separately as `checkpoints/<module>_best.pt` and are what the inference pipeline loads.

---

## Inference

```python
from inference.pipeline import SentinelPipeline

pipeline = SentinelPipeline.from_pretrained("checkpoints/")

result = pipeline.verify(
    selfie_path="face.jpg",
    document_path="passport.jpg",
)

print(result.decision)       # ACCEPT | REVIEW | REJECT
print(f"{result.trust_score:.3f}")
print(result.explanations)
# {
#   "liveness":    0.94,
#   "anti_deepfake": 0.97,
#   "face_match":  0.88,
#   "behavioral":  0.81,
#   "document":    0.96,
# }
```

### REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
|---|---|---|
| POST | `/verify` | Full pipeline, returns `VerificationResult` |
| POST | `/verify/liveness` | Liveness check only |
| POST | `/verify/face` | Face match against enrolled identity |
| POST | `/verify/document` | Document OCR + forgery check |
| GET | `/health` | Model load status per module |

---

## Project structure

```
sentinelid/
├── models/
│   ├── liveness/           depth_liveness.py     ResNet50 + FPN + BerHu
│   ├── deepfake/           cnn_transformer.py    EfficientNet-B4 + FFT + cross-attention
│   ├── face_recognition/   arcface.py            iResNet + ArcFace + FAISS index
│   ├── behavioral/         au_gnn.py             GATConv graph + gaze regression
│   ├── document/           layout_intelligence.py LayoutLMv3-style + forgery head
│   ├── fusion/             score_fusion.py        MLP + Platt calibration
│   └── edge/               distillation.py        MobileNetV3 student + KD loss
├── training/               one script per module, fully configurable via YAML
├── configs/                per-module YAML configs with A100 defaults
├── data/datasets/          dataset loaders for all sources
├── inference/              full pipeline + edge pipeline
├── api/                    FastAPI server
├── evaluation/             ISO 30107-3 metrics, per-demographic bias audit
├── explainability/         GradCAM, score explanation
└── scripts/                dataset download, preprocessing, frame extraction
```

---

## Datasets

| Dataset | Scale | Module |
|---|---|---|
| MS-Celeb-1M | 10M images, 100K identities | Face recognition |
| VGGFace2 | 3.3M images, 9K identities | Face recognition |
| CASIA-WebFace | 494K images, 10K identities | Face recognition |
| FaceForensics++ | 1K real + 4K fake videos (HQ/LQ/raw) | Deepfake |
| CelebDF-v2 | 590 real + 5639 fake videos | Deepfake |
| WildDeepfake | 7314 sequences, in-the-wild conditions | Deepfake robustness |
| NUAA | 7509 images, print attacks | Liveness |
| MSU-MFSD | 280 videos, replay + print | Liveness |
| CASIA-SURF | RGB + IR + depth trimodal | Liveness (depth supervision) |
| SiW | 165 subjects, 4 spoof types | Liveness |
| DISFA | 27 subjects, 130K frames | AU intensities |
| BP4D | 41 subjects, 22 AUs | AU classification |
| MPIIGaze | 15 subjects, 213K samples | Gaze estimation |
| MIDV-500 | 500 document types, 15 conditions | Document OCR + forgery |
| MIDV-2020 | 10 document types, video | Document OCR |

---

## License

MIT
