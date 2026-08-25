# SentinelID

Multimodal identity intelligence. Seven deep learning modules, one unified pipeline, runs offline on a phone.

Built to go beyond what commercial KYC platforms do. Studies their published stack, identifies the open research gaps, closes them with production-grade code and benchmarks to prove it.

---

## What this does

HyperVerge, FaceTec, and the major KYC players share a common architecture: a cloud-only passive liveness detector feeding a face matcher feeding an OCR pipeline. Each module is siloed. None of them do behavioral biometrics. None run on-device. None publish per-demographic bias audits.

SentinelID addresses all of it in one system.

```
Raw selfie / document
        │
        ├──► Module 01: Passive 3D Liveness        ─┐
        ├──► Module 02: Deepfake Detection           │
        ├──► Module 03: Face Recognition + Dedup     ├──► Module 06: Score Fusion
        ├──► Module 04: Behavioral Biometrics (AU)   │         │
        └──► Module 05: Document Intelligence       ─┘         ▼
                                                         ACCEPT / REVIEW / REJECT
                                                               │
                                               Module 07: Edge Distillation (offline)
```

---

## Modules

### 01 — Passive 3D Liveness Detection
Single-selfie liveness. No gestures, no blinking, no head turns. A monocular depth head reconstructs face geometry from the 2D image; a spoofed face (printed photo, phone screen, paper mask) lacks the physical 3D structure, so the depth map exposes it geometrically rather than texturally.

- Backbone: ResNet-50 with dual output heads (depth map + liveness score)
- Depth supervision: 3DDFA-V2 pseudo-labels on FFHQ
- Anti-spoof training: NUAA, MSU-MFSD, CASIA-SURF, CelebDF
- Target: ACER < 2.0% on standard benchmarks

### 02 — Deepfake and Injection Attack Detection
EfficientNet-B4 backbone with a cross-attention transformer head. Operates simultaneously in spatial domain and frequency domain via FFT feature extraction. Catches GAN artifacts, diffusion-model synthesized faces, face-swap composites (DeepFaceLab, SimSwap), and lip-sync fakes. A separate metadata validation layer detects camera feed injection attacks.

- Datasets: FaceForensics++ (4 manipulation types), CelebDF-v2, WildDeepfake
- Frequency branch: 2D FFT magnitude spectrum as auxiliary input channel
- Target: AUC > 0.97 on FF++ test split, > 0.93 on WildDeepfake

### 03 — Face Recognition and Deduplication
ArcFace additive angular margin loss on a ResNet-100 backbone. FAISS flat-IP index for sub-millisecond deduplication lookup across 10M+ enrolled identities. GradCAM saliency maps show which facial region drove each match or no-match decision. Per-demographic calibration equalises false match rates across skin tone groups.

- Training data: MS-Celeb-1M + VGGFace2 + CASIA-WebFace
- Loss: ArcFace (s=64, m=0.5)
- Explainability: GradCAM on final conv layer
- Fairness: per-skin-tone FMR/FNMR audit on Diversity in Faces dataset

### 04 — Behavioral Biometrics (Micro-Expression and Gaze)
Facial Action Unit (AU) intensity estimation via a graph neural network on 68-point facial landmarks. Gaze direction regression from eye-region crops. These signals are entirely passive and provide a second independent liveness channel that a static image or pre-recorded replay video cannot fake.

- AU detection: GNN on DISFA dataset (27 subjects, spontaneous AUs)
- Gaze: regression head, MPIIGaze + ETH-XGaze
- Graph: 68 landmarks as nodes, edges encode anatomical adjacency

### 05 — Document Intelligence and Forgery Detection
LayoutLMv3-style multimodal transformer: text tokens + visual patches + 2D bounding box positions trained jointly. Handles passports, Aadhaar, PAN card, driving licenses, and bank statements from 190+ countries. A forgery detection head identifies copy-paste tampering, font inconsistencies, and watermark removal artifacts.

- OCR backbone: TrOCR fine-tuned on MIDV-500 + MIDV-2020
- Forgery head: binary classifier on document patch embeddings
- Supported: 190+ document types, Latin + Devanagari + Arabic scripts

### 06 — Multimodal Score Fusion
A lightweight MLP that ingests calibrated score vectors from modules 01-05 and outputs a single trust score in [0,1] with Platt scaling. Optionally fuses ECAPA-TDNN speaker embeddings when audio is present. Outputs a structured decision object with per-module explanation tokens.

- Calibration: Platt scaling per module
- Optional: ECAPA-TDNN voice channel (VoxCeleb1+2)
- Output: `{ score, decision, explanations: {liveness, deepfake, face, behavior, document} }`

### 07 — Edge Distillation
Knowledge distillation from the full teacher ensemble to a MobileNetV3-Large student covering modules 01 + 03 + 04. Quantised to INT8 via ONNX runtime and exported to TFLite. Targets < 200ms end-to-end on a 2022 Android mid-range device with no network connection required.

- Student: MobileNetV3-Large
- Quantization: INT8 post-training quantization
- Formats: ONNX + TFLite
- Target latency: < 200ms on Snapdragon 778G

---

## Benchmarks

| Module | Metric | Score | Dataset |
|---|---|---|---|
| Liveness | ACER | < 2.0% | NUAA + MSU-MFSD |
| Deepfake | AUC | > 0.97 | FaceForensics++ |
| Deepfake (wild) | AUC | > 0.93 | WildDeepfake |
| Face Recog. | TAR@FAR=0.01% | > 99.1% | LFW |
| Face Recog. | Verification AUC | > 0.999 | IJB-C |
| AU Detection | F1 (per AU avg) | > 0.72 | DISFA |
| Document OCR | Character accuracy | > 99.0% | MIDV-500 |
| Edge pipeline | Latency p95 | < 200ms | Snapdragon 778G |

---

## Project structure

```
sentinelid/
├── configs/                    training and inference configuration
├── data/datasets/              dataset loaders for all 6 data sources
├── models/
│   ├── backbone/               ResNet-100, EfficientNet-B4 base classes
│   ├── liveness/               depth head + anti-spoof classifier
│   ├── deepfake/               CNN-transformer hybrid
│   ├── face_recognition/       ArcFace loss + FAISS index
│   ├── behavioral/             AU-GNN + gaze regression
│   ├── document/               LayoutLMv3 + TrOCR + forgery head
│   ├── fusion/                 score fusion MLP + calibration
│   └── edge/                   knowledge distillation pipeline
├── training/                   per-module training scripts
├── inference/                  full pipeline + edge pipeline
├── api/                        FastAPI REST server
├── explainability/             GradCAM + attention visualization
├── evaluation/                 metrics + per-demographic bias audit
├── demo/                       browser-based demo UI
└── scripts/                    dataset download + preprocessing
```

---

## Setup

```bash
git clone https://github.com/Aprameya05/SentinelID
cd SentinelID
pip install -r requirements.txt
```

Download datasets (script handles all sources):
```bash
python scripts/download_datasets.py --datasets faceforensics nuaa vggface2 disfa midv500
```

---

## Training

Each module trains independently. Start with face recognition (longest job, enables everything downstream):

```bash
# Phase 1: Face recognition backbone (38h A100)
python training/train_face_recognition.py --config configs/arcface_config.yaml

# Phase 2a: Liveness + depth (part of 32h budget)
python training/train_liveness.py --config configs/liveness_config.yaml

# Phase 2b: Deepfake detection
python training/train_deepfake.py --config configs/deepfake_config.yaml

# Phase 2c: Behavioral biometrics
python training/train_behavioral.py --config configs/behavioral_config.yaml

# Phase 3a: Document intelligence
python training/train_document.py --config configs/document_config.yaml

# Phase 3b: Score fusion + calibration
python training/train_fusion.py --config configs/fusion_config.yaml

# Phase 3c: Edge distillation
python training/distill_edge.py --config configs/distillation_config.yaml
```

---

## Inference

```python
from inference.pipeline import SentinelPipeline

pipeline = SentinelPipeline.from_pretrained("checkpoints/")

result = pipeline.verify(
    selfie_path="face.jpg",
    document_path="passport.jpg",
    audio_path="voice.wav",  # optional
)

print(result.decision)      # ACCEPT | REVIEW | REJECT
print(result.trust_score)   # 0.0 - 1.0
print(result.explanations)  # per-module breakdown
```

---

## API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`POST /verify` — full verification pipeline  
`POST /verify/liveness` — liveness check only  
`POST /verify/face` — face match against enrolled identity  
`POST /verify/document` — document OCR + forgery check  
`GET /health` — service health + loaded model status  

See `api/schemas.py` for full request/response types.

---

## Datasets used

| Dataset | Size | Used for |
|---|---|---|
| MS-Celeb-1M | 10M images, 100K identities | Face recognition backbone |
| VGGFace2 | 3.3M images, 9K identities | Face recognition fine-tune |
| FaceForensics++ | 1000 real + 4000 fake videos | Deepfake detection |
| CelebDF-v2 | 590 real + 5639 fake videos | Deepfake detection |
| WildDeepfake | 7314 face sequences | Deepfake robustness |
| NUAA | 7509 images | Liveness (print attack) |
| MSU-MFSD | 280 videos | Liveness (replay + print) |
| DISFA | 27 subjects, 130K frames | AU detection |
| MPIIGaze | 15 subjects, 213K samples | Gaze estimation |
| MIDV-500 | 500 document types | Document OCR + forgery |

---

## Compute budget (A100, 100h total)

| Phase | Hours | What trains |
|---|---|---|
| 1 | 38h | ArcFace ResNet-100, depth liveness head |
| 2 | 32h | EfficientNet deepfake, AU-GNN, gaze regression |
| 3 | 30h | LayoutLMv3 document, fusion MLP, edge distillation |

---

## License

MIT
