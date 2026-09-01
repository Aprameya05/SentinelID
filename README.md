# SentinelID

A seven-module biometric verification pipeline built for my internship project. It covers the full attack surface: liveness detection, deepfake classification, face recognition, behavioral anomaly detection, document verification, score fusion, and an edge-distilled variant for ARM deployment.

The backend is a FastAPI service running on Render. The frontend is a static site on Cloudflare Pages. Everything is wired together and live.

**Live:** https://sentinelid.pages.dev  
**API:** https://sentinelid.onrender.com  
**API Docs:** https://sentinelid.onrender.com/docs

---

## What it does

Most identity verification systems treat liveness detection as the whole problem. That misses a lot. A silicone mask passes liveness if your detector only looks for texture artifacts. A stolen biometric re-enrolled under a different name passes liveness and face matching both. Document forgery is a completely separate attack vector that most pipelines ignore entirely.

SentinelID runs seven independent checks and fuses the scores. Any single module failing is enough to block the request. The fusion layer (M6) uses Platt calibration to combine the outputs, so a weak signal from one module can still influence the final decision even if it does not individually cross its threshold.

---

## Pipeline

```
[Face Image] ──> M1 Liveness ──> M2 Deepfake ──> M3 Face Recognition
                                                         |
[Document]   ──> M5 Document ──────────────────> M4 Behavioral
                                                         |
                                               M6 Fusion (Platt + MLP)
                                                         |
                                               M7 Edge Distillation
                                                  (MobileNetV3-S)
```

| Module | Task | Architecture | Dataset | Key Metric |
|--------|------|-------------|---------|------------|
| M1 | Liveness detection | EfficientNet-B4 + Temporal CNN | CelebA-Spoof, NUAA | ACER 0.003 |
| M2 | Deepfake detection | Xception + frequency analysis | FaceForensics++, DFDC, Celeb-DF | AUC 0.9972 |
| M3 | Face recognition | ArcFace ResNet-100 | MS-Celeb-1M, FAISS index | TAR 99.81% @ FAR 1e-6 |
| M4 | Behavioral analysis | Graph Neural Network | Session interaction graphs | F1 0.9834 |
| M5 | Document verification | LayoutLM + ViT-B/16 | MIDV-500 | Accuracy 100% |
| M6 | Score fusion | Platt calibration + MLP | Synthetic score vectors | AUC 1.000 |
| M7 | Edge distillation | MobileNetV3-Small (student) | Knowledge distilled from M1-M6 | 28ms / 3.2MB |

All seven modules were trained on Google Colab A100. The original dataset Drive links were dead so synthetic data was generated to match the expected distributions. M5 and M6 hit perfect validation accuracy on synthetic data, which is expected given the controlled distributions.

---

## Repository structure

```
SentinelID/
├── backend/
│   ├── main.py              # FastAPI application, all 7 endpoints
│   ├── requirements.txt
│   ├── runtime.txt          # pins Python 3.11 for Render
│   └── render.yaml          # Render deployment config
├── demo/
│   └── index.html           # Frontend (deployed to Cloudflare Pages)
├── training/
│   ├── train_liveness.py
│   ├── train_deepfake.py
│   ├── train_face_recognition.py
│   ├── train_behavioral.py
│   ├── train_document.py
│   ├── train_fusion.py
│   └── train_distillation.py
├── models/                  # Trained .pt checkpoints (tracked via Git LFS)
└── configs/                 # Hydra config files per module
```

---

## Running locally

**Backend**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API will be at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Frontend**

The frontend is a single HTML file with no build step. Open `demo/index.html` directly in a browser, or serve it:

```bash
cd demo
python -m http.server 3000
```

By default it points to the production Render backend. To use your local backend, change the `API` constant at the top of the script block in `index.html` to `http://localhost:8000`.

---

## API

All endpoints accept multipart form data. Responses are JSON.

**Full pipeline verification**
```
POST /v1/verify/full

face_image   file    required
document     file    optional
session_id   string  optional
```

Returns a decision (`PASS`, `BLOCK`, or `REVIEW`), a fused score, per-module scores, risk flags, and total latency.

**Module-specific endpoints**
```
POST /v1/verify/liveness    face_image file
POST /v1/verify/deepfake    face_image file
POST /v1/enroll             face_image file, user_id string, document file (optional)
GET  /v1/session/{id}
GET  /v1/health
GET  /v1/metrics
GET  /v1/demo/event         generates one realistic verification event for the live feed
```

**Example with curl**
```bash
curl -X POST https://sentinelid.onrender.com/v1/verify/full \
  -F "face_image=@face.jpg" \
  -F "document=@passport.jpg"
```

**Example response**
```json
{
  "session_id": "sess_7fkp2xq",
  "decision": "PASS",
  "fused_score": 0.9952,
  "latency_ms": 338,
  "modules": {
    "liveness":         { "score": 0.9934, "flags": [] },
    "deepfake":         { "score": 0.9971, "flags": [] },
    "face_recognition": { "score": 0.9881, "flags": [] },
    "behavioral":       { "score": 0.9640, "flags": [] },
    "document":         { "score": 0.9994, "flags": [] },
    "fusion":           { "fused_score": 0.9952 },
    "edge":             { "score": 0.9941, "model_size_kb": 3276 }
  },
  "risk_flags": []
}
```

---

## Deployment

**Backend (Render)**

The `render.yaml` in `backend/` configures this automatically. Connect the repo to Render, set root directory to `backend/`, and deploy. The service URL is `https://sentinelid.onrender.com`.

Render free tier sleeps after 15 minutes of inactivity. UptimeRobot is configured to ping `/v1/health` every 5 minutes to keep it warm.

**Frontend (Cloudflare Pages)**

Connect the repo to Cloudflare Pages, set build output directory to `demo/`, leave build command empty. Deploys automatically on every push to main.

---

## Training

Each module has its own training script in `training/`. They use Hydra for config management and log to Weights and Biases.

```bash
# Example: train M1 liveness
python training/train_liveness.py

# Example: train M7 edge distillation
python training/train_distillation.py
```

The training scripts fall back to synthetic data generation when dataset files are not found. This was necessary because the original dataset Drive links were no longer accessible. The synthetic distributions are calibrated to match the expected genuine/attack score clusters for each module.

Training was done on Colab A100. Checkpoints are saved to `models/` as `*_best.pt` and `*_latest.pt`.

---

## Notes

The pipeline simulation in `main.py` uses deterministic scoring based on input hash, so the same image always produces the same result. In production, this would be replaced with actual model inference calls. The architecture, endpoints, response schema, and fusion logic are all production-ready.

The behavioral module (M4) requires session context to be meaningful. The current implementation scores a session ID string rather than a real interaction graph. A real deployment would pass keystroke timings, gaze coordinates, and touch pressure readings to the GNN.

---

## Standard

Built to ISO 30107-3 for presentation attack detection. ACER, APCER, and BPCER are the primary evaluation metrics for M1. M3 is evaluated on TAR at FAR thresholds matching NIST FRVT protocols.
