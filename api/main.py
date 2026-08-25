"""
SentinelID REST API — FastAPI

Endpoints:
  POST /verify          Full multimodal verification
  POST /verify/liveness Liveness check only
  POST /verify/face     Face match against enrolled identity
  POST /verify/document Document OCR + forgery check
  GET  /health          Service status + loaded model info
  GET  /metrics         Prometheus-style metrics
"""

import time
import io
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2

from api.schemas import (
    VerifyResponse, LivenessResponse, FaceMatchResponse,
    DocumentResponse, HealthResponse, MetricsResponse,
)
from inference.pipeline import SentinelPipeline, PipelineConfig


# ──────────────────────────────────────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────────────────────────────────────

pipeline: SentinelPipeline | None = None
_start_time = time.time()
_request_counts: dict[str, int] = {}
_latencies: dict[str, list[float]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    config = PipelineConfig()
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    pipeline = SentinelPipeline.from_pretrained(checkpoint_dir, config)
    print("SentinelID pipeline loaded.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="SentinelID",
    description="Multimodal identity intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

async def read_image(upload: UploadFile) -> np.ndarray:
    data = await upload.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def get_pipeline() -> SentinelPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return pipeline


def record_latency(endpoint: str, latency: float):
    _request_counts[endpoint] = _request_counts.get(endpoint, 0) + 1
    if endpoint not in _latencies:
        _latencies[endpoint] = []
    _latencies[endpoint].append(latency)
    if len(_latencies[endpoint]) > 1000:
        _latencies[endpoint] = _latencies[endpoint][-500:]


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/verify", response_model=VerifyResponse, summary="Full multimodal verification")
async def verify(
    selfie: UploadFile = File(..., description="Selfie image (JPEG/PNG)"),
    document: Optional[UploadFile] = File(None, description="ID document image"),
    enrolled_embedding: Optional[str] = None,
    pl: SentinelPipeline = Depends(get_pipeline),
):
    """
    Run the full SentinelID verification pipeline on a selfie (and optionally
    a document image). Returns a trust score, ACCEPT/REVIEW/REJECT decision,
    and per-module explanation scores.
    """
    t0 = time.perf_counter()
    selfie_img = await read_image(selfie)

    doc_path = None
    if document is not None:
        doc_data = await document.read()
        doc_arr = np.frombuffer(doc_data, np.uint8)
        doc_img = cv2.imdecode(doc_arr, cv2.IMREAD_COLOR)
        # Save temp file for pipeline
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, doc_img)
        doc_path = tmp.name

    result = pl.verify(selfie_path=selfie_img, document_path=doc_path)

    latency = (time.perf_counter() - t0) * 1000
    record_latency("/verify", latency)

    return VerifyResponse(
        trust_score=result.trust_score,
        decision=result.decision,
        explanations=result.explanations,
        raw_scores=result.raw_scores,
        latency_ms=round(latency, 2),
    )


@app.post("/verify/liveness", response_model=LivenessResponse, summary="Liveness check only")
async def liveness_check(
    selfie: UploadFile = File(...),
    pl: SentinelPipeline = Depends(get_pipeline),
):
    t0 = time.perf_counter()
    img = await read_image(selfie)
    from inference.pipeline import load_and_preprocess
    tensor = load_and_preprocess(img)
    score, depth = pl.run_liveness(tensor)
    latency = (time.perf_counter() - t0) * 1000
    record_latency("/verify/liveness", latency)
    return LivenessResponse(
        liveness_score=score,
        is_live=score > pl.config.liveness_threshold,
        latency_ms=round(latency, 2),
    )


@app.post("/verify/face", response_model=FaceMatchResponse, summary="Face match")
async def face_match(
    selfie: UploadFile = File(...),
    enrolled_selfie: UploadFile = File(...),
    pl: SentinelPipeline = Depends(get_pipeline),
):
    t0 = time.perf_counter()
    from inference.pipeline import load_and_preprocess
    img_a = await read_image(selfie)
    img_b = await read_image(enrolled_selfie)

    emb_a, _ = pl.run_face_recognition(load_and_preprocess(img_a))
    emb_b, _ = pl.run_face_recognition(load_and_preprocess(img_b))

    cos_sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8))
    match = cos_sim > pl.config.face_match_threshold
    latency = (time.perf_counter() - t0) * 1000
    record_latency("/verify/face", latency)
    return FaceMatchResponse(
        similarity=round(cos_sim, 4),
        match=match,
        latency_ms=round(latency, 2),
    )


@app.get("/health", response_model=HealthResponse, summary="Service health")
async def health():
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
        models_loaded=pipeline is not None,
        device=str(pipeline.device) if pipeline else "unknown",
    )


@app.get("/metrics", response_model=MetricsResponse, summary="Request metrics")
async def metrics():
    def p95(lst):
        if not lst:
            return 0.0
        arr = sorted(lst)
        return round(arr[int(len(arr) * 0.95)], 2)

    return MetricsResponse(
        request_counts=dict(_request_counts),
        p95_latency_ms={k: p95(v) for k, v in _latencies.items()},
    )
