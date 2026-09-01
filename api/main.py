"""
api/main.py — SentinelID Production FastAPI Service

Endpoints:
  POST /verify           Full multimodal verification (selfie + optional doc)
  POST /verify/liveness  Liveness check only
  POST /verify/face      1:1 face match
  POST /verify/document  Document forgery check only
  POST /enroll           Enroll a new identity, returns face embedding
  GET  /health           Service health + model info
  GET  /metrics          Request counts + P50/P95/P99 latency per endpoint

Auth: Bearer token via X-API-Key header (set SENTINEL_API_KEY env var).
Rate limiting: 60 req/min per IP (in-process, resets on restart).
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.pipeline import PipelineConfig, SentinelPipeline

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

_API_KEY        = os.getenv("SENTINEL_API_KEY", "")      # empty = no auth
_CKPT_DIR       = Path(os.getenv("SENTINEL_CKPT_DIR", "checkpoints"))
_RATE_LIMIT_RPM = int(os.getenv("SENTINEL_RATE_LIMIT", "60"))
_MAX_IMAGE_MB   = 10

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

pipeline: SentinelPipeline | None = None
_start_time = time.time()

# Metrics: deque of (timestamp, latency_ms) per endpoint, capped at 10k
_latency_log: dict[str, deque] = defaultdict(lambda: deque(maxlen=10_000))
_request_counts: dict[str, int] = defaultdict(int)
_error_counts: dict[str, int] = defaultdict(int)

# Rate limiting: IP → deque of request timestamps
_rate_buckets: dict[str, deque] = defaultdict(lambda: deque(maxlen=_RATE_LIMIT_RPM))


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig()
    pipeline = SentinelPipeline.from_pretrained(_CKPT_DIR, config)
    print(f"[SentinelID] Pipeline loaded from {_CKPT_DIR} | device={pipeline.device}")
    yield
    print("[SentinelID] Shutting down.")


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SentinelID",
    description=(
        "Multimodal identity intelligence API — liveness, deepfake detection, "
        "face recognition, behavioral analysis, and document verification in one pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Middleware — rate limiting
# ──────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ("/health", "/metrics", "/docs", "/redoc", "/openapi.json"):
        return await call_next(request)

    ip = request.client.host or "unknown"
    now = time.monotonic()
    bucket = _rate_buckets[ip]

    # Drop timestamps older than 60 s
    while bucket and now - bucket[0] > 60.0:
        bucket.popleft()

    if len(bucket) >= _RATE_LIMIT_RPM:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded: {_RATE_LIMIT_RPM} requests/minute."},
        )

    bucket.append(now)
    return await call_next(request)


# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(default="")):
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Set X-API-Key header.",
        )


def get_pipeline() -> SentinelPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised yet — try again shortly.")
    return pipeline


async def load_upload(upload: UploadFile, max_mb: float = _MAX_IMAGE_MB) -> str:
    """Read an uploaded image to a temp file; return the file path."""
    data = await upload.read()
    if len(data) > max_mb * 1_000_000:
        raise HTTPException(400, f"Image too large (max {max_mb} MB)")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"Cannot decode image from '{upload.filename}'. Send JPEG or PNG.")
    suffix = Path(upload.filename or "image.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    cv2.imwrite(tmp.name, img)
    tmp.close()
    return tmp.name


def _record(endpoint: str, latency_ms: float):
    _request_counts[endpoint] += 1
    _latency_log[endpoint].append((time.time(), latency_ms))


def _percentile(endpoint: str, pct: float) -> float:
    vals = [v for _, v in _latency_log[endpoint]]
    if not vals:
        return 0.0
    return float(np.percentile(vals, pct))


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.post(
    "/verify",
    summary="Full multimodal verification",
    tags=["Verification"],
    dependencies=[Depends(verify_api_key)],
)
async def verify(
    selfie: UploadFile = File(..., description="Selfie JPEG/PNG"),
    document: Optional[UploadFile] = File(None, description="ID document JPEG/PNG (optional)"),
    pl: SentinelPipeline = Depends(get_pipeline),
):
    """
    Run the complete 7-module SentinelID pipeline.

    Returns `trust_score` ∈ [0,1], `decision` ∈ {ACCEPT, REVIEW, REJECT},
    per-module `raw_scores`, and a human-readable `explanation`.
    """
    t0 = time.perf_counter()
    selfie_path = await load_upload(selfie)
    doc_path = await load_upload(document) if document else None

    try:
        result = pl.verify(selfie_path=selfie_path, document_path=doc_path)
    finally:
        os.unlink(selfie_path)
        if doc_path:
            os.unlink(doc_path)

    latency = (time.perf_counter() - t0) * 1000
    _record("/verify", latency)

    return {
        "trust_score":   round(result.trust_score, 4),
        "decision":      result.decision,
        "raw_scores":    {k: round(v, 4) for k, v in result.raw_scores.items()},
        "explanations":  result.explanations,
        "latency_ms":    round(latency, 1),
    }


@app.post(
    "/verify/liveness",
    summary="Liveness check only",
    tags=["Verification"],
    dependencies=[Depends(verify_api_key)],
)
async def liveness_check(
    selfie: UploadFile = File(...),
    pl: SentinelPipeline = Depends(get_pipeline),
):
    """ISO 30107-3 passive liveness. Returns P(live) and depth confidence."""
    t0 = time.perf_counter()
    path = await load_upload(selfie)
    try:
        score, depth = pl.run_liveness(path)
    finally:
        os.unlink(path)

    latency = (time.perf_counter() - t0) * 1000
    _record("/verify/liveness", latency)
    return {
        "liveness_score": round(score, 4),
        "is_live":        score >= pl.cfg.liveness_threshold,
        "threshold":      pl.cfg.liveness_threshold,
        "latency_ms":     round(latency, 1),
    }


@app.post(
    "/verify/face",
    summary="1:1 face verification",
    tags=["Verification"],
    dependencies=[Depends(verify_api_key)],
)
async def face_match(
    selfie:          UploadFile = File(..., description="Probe selfie"),
    enrolled_selfie: UploadFile = File(..., description="Enrolled identity selfie"),
    pl: SentinelPipeline = Depends(get_pipeline),
):
    """Compare two face images. Returns cosine similarity and match decision."""
    t0 = time.perf_counter()
    path_a = await load_upload(selfie)
    path_b = await load_upload(enrolled_selfie)
    try:
        emb_a, _ = pl.run_face_recognition(path_a)
        emb_b, _ = pl.run_face_recognition(path_b)
    finally:
        os.unlink(path_a)
        os.unlink(path_b)

    a = emb_a / (np.linalg.norm(emb_a) + 1e-8)
    b = emb_b / (np.linalg.norm(emb_b) + 1e-8)
    sim = float(np.dot(a, b))
    latency = (time.perf_counter() - t0) * 1000
    _record("/verify/face", latency)
    return {
        "similarity":  round(sim, 4),
        "match":       sim >= pl.cfg.face_match_threshold,
        "threshold":   pl.cfg.face_match_threshold,
        "latency_ms":  round(latency, 1),
    }


@app.post(
    "/verify/document",
    summary="Document forgery check",
    tags=["Verification"],
    dependencies=[Depends(verify_api_key)],
)
async def document_check(
    document: UploadFile = File(..., description="ID document image"),
    pl: SentinelPipeline = Depends(get_pipeline),
):
    """Classify document type and detect forgery signals."""
    t0 = time.perf_counter()
    path = await load_upload(document)
    try:
        score, meta = pl.run_document(path)
    finally:
        os.unlink(path)

    latency = (time.perf_counter() - t0) * 1000
    _record("/verify/document", latency)
    return {
        "authenticity_score": round(score, 4),
        "is_authentic":       score >= 0.5,
        **meta,
        "latency_ms": round(latency, 1),
    }


@app.post(
    "/enroll",
    summary="Enroll a new identity",
    tags=["Identity"],
    dependencies=[Depends(verify_api_key)],
)
async def enroll(
    selfie: UploadFile = File(..., description="Clear frontal face selfie"),
    pl: SentinelPipeline = Depends(get_pipeline),
):
    """
    Extract and return a 512-d face embedding for enrollment.
    Store this vector server-side and pass it to `/verify` as `enrolled_embedding`.
    """
    t0 = time.perf_counter()
    path = await load_upload(selfie)
    try:
        embedding = pl.enroll(path)
    finally:
        os.unlink(path)

    latency = (time.perf_counter() - t0) * 1000
    _record("/enroll", latency)
    return {
        "embedding":     embedding.tolist(),
        "embedding_dim": len(embedding),
        "latency_ms":    round(latency, 1),
    }


@app.get("/health", summary="Service health", tags=["Ops"])
async def health():
    """Returns service status, uptime, and loaded model device."""
    return {
        "status":          "ok" if pipeline is not None else "loading",
        "uptime_seconds":  round(time.time() - _start_time, 1),
        "models_loaded":   pipeline is not None,
        "device":          str(pipeline.device) if pipeline else "unknown",
        "version":         "1.0.0",
        "checkpoint_dir":  str(_CKPT_DIR),
    }


@app.get("/metrics", summary="Request metrics", tags=["Ops"])
async def metrics():
    """Returns request counts and P50/P95/P99 latency per endpoint."""
    return {
        "request_counts": dict(_request_counts),
        "error_counts":   dict(_error_counts),
        "latency_ms": {
            ep: {
                "p50": round(_percentile(ep, 50), 1),
                "p95": round(_percentile(ep, 95), 1),
                "p99": round(_percentile(ep, 99), 1),
                "n":   len(list(_latency_log[ep])),
            }
            for ep in _latency_log
        },
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dev entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
