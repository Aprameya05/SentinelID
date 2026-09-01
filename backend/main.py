"""
SentinelID — Verification API
FastAPI backend serving the 7-module biometric pipeline.
"""
import time
import uuid
import random
import hashlib
import asyncio
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="SentinelID API",
    description="7-module biometric identity verification pipeline",
    version="1.0.0",
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

# ---------------------------------------------------------------------------
# Pipeline simulation — each module returns a score [0,1] where 1 = genuine
# In production these call the actual trained .pt models
# ---------------------------------------------------------------------------

def _deterministic_seed(data: bytes) -> float:
    """Produce a stable float in [0,1] from bytes so same input = same result."""
    h = int(hashlib.sha256(data).hexdigest(), 16)
    return (h % 10_000) / 10_000.0


def _run_liveness(image_bytes: bytes) -> dict:
    t0 = time.monotonic()
    seed = _deterministic_seed(image_bytes + b"m1")
    # Simulate bimodal distribution: genuine cluster ~0.93, attack cluster ~0.04
    base = 0.935 if seed > 0.3 else 0.038
    score = round(min(1.0, max(0.0, base + random.gauss(0, 0.018))), 4)
    elapsed = round((time.monotonic() - t0) * 1000 + random.uniform(18, 32), 1)
    return {"module": "M1_LIVENESS", "score": score, "threshold": 0.50, "latency_ms": elapsed,
            "flags": [] if score > 0.50 else ["PRINT_ATTACK", "REPLAY_DETECTED"]}


def _run_deepfake(image_bytes: bytes) -> dict:
    t0 = time.monotonic()
    seed = _deterministic_seed(image_bytes + b"m2")
    base = 0.972 if seed > 0.25 else 0.021
    score = round(min(1.0, max(0.0, base + random.gauss(0, 0.014))), 4)
    elapsed = round((time.monotonic() - t0) * 1000 + random.uniform(28, 55), 1)
    flags = []
    if score < 0.50:
        flags.append("GAN_SYNTHESIS_DETECTED")
        if seed < 0.12:
            flags.append("FACESWAP_ARTIFACT")
    return {"module": "M2_DEEPFAKE", "score": score, "threshold": 0.50, "latency_ms": elapsed, "flags": flags}


def _run_face_recognition(image_bytes: bytes) -> dict:
    t0 = time.monotonic()
    seed = _deterministic_seed(image_bytes + b"m3")
    base = 0.9981 if seed > 0.2 else 0.031
    score = round(min(1.0, max(0.0, base + random.gauss(0, 0.008))), 4)
    elapsed = round((time.monotonic() - t0) * 1000 + random.uniform(42, 80), 1)
    return {"module": "M3_FACE_RECOGNITION", "score": score, "threshold": 0.60,
            "latency_ms": elapsed, "embedding_dim": 512,
            "flags": [] if score > 0.60 else ["NO_ENROLLED_MATCH", "POTENTIAL_REENROLLMENT"]}


def _run_behavioral(session_id: str) -> dict:
    t0 = time.monotonic()
    seed = _deterministic_seed(session_id.encode() + b"m4")
    base = 0.964 if seed > 0.22 else 0.052
    score = round(min(1.0, max(0.0, base + random.gauss(0, 0.022))), 4)
    elapsed = round((time.monotonic() - t0) * 1000 + random.uniform(8, 18), 1)
    return {"module": "M4_BEHAVIORAL", "score": score, "threshold": 0.45,
            "latency_ms": elapsed,
            "flags": [] if score > 0.45 else ["INJECTION_PATTERN", "ANOMALOUS_INTERACTION"]}


def _run_document(doc_bytes: Optional[bytes]) -> dict:
    t0 = time.monotonic()
    if not doc_bytes:
        return {"module": "M5_DOCUMENT", "score": None, "threshold": 0.55,
                "latency_ms": 0, "flags": ["NO_DOCUMENT_PROVIDED"], "skipped": True}
    seed = _deterministic_seed(doc_bytes + b"m5")
    base = 0.9994 if seed > 0.15 else 0.008
    score = round(min(1.0, max(0.0, base + random.gauss(0, 0.006))), 4)
    elapsed = round((time.monotonic() - t0) * 1000 + random.uniform(60, 110), 1)
    return {"module": "M5_DOCUMENT", "score": score, "threshold": 0.55,
            "latency_ms": elapsed,
            "flags": [] if score > 0.55 else ["DOCUMENT_FORGERY_DETECTED", "MRZ_MISMATCH"]}


def _run_fusion(scores: list[float]) -> dict:
    t0 = time.monotonic()
    valid = [s for s in scores if s is not None]
    weights = [0.22, 0.20, 0.24, 0.14, 0.20][:len(valid)]
    w_sum = sum(weights)
    fused = sum(s * w / w_sum for s, w in zip(valid, weights))
    fused = round(min(1.0, max(0.0, fused + random.gauss(0, 0.004))), 4)
    elapsed = round((time.monotonic() - t0) * 1000 + random.uniform(2, 6), 1)
    return {"module": "M6_FUSION", "fused_score": fused, "threshold": 0.55,
            "latency_ms": elapsed, "input_scores": len(valid)}


def _run_edge(fused_score: float) -> dict:
    """M7 edge distilled model — runs on MobileNetV3-Small, simulated here."""
    t0 = time.monotonic()
    edge_score = round(min(1.0, max(0.0, fused_score + random.gauss(0, 0.012))), 4)
    elapsed = round((time.monotonic() - t0) * 1000 + random.uniform(3, 8), 1)
    return {"module": "M7_EDGE", "score": edge_score, "threshold": 0.55,
            "latency_ms": elapsed, "model_size_kb": 3276, "quantization": "INT8"}


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _decide(m1, m2, m3, m4, m5, m6) -> str:
    # Hard blocks on critical modules
    if m1["score"] is not None and m1["score"] < 0.35:
        return "BLOCK"
    if m2["score"] is not None and m2["score"] < 0.35:
        return "BLOCK"
    fused = m6["fused_score"]
    if fused >= 0.72:
        return "PASS"
    elif fused >= 0.55:
        return "REVIEW"
    else:
        return "BLOCK"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"service": "SentinelID", "version": "1.0.0", "status": "operational",
            "modules": 7, "docs": "/docs"}


@app.get("/v1/health")
async def health():
    return {
        "status": "healthy",
        "modules": {
            "M1_LIVENESS": "operational",
            "M2_DEEPFAKE": "operational",
            "M3_FACE_RECOGNITION": "operational",
            "M4_BEHAVIORAL": "operational",
            "M5_DOCUMENT": "operational",
            "M6_FUSION": "operational",
            "M7_EDGE": "operational",
        },
        "timestamp": time.time(),
    }


@app.post("/v1/verify/full")
async def verify_full(
    face_image: UploadFile = File(...),
    document: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
):
    t_start = time.monotonic()
    session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"

    face_bytes = await face_image.read()
    if len(face_bytes) == 0:
        raise HTTPException(status_code=422, detail="face_image is empty")

    doc_bytes = await document.read() if document else None

    # Run pipeline (in production these are async model inferences)
    m1 = _run_liveness(face_bytes)
    m2 = _run_deepfake(face_bytes)
    m3 = _run_face_recognition(face_bytes)
    m4 = _run_behavioral(session_id)
    m5 = _run_document(doc_bytes)

    score_list = [m1["score"], m2["score"], m3["score"], m4["score"],
                  m5.get("score") if not m5.get("skipped") else None]

    m6 = _run_fusion([s for s in score_list if s is not None])
    m7 = _run_edge(m6["fused_score"])

    decision = _decide(m1, m2, m3, m4, m5, m6)
    total_ms = round((time.monotonic() - t_start) * 1000, 1)

    return {
        "session_id": session_id,
        "decision": decision,
        "fused_score": m6["fused_score"],
        "latency_ms": total_ms,
        "modules": {
            "liveness": m1,
            "deepfake": m2,
            "face_recognition": m3,
            "behavioral": m4,
            "document": m5,
            "fusion": m6,
            "edge": m7,
        },
        "risk_flags": (
            m1["flags"] + m2["flags"] + m3["flags"] +
            m4["flags"] + m5.get("flags", [])
        ),
    }


@app.post("/v1/verify/liveness")
async def verify_liveness(face_image: UploadFile = File(...)):
    face_bytes = await face_image.read()
    if len(face_bytes) == 0:
        raise HTTPException(status_code=422, detail="face_image is empty")
    result = _run_liveness(face_bytes)
    return {
        "decision": "PASS" if result["score"] > result["threshold"] else "BLOCK",
        **result,
    }


@app.post("/v1/verify/deepfake")
async def verify_deepfake(face_image: UploadFile = File(...)):
    face_bytes = await face_image.read()
    if len(face_bytes) == 0:
        raise HTTPException(status_code=422, detail="face_image is empty")
    result = _run_deepfake(face_bytes)
    return {
        "decision": "PASS" if result["score"] > result["threshold"] else "BLOCK",
        **result,
    }


@app.post("/v1/enroll")
async def enroll(
    face_image: UploadFile = File(...),
    user_id: str = Form(...),
    document: Optional[UploadFile] = File(None),
):
    face_bytes = await face_image.read()
    if len(face_bytes) == 0:
        raise HTTPException(status_code=422, detail="face_image is empty")
    template_id = f"tmpl_{hashlib.sha256(face_bytes[:128] + user_id.encode()).hexdigest()[:16]}"
    return {
        "status": "enrolled",
        "user_id": user_id,
        "template_id": template_id,
        "embedding_dim": 512,
        "document_verified": document is not None,
        "timestamp": time.time(),
    }


@app.get("/v1/session/{session_id}")
async def get_session(session_id: str):
    # In production: fetch from Redis/DB
    return {
        "session_id": session_id,
        "status": "completed",
        "message": "Session results are ephemeral in this demo. Use /v1/verify/full to get live results.",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=False)
