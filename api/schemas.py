
from pydantic import BaseModel, Field


class VerifyResponse(BaseModel):
    trust_score: float = Field(..., ge=0.0, le=1.0, description="Overall identity trust score")
    decision: str = Field(..., description="ACCEPT | REVIEW | REJECT")
    explanations: dict[str, float] = Field(..., description="Per-module weighted contribution")
    raw_scores: dict[str, float] = Field(..., description="Raw module output scores")
    latency_ms: float = Field(..., description="End-to-end pipeline latency in milliseconds")


class LivenessResponse(BaseModel):
    liveness_score: float = Field(..., ge=0.0, le=1.0)
    is_live: bool
    latency_ms: float


class FaceMatchResponse(BaseModel):
    similarity: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity")
    match: bool
    latency_ms: float


class DocumentResponse(BaseModel):
    ocr_text: dict[str, str] = Field(..., description="Extracted fields from document")
    is_genuine: bool
    forgery_score: float = Field(..., ge=0.0, le=1.0, description="P(forgery)")
    document_type: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    models_loaded: bool
    device: str


class MetricsResponse(BaseModel):
    request_counts: dict[str, int]
    p95_latency_ms: dict[str, float]
