"""Standalone TTS service stub aligned with the provided HTTP API."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


MODEL_ID = "voice_lab/liliya"
DEFAULT_SPEAKER = "liliya"
SAMPLE_RATE = 24000
MAX_TEXT_CHARS = 500
STUB_AUDIO_B64 = (
    "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqF"
)

app = FastAPI(title="Standalone TTS HTTP API", version="0.1.0")


class TTSMetrics(BaseModel):
    """Performance metrics returned by the TTS API."""

    total_ms: float
    audio_duration_s: float
    rtf: float


class TTSGenerateResponse(BaseModel):
    """Non-streaming TTS response matching API.md."""

    audio_b64: str
    sample_rate: int = SAMPLE_RATE
    metrics: TTSMetrics


class TTSStatusResponse(BaseModel):
    """Service status response matching API.md."""

    service: str = "tts"
    loaded: bool = True
    model: Optional[str] = MODEL_ID
    loading: bool = False
    model_type: Optional[str] = "custom_voice"
    speakers: list[str] = Field(default_factory=lambda: [DEFAULT_SPEAKER])
    available_models: list[str] = Field(default_factory=lambda: [MODEL_ID])
    trained_voice_lab_speakers: list[str] = Field(default_factory=lambda: [DEFAULT_SPEAKER])
    cached_models: list[str] = Field(default_factory=lambda: [MODEL_ID])
    queue_depth: int = 0
    preset_refs: list[dict[str, str]] = Field(default_factory=list)
    emotion_refs_enabled: bool = False
    drama_styles: list[dict[str, str]] = Field(
        default_factory=lambda: [
            {
                "id": "happy",
                "label": "开心",
                "folder": "happy",
                "instruct": "用开心、轻快、带笑意的口气说，语速略活泼",
            },
            {
                "id": "neutral",
                "label": "平静",
                "folder": "neutral",
                "instruct": "用平和、自然的口气说",
            },
        ]
    )
    voice_lab: dict[str, object] = Field(
        default_factory=lambda: {
            "enabled": True,
            "speaker": DEFAULT_SPEAKER,
            "speakers": [DEFAULT_SPEAKER],
            "model_id": MODEL_ID,
            "checkpoint": "stub",
            "ready": True,
            "dir": "voice_lab",
        }
    )


class ModelLoadResponse(BaseModel):
    """Model load response."""

    status: str
    model: str


class PrewarmResponse(BaseModel):
    """Voice clone prewarm response."""

    status: str
    xvec_only: Optional[bool] = None
    ref_preset: Optional[str] = None
    reason: Optional[str] = None


@app.get("/")
async def index() -> dict[str, str]:
    """Return service index paths."""

    return {
        "service": "tts",
        "docs": "/docs",
        "health": "/api/v1/health",
        "status": "/api/v1/status",
    }


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    """Return HTTP service health without checking model state."""

    return {"status": "ok", "service": "tts"}


@app.get("/api/v1/status", response_model=TTSStatusResponse)
async def status() -> TTSStatusResponse:
    """Return model and voice-lab status."""

    return TTSStatusResponse()


@app.get("/api/v1/preset_ref/{preset_id}")
async def preset_ref(preset_id: str) -> dict[str, str]:
    """Return preset reference audio metadata."""

    if preset_id != "demo_ref":
        raise HTTPException(status_code=404, detail="Preset not found")
    return {
        "id": "demo_ref",
        "label": "示例参考",
        "filename": "ref.wav",
        "ref_text": "这是一段参考语音的转写。",
        "audio_b64": STUB_AUDIO_B64,
    }


@app.post("/api/v1/models/load", response_model=ModelLoadResponse)
async def load_model(model_id: str = Form(...)) -> ModelLoadResponse:
    """Load or switch the active model."""

    if not model_id.strip():
        raise HTTPException(status_code=400, detail="model_id is required")
    return ModelLoadResponse(status="already_loaded" if model_id == MODEL_ID else "loaded", model=model_id)


@app.post("/api/v1/tts/generate", response_model=TTSGenerateResponse)
async def generate_tts(
    text: str = Form(...),
    language: str = Form("Chinese"),
    mode: str = Form("custom"),
    speaker: Optional[str] = Form(None),
    instruct: str = Form(""),
    ref_audio: Optional[UploadFile] = File(None),
    ref_text: str = Form(""),
    ref_preset: str = Form(""),
    xvec_only: bool = Form(True),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
) -> TTSGenerateResponse:
    """Generate a complete WAV payload as Base64."""

    _validate_generate_request(
        text=text,
        mode=mode,
        speaker=speaker,
        ref_audio=ref_audio,
        ref_preset=ref_preset,
        ref_text=ref_text,
        xvec_only=xvec_only,
    )
    _ = (language, instruct, temperature, top_k, repetition_penalty)
    audio_duration_s = max(0.8, len(text) * 0.08)
    total_ms = max(300.0, audio_duration_s * 600)
    return TTSGenerateResponse(
        audio_b64=STUB_AUDIO_B64,
        sample_rate=SAMPLE_RATE,
        metrics=TTSMetrics(
            total_ms=total_ms,
            audio_duration_s=audio_duration_s,
            rtf=round(audio_duration_s / (total_ms / 1000), 3),
        ),
    )


@app.post("/api/v1/tts/stream")
async def stream_tts(
    text: str = Form(...),
    language: str = Form("Chinese"),
    mode: str = Form("custom"),
    speaker: Optional[str] = Form(None),
    instruct: str = Form(""),
    ref_audio: Optional[UploadFile] = File(None),
    ref_text: str = Form(""),
    ref_preset: str = Form(""),
    xvec_only: bool = Form(True),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
    chunk_size: int = Form(12),
) -> StreamingResponse:
    """Stream placeholder TTS events as Server-Sent Events."""

    _validate_generate_request(
        text=text,
        mode=mode,
        speaker=speaker,
        ref_audio=ref_audio,
        ref_preset=ref_preset,
        ref_text=ref_text,
        xvec_only=xvec_only,
    )
    _ = (language, instruct, temperature, top_k, repetition_penalty, chunk_size)

    async def event_source():
        chunk = {
            "type": "chunk",
            "audio_b64": STUB_AUDIO_B64,
            "sample_rate": SAMPLE_RATE,
            "ttfa_ms": 120,
            "voice_clone_ms": 0,
            "rtf": 1.6,
            "total_audio_s": 0.8,
            "elapsed_ms": 500,
        }
        done = {
            "type": "done",
            "ttfa_ms": 120,
            "voice_clone_ms": 0,
            "rtf": 1.6,
            "total_audio_s": 0.8,
            "total_ms": 500,
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/api/v1/tts/prewarm", response_model=PrewarmResponse)
async def prewarm_tts(
    ref_audio: Optional[UploadFile] = File(None),
    ref_preset: str = Form(""),
    ref_text: str = Form(""),
    xvec_only: bool = Form(True),
) -> PrewarmResponse:
    """Prewarm clone reference features."""

    _ = ref_text
    if ref_audio is None and not ref_preset:
        raise HTTPException(status_code=400, detail="Reference audio required.")
    return PrewarmResponse(status="warmed", xvec_only=xvec_only, ref_preset=ref_preset or None)


def _validate_generate_request(
    text: str,
    mode: str,
    speaker: Optional[str],
    ref_audio: Optional[UploadFile],
    ref_preset: str,
    ref_text: str,
    xvec_only: bool,
) -> None:
    """Validate documented TTS generation constraints."""

    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_CHARS} characters.",
        )
    if mode not in {"custom", "voice_clone", "voice_design"}:
        raise HTTPException(status_code=400, detail="mode must be one of: custom, voice_clone, voice_design")
    if mode == "custom" and not speaker:
        raise HTTPException(status_code=400, detail="Speaker ID is required for custom voice")
    if mode == "voice_clone" and ref_audio is None and not ref_preset:
        raise HTTPException(status_code=400, detail="Reference audio required.")
    if mode == "voice_clone" and not xvec_only and not ref_text:
        raise HTTPException(status_code=400, detail="Advanced (ICL) mode requires reference transcript.")
