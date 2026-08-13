"""Standalone TTS HTTP API (/api/v1). No ASR or LLM."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import secrets
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Request
from pydantic import BaseModel, Field
try:  # Pydantic 2.x
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic 1.x
    ConfigDict = None  # type: ignore[assignment]

from app.config import list_available_models
from app.service_security import require_tts_internal_api_key
from app.services.tts import handlers as tts_handlers
from app.services.tts import state
from app.services.tts.capacity import CapacityRejected
from app.services.tts.engine import load_model_into_cache
from app.services.tts import emotion_refs
from app.services.tts.presets import preset_refs
from app.services.voice_lab.checkpoint import list_trained_speakers, voice_lab_status
from app.services.voice_lab.pipeline import list_drama_styles_for_chat

router = APIRouter(prefix="/api/v1", tags=["tts"])
TTS_API_DEPENDENCIES = [Depends(require_tts_internal_api_key)]

_VALID_MODES = frozenset({"voice_clone", "custom", "voice_design"})
_DEFAULT_STANDARD_SPEAKER = os.environ.get("TTS_DEFAULT_SPEAKER", "nana")
_DEFAULT_STANDARD_LANGUAGE = os.environ.get("TTS_DEFAULT_LANGUAGE", "Chinese")


class ProtocolModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid", use_enum_values=True)
    else:  # pragma: no cover - Pydantic 1.x
        class Config:
            extra = "forbid"
            use_enum_values = True


class VoiceConfig(ProtocolModel):
    mode: str = "custom"
    speaker_id: Optional[str] = None
    language: str = "auto"
    instruct: Optional[str] = None


class AudioConfig(ProtocolModel):
    format: Literal["wav", "pcm16", "mp3"] = "wav"
    sample_rate: int = Field(24000, gt=0, le=192000)
    channels: int = Field(1, ge=1, le=8)


class TTSRequest(ProtocolModel):
    request_id: str = Field(..., min_length=1)
    trace_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=500)
    stream: bool = True
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    audio_config: AudioConfig = Field(default_factory=AudioConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TTSResponse(ProtocolModel):
    request_id: str = Field(..., min_length=1)
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    duration_ms: Optional[int] = Field(None, ge=0)
    format: str = "wav"
    sample_rate: int = Field(24000, gt=0)
    rtf: Optional[float] = Field(None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _normalize_mode(mode: str) -> str:
    m = (mode or "custom").strip().lower()
    if m in _VALID_MODES:
        return m
    raise HTTPException(
        status_code=400,
        detail=f"mode must be one of: {', '.join(sorted(_VALID_MODES))}",
    )


def _metadata_bool(metadata: dict[str, Any], key: str, default: bool) -> bool:
    value = metadata.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _standard_language(language: str) -> str:
    if not language or language == "auto":
        return _DEFAULT_STANDARD_LANGUAGE
    return language


def _standard_duration_ms(metrics: dict[str, Any]) -> Optional[int]:
    duration_s = metrics.get("audio_duration_s")
    if isinstance(duration_s, (int, float)):
        return round(float(duration_s) * 1000)
    return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _authorize_model_load(request: Request, admin_key: Optional[str]) -> None:
    client_host = request.client.host if request.client is not None else ""
    try:
        is_loopback = ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        is_loopback = client_host.lower() == "localhost"
    if is_loopback:
        return

    configured_key = os.environ.get("TTS_ADMIN_API_KEY", "").strip()
    if configured_key and admin_key and secrets.compare_digest(admin_key, configured_key):
        return
    raise HTTPException(
        status_code=403,
        detail="Remote model loading requires a valid X-TTS-Admin-Key header.",
    )


def _tts_status_payload() -> dict:
    speakers: list[str] = []
    model_type = None
    active = state.model_cache.get(state.active_model_name) if state.active_model_name else None
    if active is not None:
        try:
            model_type = active.model.model.tts_model_type
            speakers = active.model.get_supported_speakers() or []
        except Exception:
            speakers = []
    vl = voice_lab_status()
    return {
        "service": "tts",
        "loaded": active is not None,
        "model": state.active_model_name,
        "loading": state.loading,
        "model_type": model_type,
        "speakers": speakers,
        "available_models": list_available_models(),
        "trained_voice_lab_speakers": list_trained_speakers(),
        "cached_models": list(state.model_cache.keys()),
        "concurrency": vars(state.generation_capacity.snapshot()),
        "preset_refs": [
            {"id": p["id"], "label": p["label"], "ref_text": p["ref_text"]}
            for p in preset_refs.values()
        ],
        "emotion_refs_enabled": emotion_refs.emotion_refs_enabled(),
        "drama_styles": list_drama_styles_for_chat(),
        "voice_lab": vl,
    }


@router.get("/health")
async def health():
    return {"status": "ok", "service": "tts"}


@router.get("/status")
async def status():
    return _tts_status_payload()


@router.post("/models/load")
async def load_model(
    request: Request,
    model_id: str = Form(...),
    x_tts_admin_key: Optional[str] = Header(None, alias="X-TTS-Admin-Key"),
):
    _authorize_model_load(request, x_tts_admin_key)
    model_id = (model_id or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    available_models = set(list_available_models())
    if model_id not in available_models:
        raise HTTPException(status_code=400, detail="model_id is not in available_models")
    if model_id in state.model_cache:
        state.active_model_name = model_id
        state.model_cache.move_to_end(model_id)
        return {"status": "already_loaded", "model": model_id}
    try:
        lease = await state.generation_capacity.acquire()
    except CapacityRejected as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.reason,
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
    async with lease:
        await asyncio.to_thread(load_model_into_cache, model_id)
    return {"status": "loaded", "model": model_id}


@router.get("/preset_ref/{preset_id}", dependencies=TTS_API_DEPENDENCIES)
async def get_preset_ref(preset_id: str):
    preset = preset_refs.get(preset_id)
    if not preset:
        preset = emotion_refs.emotion_refs.get(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {
        "id": preset["id"],
        "label": preset["label"],
        "filename": preset.get("filename", ""),
        "ref_text": preset["ref_text"],
        "audio_b64": preset.get("audio_b64"),
    }


@router.post("/tts", response_model=TTSResponse, dependencies=TTS_API_DEPENDENCIES)
@router.post("/audio/speech", response_model=TTSResponse, dependencies=TTS_API_DEPENDENCIES)
async def tts_standard_generate(request: TTSRequest) -> TTSResponse:
    metadata = request.metadata
    voice = request.voice
    mode = _normalize_mode(str(voice.mode or "custom"))
    language = _standard_language(voice.language)
    speaker = voice.speaker_id or str(metadata.get("speaker") or _DEFAULT_STANDARD_SPEAKER)

    response = await tts_handlers.generate_non_streaming(
        text=request.text,
        language=language,
        mode=mode,
        ref_text=str(metadata.get("ref_text", "")),
        speaker=speaker,
        instruct=voice.instruct or str(metadata.get("instruct", "")),
        xvec_only=_metadata_bool(metadata, "xvec_only", True),
        temperature=float(metadata.get("temperature", 0.9)),
        top_k=int(metadata.get("top_k", 50)),
        repetition_penalty=float(metadata.get("repetition_penalty", 1.05)),
        ref_preset=str(metadata.get("ref_preset", "")),
        ref_audio=None,
        audio_format=request.audio_config.format,
        sample_rate=request.audio_config.sample_rate,
        channels=request.audio_config.channels,
    )
    body = json.loads(response.body.decode("utf-8"))
    metrics = body.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    audio_base64 = body.get("audio_b64")
    if not audio_base64:
        raise HTTPException(status_code=500, detail="TTS response missing audio_b64.")

    return TTSResponse(
        request_id=request.request_id,
        audio_base64=str(audio_base64),
        duration_ms=_standard_duration_ms(metrics),
        format=str(body.get("format") or request.audio_config.format),
        sample_rate=int(body.get("sample_rate") or request.audio_config.sample_rate),
        rtf=_optional_float(metrics.get("rtf")),
        metadata={
            "provider": "http-tts",
            "tts_mode": mode,
            "speaker": speaker,
            "language": language,
            "channels": int(body.get("channels") or request.audio_config.channels),
            "metrics": metrics,
        },
    )


@router.post("/tts/stream", dependencies=TTS_API_DEPENDENCIES)
@router.post("/audio/speech/stream", dependencies=TTS_API_DEPENDENCIES)
async def tts_standard_stream(request: TTSRequest):
    metadata = request.metadata
    voice = request.voice
    mode = _normalize_mode(str(voice.mode or "custom"))
    language = _standard_language(voice.language)
    speaker = voice.speaker_id or str(metadata.get("speaker") or _DEFAULT_STANDARD_SPEAKER)

    return await tts_handlers.generate_stream(
        text=request.text,
        language=language,
        mode=mode,
        ref_text=str(metadata.get("ref_text", "")),
        speaker=speaker,
        instruct=voice.instruct or str(metadata.get("instruct", "")),
        xvec_only=_metadata_bool(metadata, "xvec_only", True),
        chunk_size=int(metadata.get("chunk_size", 25)),
        temperature=float(metadata.get("temperature", 0.9)),
        top_k=int(metadata.get("top_k", 50)),
        repetition_penalty=float(metadata.get("repetition_penalty", 1.05)),
        ref_preset=str(metadata.get("ref_preset", "")),
        ref_audio=None,
        audio_format=request.audio_config.format,
        sample_rate=request.audio_config.sample_rate,
        channels=request.audio_config.channels,
    )
