"""Gateway/BFF composing ASR, LLM, and TTS through app_protocols."""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Sequence

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import Field
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

from app_protocols import (
    ASRDetails,
    ASRRequest,
    ASRResponse,
    AudioConfig,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorCode,
    EventType,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    StreamEvent,
    TTSRequest,
    TTSResponse,
    VoiceConfig,
)
from app_protocols.base import Metadata, ProtocolModel, metadata_field
from app_protocols.orchestrator import ConversationOrchestrator
from services.gateway.auth import authenticate_user, authorize_request_user
from services.gateway.concurrency import ConcurrencyLease, ConcurrencyRejected, KeyedConcurrencyLimiter


DEFAULT_LLM_URL = "http://127.0.0.1:8002/api/v1/llm"
DEFAULT_LLM_STREAM_URL = "http://127.0.0.1:8002/api/v1/llm/stream"
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_ASR_URL = "http://127.0.0.1:8001/api/v1/audio/transcriptions/base64"
DEFAULT_ASR_REALTIME_URL = "ws://127.0.0.1:8001/api/v1/asr/realtime"
DEFAULT_ASR_TIMEOUT_SECONDS = 300.0
DEFAULT_TTS_URL = "http://127.0.0.1:19000/api/v1/audio/speech"
DEFAULT_TTS_STREAM_URL = "http://127.0.0.1:19000/api/v1/audio/speech/stream"
DEFAULT_TTS_TIMEOUT_SECONDS = 300.0
DEFAULT_ASR_HEALTH_URL = "http://127.0.0.1:8001/health"
DEFAULT_LLM_HEALTH_URL = "http://127.0.0.1:8002/health"
DEFAULT_TTS_HEALTH_URL = "http://127.0.0.1:19000/api/v1/status"
DEFAULT_HEALTH_TIMEOUT_SECONDS = 3.0
@asynccontextmanager
async def _gateway_lifespan(application: FastAPI) -> AsyncIterator[None]:
    max_connections = int(os.getenv("GATEWAY_HTTP_MAX_CONNECTIONS", "50"))
    max_keepalive_connections = int(os.getenv("GATEWAY_HTTP_MAX_KEEPALIVE_CONNECTIONS", "20"))
    keepalive_expiry = _read_float_env("GATEWAY_HTTP_KEEPALIVE_EXPIRY_SECONDS", 30.0)
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=keepalive_expiry,
    )
    timeouts = {
        _read_float_env("GATEWAY_ASR_TIMEOUT_SECONDS", DEFAULT_ASR_TIMEOUT_SECONDS),
        _read_float_env("GATEWAY_LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS),
        _read_float_env("GATEWAY_TTS_TIMEOUT_SECONDS", DEFAULT_TTS_TIMEOUT_SECONDS),
        _read_float_env("GATEWAY_HEALTH_TIMEOUT_SECONDS", DEFAULT_HEALTH_TIMEOUT_SECONDS),
    }
    async with AsyncExitStack() as stack:
        clients = {
            timeout: await stack.enter_async_context(
                httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False)
            )
            for timeout in timeouts
        }
        application.state.http_clients = clients
        yield
        application.state.http_clients = {}


app = FastAPI(title="Navi Gateway Service", version="0.1.0", lifespan=_gateway_lifespan)


@asynccontextmanager
async def _gateway_http_client(timeout_seconds: float) -> AsyncIterator[httpx.AsyncClient]:
    clients = getattr(app.state, "http_clients", {})
    shared = clients.get(timeout_seconds)
    if shared is not None:
        yield shared
        return
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        yield client


class RealtimeVoiceStart(ProtocolModel):
    """Initial WebSocket message for realtime voice input."""

    type: str = "start"
    request_id: str = Field(..., min_length=1)
    trace_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    access_token: Optional[str] = None
    language: str = "auto"
    output_audio: bool = True
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    output_audio_config: AudioConfig = Field(default_factory=AudioConfig)
    metadata: Metadata = Field(default_factory=metadata_field)


class RealtimeVoiceRequest(ProtocolModel):
    """Internal context shared by the realtime ASR, LLM, and TTS stages."""

    request_id: str
    trace_id: str
    session_id: str
    user_id: str
    language: str = "auto"
    output_audio: bool = True
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    input_audio_config: AudioConfig = Field(
        default_factory=lambda: AudioConfig(format="pcm16", sample_rate=16000, channels=1)
    )
    output_audio_config: AudioConfig = Field(default_factory=AudioConfig)
    metadata: Metadata = Field(default_factory=metadata_field)


class HTTPASRProvider:
    """ASR provider that calls the standalone ASR service over HTTP."""

    def __init__(self, url: str = DEFAULT_ASR_URL, timeout_seconds: float = DEFAULT_ASR_TIMEOUT_SECONDS) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    async def transcribe(self, request: ASRRequest) -> ASRResponse:
        if not request.audio_base64:
            raise HTTPException(status_code=400, detail="ASR HTTP adapter currently requires audio_base64 input.")

        payload = _model_to_dict(request)
        headers: dict[str, str] = {}
        internal_api_key = os.getenv("ASR_INTERNAL_API_KEY", "").strip()
        if internal_api_key:
            headers["X-Internal-API-Key"] = internal_api_key
        try:
            async with _gateway_http_client(self.timeout_seconds) as client:
                request_kwargs: dict[str, Any] = {"json": payload}
                if headers:
                    request_kwargs["headers"] = headers
                response = await client.post(self.url, **request_kwargs)
                response.raise_for_status()
                body = response.json()
                if "result" in body:
                    return _asr_document_to_response(request, body)
                return ASRResponse(**body)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=_extract_upstream_error(exc.response)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"ASR service request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"Invalid ASR service response: {exc}") from exc

    def stream_transcribe(self, request: ASRRequest) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError("Streaming ASR is not implemented in the HTTP provider.")


class HTTPLLMProvider:
    """LLM provider that calls the standalone LLM service over HTTP."""

    def __init__(
        self,
        url: str = DEFAULT_LLM_URL,
        stream_url: str = DEFAULT_LLM_STREAM_URL,
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.url = url
        self.stream_url = stream_url
        self.timeout_seconds = timeout_seconds

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = _model_to_dict(request)
        headers: dict[str, str] = {}
        internal_api_key = os.getenv("LLM_INTERNAL_API_KEY", "").strip()
        if internal_api_key:
            headers["X-Internal-API-Key"] = internal_api_key
        try:
            async with _gateway_http_client(self.timeout_seconds) as client:
                request_kwargs: dict[str, Any] = {"json": payload}
                if headers:
                    request_kwargs["headers"] = headers
                response = await client.post(self.url, **request_kwargs)
                response.raise_for_status()
                return LLMResponse(**response.json())
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=_extract_upstream_error(exc.response)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"LLM service request failed: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"Invalid LLM service response: {exc}") from exc

    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        payload = _model_to_dict(request)
        headers: dict[str, str] = {}
        internal_api_key = os.getenv("LLM_INTERNAL_API_KEY", "").strip()
        if internal_api_key:
            headers["X-Internal-API-Key"] = internal_api_key
        try:
            async with _gateway_http_client(self.timeout_seconds) as client:
                request_kwargs: dict[str, Any] = {"json": payload}
                if headers:
                    request_kwargs["headers"] = headers
                async with client.stream("POST", self.stream_url, **request_kwargs) as response:
                    response.raise_for_status()
                    async for body in _iter_sse_json(response):
                        yield StreamEvent(**body)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=_extract_upstream_error(exc.response)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"LLM stream request failed: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"Invalid LLM stream response: {exc}") from exc


class HTTPTTSProvider:
    """TTS provider that calls the standalone TTS service over HTTP."""

    def __init__(
        self,
        url: str = DEFAULT_TTS_URL,
        stream_url: str = DEFAULT_TTS_STREAM_URL,
        timeout_seconds: float = DEFAULT_TTS_TIMEOUT_SECONDS,
    ) -> None:
        self.url = url
        self.stream_url = stream_url
        self.timeout_seconds = timeout_seconds

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        payload = _model_to_dict(request)
        headers: dict[str, str] = {}
        internal_api_key = os.getenv("TTS_INTERNAL_API_KEY", "").strip()
        if internal_api_key:
            headers["X-Internal-API-Key"] = internal_api_key
        try:
            async with _gateway_http_client(self.timeout_seconds) as client:
                request_kwargs: dict[str, Any] = {"json": payload}
                if headers:
                    request_kwargs["headers"] = headers
                response = await client.post(self.url, **request_kwargs)
                response.raise_for_status()
                return TTSResponse(**response.json())
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=_extract_upstream_error(exc.response)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"TTS service request failed: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"Invalid TTS service response: {exc}") from exc

    async def stream_synthesize(self, request: TTSRequest) -> AsyncIterator[StreamEvent]:
        payload = _model_to_dict(request)
        headers: dict[str, str] = {}
        internal_api_key = os.getenv("TTS_INTERNAL_API_KEY", "").strip()
        if internal_api_key:
            headers["X-Internal-API-Key"] = internal_api_key
        try:
            async with _gateway_http_client(self.timeout_seconds) as client:
                request_kwargs: dict[str, Any] = {"json": payload}
                if headers:
                    request_kwargs["headers"] = headers
                async with client.stream("POST", self.stream_url, **request_kwargs) as response:
                    response.raise_for_status()
                    async for body in _iter_sse_json(response):
                        event_type = body.get("type")
                        if event_type == "chunk":
                            yield StreamEvent(
                                event=EventType.AUDIO_DELTA,
                                request_id=request.request_id,
                                trace_id=request.trace_id,
                                session_id=request.session_id,
                                data={
                                    "audio_base64": body.get("audio_b64", ""),
                                    "format": body.get("format", request.audio_config.format),
                                    "sample_rate": body.get("sample_rate", request.audio_config.sample_rate),
                                    "channels": body.get("channels", request.audio_config.channels),
                                    "rtf": body.get("rtf"),
                                    "elapsed_ms": body.get("elapsed_ms"),
                                },
                            )
                        elif event_type == "error":
                            raise HTTPException(
                                status_code=502,
                                detail=str(body.get("message") or "TTS stream failed."),
                            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=_extract_upstream_error(exc.response)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"TTS stream request failed: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"Invalid TTS stream response: {exc}") from exc


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    value = int(raw) if raw else default
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value

orchestrator = ConversationOrchestrator(
    asr_provider=HTTPASRProvider(
        url=os.getenv("GATEWAY_ASR_URL", DEFAULT_ASR_URL),
        timeout_seconds=_read_float_env("GATEWAY_ASR_TIMEOUT_SECONDS", DEFAULT_ASR_TIMEOUT_SECONDS),
    ),
    llm_provider=HTTPLLMProvider(
        url=os.getenv("GATEWAY_LLM_URL", DEFAULT_LLM_URL),
        stream_url=os.getenv("GATEWAY_LLM_STREAM_URL", DEFAULT_LLM_STREAM_URL),
        timeout_seconds=_read_float_env("GATEWAY_LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS),
    ),
    tts_provider=HTTPTTSProvider(
        url=os.getenv("GATEWAY_TTS_URL", DEFAULT_TTS_URL),
        stream_url=os.getenv("GATEWAY_TTS_STREAM_URL", DEFAULT_TTS_STREAM_URL),
        timeout_seconds=_read_float_env("GATEWAY_TTS_TIMEOUT_SECONDS", DEFAULT_TTS_TIMEOUT_SECONDS),
    ),
)

gateway_concurrency = KeyedConcurrencyLimiter(
    global_limit=_read_int_env("GATEWAY_MAX_CONCURRENT_REQUESTS", 20),
    per_key_limit=_read_int_env("GATEWAY_MAX_CONCURRENT_REQUESTS_PER_USER", 2),
    queue_limit=_read_int_env("GATEWAY_MAX_QUEUED_REQUESTS", 100),
    per_key_queue_limit=_read_int_env("GATEWAY_MAX_QUEUED_REQUESTS_PER_USER", 4),
    wait_timeout_seconds=_read_float_env("GATEWAY_CONCURRENCY_WAIT_TIMEOUT_SECONDS", 10.0),
)


async def _acquire_gateway_capacity(user_id: str) -> ConcurrencyLease:
    try:
        return await gateway_concurrency.acquire(user_id)
    except ConcurrencyRejected as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.reason,
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc


async def _complete_with_session_history(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Delegate conversation context and persistence to the LLM service."""

    return await orchestrator.complete(request)


@app.get("/", include_in_schema=False)
async def service_root() -> dict[str, str]:
    """Describe the API-only Gateway entrypoint."""

    return {
        "service": "gateway",
        "health": "/health",
        "status": "/api/v1/status",
        "realtime_voice": "/api/v1/voice/realtime",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, object]:
    """Return Gateway health and dependency mode."""

    return {
        "status": "ok",
        "services": {
            "gateway": "self",
            "asr": "http",
            "llm": "http",
            "tts": "http",
        },
    }


@app.get("/api/v1/status")
async def dependency_status() -> dict[str, object]:
    """Probe the actual readiness of Gateway upstream services."""

    timeout = _read_float_env("GATEWAY_HEALTH_TIMEOUT_SECONDS", DEFAULT_HEALTH_TIMEOUT_SECONDS)
    targets = [
        ("asr", os.getenv("GATEWAY_ASR_HEALTH_URL", DEFAULT_ASR_HEALTH_URL)),
        ("llm", os.getenv("GATEWAY_LLM_HEALTH_URL", DEFAULT_LLM_HEALTH_URL)),
        ("tts", os.getenv("GATEWAY_TTS_HEALTH_URL", DEFAULT_TTS_HEALTH_URL)),
    ]
    async with _gateway_http_client(timeout) as client:
        results = await asyncio.gather(
            *(_probe_dependency(client, name, url) for name, url in targets)
        )
    return {
        "status": "ready" if all(result["ready"] for result in results) else "degraded",
        "services": {str(result["name"]): result for result in results},
        "concurrency": vars(gateway_concurrency.snapshot()),
    }


async def _probe_dependency(client: httpx.AsyncClient, name: str, url: str) -> dict[str, object]:
    started = asyncio.get_running_loop().time()
    try:
        response = await client.get(url)
        response.raise_for_status()
        body = response.json()
        ready = bool(body.get("loaded")) if name == "tts" else body.get("status") == "ok"
        return {
            "name": name,
            "ready": ready,
            "latency_ms": round((asyncio.get_running_loop().time() - started) * 1000),
            "detail": body,
        }
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        return {
            "name": name,
            "ready": False,
            "latency_ms": round((asyncio.get_running_loop().time() - started) * 1000),
            "error": str(exc),
        }


@app.post("/api/v1/audio/transcriptions/base64", response_model=AudioTranscriptionResponse)
async def transcribe_base64(
    request: AudioTranscriptionRequest,
    authenticated_user_id: str = Depends(authenticate_user),
) -> AudioTranscriptionResponse:
    """Run ASR-only transcription through the unified protocol."""

    return await orchestrator.transcribe(
        ASRRequest(
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            audio_url=request.audio_url,
            audio_base64=request.audio_base64,
            language=request.language,
            return_timestamps=request.return_timestamps,
            format=request.format,
            sample_rate=request.sample_rate,
            channels=request.channels,
            options=request.metadata,
        )
    )


@app.post("/api/v1/text/completions", response_model=ChatCompletionResponse)
async def complete_text(request: ChatCompletionRequest, http_request: Request) -> ChatCompletionResponse:
    """Run a text turn directly through the standalone LLM service."""

    _authorize_request_identity(request.user_id, http_request)
    lease = await _acquire_gateway_capacity(request.user_id)
    async with lease:
        return await _complete_with_session_history(
            ChatCompletionRequest(
                request_id=request.request_id,
                trace_id=request.trace_id,
                session_id=request.session_id,
                user_id=request.user_id,
                message=request.message,
                stream=False,
                output_audio=False,
                voice=request.voice,
                audio_config=request.audio_config,
                metadata=request.metadata,
            )
        )


@app.post("/api/v1/diagnostics/llm", response_model=ChatCompletionResponse)
async def diagnose_llm(request: ChatCompletionRequest, http_request: Request) -> ChatCompletionResponse:
    """Exercise the direct Gateway-to-LLM boundary for the diagnostics panel."""

    _authorize_request_identity(request.user_id, http_request)
    return await orchestrator.complete(
        ChatCompletionRequest(
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            user_id=request.user_id,
            message=request.message,
            stream=False,
            output_audio=False,
            voice=request.voice,
            audio_config=request.audio_config,
            metadata=request.metadata,
        )
    )


@app.post("/api/v1/text/speech-completions", response_model=ChatCompletionResponse)
async def complete_text_with_speech(request: ChatCompletionRequest, http_request: Request) -> ChatCompletionResponse:
    """Run LLM then synthesize the LLM reply through TTS."""

    _authorize_request_identity(request.user_id, http_request)
    lease = await _acquire_gateway_capacity(request.user_id)
    async with lease:
        return await _complete_with_session_history(
            ChatCompletionRequest(
                request_id=request.request_id,
                trace_id=request.trace_id,
                session_id=request.session_id,
                user_id=request.user_id,
                message=request.message,
                stream=False,
                output_audio=True,
                voice=request.voice,
                audio_config=request.audio_config,
                metadata=request.metadata,
            )
        )


@app.post("/api/v1/audio/speech", response_model=TTSResponse)
async def synthesize_speech(
    request: TTSRequest,
    authenticated_user_id: str = Depends(authenticate_user),
) -> TTSResponse:
    """Run a TTS-only speech synthesis through the unified protocol."""

    if orchestrator.tts_provider is None:
        raise HTTPException(status_code=503, detail="TTS provider is not configured.")
    return await orchestrator.tts_provider.synthesize(request)


@app.post("/api/v1/audio/speech/stream")
async def stream_speech(
    request: TTSRequest,
    authenticated_user_id: str = Depends(authenticate_user),
) -> StreamingResponse:
    """Proxy streaming TTS SSE events from the standalone TTS service."""

    url = os.getenv("GATEWAY_TTS_STREAM_URL", DEFAULT_TTS_STREAM_URL)
    payload = _model_to_dict(request)
    headers: dict[str, str] = {}
    internal_api_key = os.getenv("TTS_INTERNAL_API_KEY", "").strip()
    if internal_api_key:
        headers["X-Internal-API-Key"] = internal_api_key

    async def event_source() -> AsyncIterator[bytes]:
        try:
            timeout_seconds = _read_float_env(
                "GATEWAY_TTS_TIMEOUT_SECONDS",
                DEFAULT_TTS_TIMEOUT_SECONDS,
            )
            async with _gateway_http_client(timeout_seconds) as client:
                request_kwargs: dict[str, Any] = {"json": payload}
                if headers:
                    request_kwargs["headers"] = headers
                async with client.stream("POST", url, **request_kwargs) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except httpx.HTTPStatusError as exc:
            detail = _extract_upstream_error(exc.response)
            yield f"data: {json.dumps({'type': 'error', 'message': detail})}\n\n".encode("utf-8")
        except httpx.HTTPError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': f'TTS stream request failed: {exc}'})}\n\n".encode("utf-8")

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/api/v1/voice/realtime")
async def realtime_voice_completion(websocket: WebSocket) -> None:
    """Receive realtime PCM input and continue through the existing response pipeline."""

    await websocket.accept()
    lease: Optional[ConcurrencyLease] = None
    try:
        start_payload = await websocket.receive_json()
        start = _parse_realtime_voice_start(start_payload)
        authorization = f"Bearer {start.access_token}" if start.access_token else None
        authorize_request_user(start.user_id, authorization, start.user_id)
        request = _realtime_voice_request(start)
        lease = await _acquire_gateway_capacity(request.user_id)
        async with lease:
            lease = None
            segments = await _relay_realtime_asr(websocket, request)
            await _complete_realtime_voice_turn(websocket, request, segments)
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        return
    except HTTPException as exc:
        await _safe_websocket_json(
            websocket,
            {
                "event": EventType.ERROR,
                "data": {
                    "code": ErrorCode.INVALID_ARGUMENT if exc.status_code < 500 else ErrorCode.MODEL_UNAVAILABLE,
                    "message": str(exc.detail),
                    "retryable": exc.status_code >= 500,
                },
            },
        )
        await _safe_websocket_close(websocket, 4400 if exc.status_code < 500 else 1011)
    except Exception as exc:
        await _safe_websocket_json(
            websocket,
            {
                "event": EventType.ERROR,
                "data": {"code": ErrorCode.INTERNAL_ERROR, "message": str(exc), "retryable": True},
            },
        )
        await _safe_websocket_close(websocket, 1011)
    finally:
        if lease is not None:
            await lease.release()


def _parse_realtime_voice_start(payload: Any) -> RealtimeVoiceStart:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Realtime start message must be a JSON object.")
    try:
        start = (
            RealtimeVoiceStart.model_validate(payload)
            if hasattr(RealtimeVoiceStart, "model_validate")
            else RealtimeVoiceStart.parse_obj(payload)
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid realtime start message: {exc}") from exc
    if start.type != "start":
        raise HTTPException(status_code=400, detail="First realtime message must have type=start.")
    return start


def _realtime_voice_request(start: RealtimeVoiceStart) -> RealtimeVoiceRequest:
    return RealtimeVoiceRequest(
        request_id=start.request_id,
        trace_id=start.trace_id,
        session_id=start.session_id,
        user_id=start.user_id,
        language=start.language,
        output_audio=start.output_audio,
        voice=start.voice,
        input_audio_config=AudioConfig(format="pcm16", sample_rate=16000, channels=1),
        output_audio_config=start.output_audio_config,
        metadata=start.metadata,
    )


async def _relay_realtime_asr(websocket: WebSocket, request: RealtimeVoiceRequest) -> list[dict[str, Any]]:
    upstream_url = os.getenv("GATEWAY_ASR_REALTIME_URL", DEFAULT_ASR_REALTIME_URL)
    try:
        async with websocket_connect(
            upstream_url,
            open_timeout=_read_float_env("GATEWAY_ASR_REALTIME_OPEN_TIMEOUT_SECONDS", 10.0),
            max_size=_read_int_env("GATEWAY_ASR_REALTIME_MAX_MESSAGE_BYTES", 1024 * 1024),
        ) as upstream:
            await upstream.send(
                json.dumps(
                    {
                        "type": "start",
                        "request_id": request.request_id,
                        "trace_id": request.trace_id,
                        "session_id": request.session_id,
                        "language": request.language,
                        "sample_rate": request.input_audio_config.sample_rate,
                        "channels": request.input_audio_config.channels,
                        "use_itn": bool(request.metadata.get("use_itn", True)),
                        "internal_api_key": os.getenv("ASR_INTERNAL_API_KEY", "").strip(),
                    }
                )
            )
            client_task = asyncio.create_task(_forward_realtime_audio(websocket, upstream))
            asr_task = asyncio.create_task(_collect_realtime_asr(websocket, upstream))
            done, _ = await asyncio.wait({client_task, asr_task}, return_when=asyncio.FIRST_COMPLETED)
            if asr_task in done:
                if not client_task.done():
                    client_task.cancel()
                    await asyncio.gather(client_task, return_exceptions=True)
                return await asr_task

            client_status = await client_task
            if client_status == "disconnected":
                if not asr_task.done():
                    asr_task.cancel()
                    await asyncio.gather(asr_task, return_exceptions=True)
                raise WebSocketDisconnect()
            return await asr_task
    except WebSocketDisconnect:
        raise
    except ConnectionClosed as exc:
        raise HTTPException(status_code=502, detail=f"ASR realtime connection closed: {exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"ASR realtime connection failed: {exc}") from exc


async def _forward_realtime_audio(websocket: WebSocket, upstream: Any) -> str:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            await upstream.close()
            return "disconnected"
        pcm = message.get("bytes")
        if pcm is not None:
            await upstream.send(pcm)
            continue
        text = message.get("text")
        if text is None:
            continue
        try:
            command = json.loads(text)
        except json.JSONDecodeError:
            await _safe_websocket_json(
                websocket,
                {"event": EventType.ERROR, "data": {"code": ErrorCode.INVALID_ARGUMENT, "message": "Invalid command."}},
            )
            continue
        command_type = command.get("type")
        if command_type == "stop":
            await upstream.send(json.dumps({"type": "stop"}))
            return "stopped"
        if command_type == "cancel":
            await upstream.close()
            return "disconnected"
        if command_type == "ping":
            await upstream.send(json.dumps({"type": "ping"}))


async def _collect_realtime_asr(websocket: WebSocket, upstream: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    async for raw in upstream:
        if not isinstance(raw, str):
            continue
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise HTTPException(status_code=502, detail="ASR realtime event must be a JSON object.")
        event_type = str(body.get("type") or "")
        await _safe_websocket_json(websocket, {"event": event_type, "data": body})
        if event_type == "transcript.final" and str(body.get("text") or "").strip():
            segments.append(body)
        if event_type == "session.done":
            return segments
    raise HTTPException(status_code=502, detail="ASR realtime connection ended before session.done.")


async def _complete_realtime_voice_turn(
    websocket: WebSocket,
    request: RealtimeVoiceRequest,
    segments: Sequence[dict[str, Any]],
) -> None:
    transcript = _join_realtime_transcripts([str(segment.get("text") or "") for segment in segments])
    if not transcript:
        raise HTTPException(status_code=422, detail="ASR returned an empty realtime transcription.")
    last_segment = segments[-1]
    asr_details = {
        "emotion": last_segment.get("emotion"),
        "emotion_label": last_segment.get("emotion_label"),
        "emotion_emoji": last_segment.get("emotion_emoji"),
        "events": last_segment.get("events") or [],
        "emotion_hint": last_segment.get("emotion_hint"),
        "raw_text": last_segment.get("raw_text"),
        "clean_text": last_segment.get("clean_text") or transcript,
    }
    await _send_websocket_stream_event(
        websocket,
        _gateway_event(
            request,
            EventType.ASR_COMPLETED,
            {
                "text": transcript,
                "language": str(last_segment.get("language") or request.language),
                "duration_ms": sum(int(segment.get("duration_ms") or 0) for segment in segments),
                "asr": asr_details,
                "segments": list(segments),
            },
        ),
    )
    await _send_websocket_stream_event(
        websocket,
        _gateway_event(request, EventType.MESSAGE_STARTED, {"message_id": request.request_id}),
    )

    current_message = Message(role=MessageRole.USER, content=transcript, metadata={"asr": asr_details})
    async for event in _stream_voice_response(request, current_message):
        await _send_websocket_stream_event(websocket, event)


def _join_realtime_transcripts(parts: Sequence[str]) -> str:
    result = ""
    for raw in parts:
        part = raw.strip()
        if not part:
            continue
        if result and result[-1].isascii() and part[0].isascii() and result[-1].isalnum() and part[0].isalnum():
            result += " "
        result += part
    return result


async def _send_websocket_stream_event(websocket: WebSocket, event: StreamEvent) -> None:
    await websocket.send_json(_model_to_dict(event))


async def _safe_websocket_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await websocket.send_json(payload)
    except (RuntimeError, WebSocketDisconnect):
        return


async def _safe_websocket_close(websocket: WebSocket, code: int) -> None:
    try:
        await websocket.close(code=code)
    except RuntimeError:
        return


@dataclass
class _VoiceStreamState:
    tts_buffer: str = ""
    audio_started: bool = False
    segment_index: int = 0
    message_done: Optional[StreamEvent] = None
    message_parts: list[str] = field(default_factory=list)

    @property
    def assistant_text(self) -> str:
        return "".join(self.message_parts).strip()


async def _produce_voice_llm_events(
    llm_request: LLMRequest,
    event_queue: asyncio.Queue[StreamEvent | Exception | None],
) -> None:
    try:
        async for event in orchestrator.llm_provider.stream_complete(llm_request):
            await event_queue.put(event)
    except Exception as exc:
        await event_queue.put(exc)
    finally:
        await event_queue.put(None)


def _voice_audio_started_event(request: RealtimeVoiceRequest) -> StreamEvent:
    return _gateway_event(
        request,
        EventType.AUDIO_STARTED,
        {
            "format": request.output_audio_config.format,
            "sample_rate": request.output_audio_config.sample_rate,
            "channels": request.output_audio_config.channels,
        },
    )


async def _stream_voice_tts_segments(
    request: RealtimeVoiceRequest,
    segments: Sequence[str],
    state: _VoiceStreamState,
    *,
    voice: Optional[VoiceConfig] = None,
    metadata: Optional[Metadata] = None,
) -> AsyncIterator[StreamEvent]:
    for segment in segments:
        if not state.audio_started:
            state.audio_started = True
            yield _voice_audio_started_event(request)
        async for audio_event in _stream_tts_segment(
            request,
            segment,
            state.segment_index,
            voice=voice,
            metadata=metadata,
        ):
            yield audio_event
        state.segment_index += 1


async def _handle_voice_llm_event(
    request: RealtimeVoiceRequest,
    event: StreamEvent,
    state: _VoiceStreamState,
) -> AsyncIterator[StreamEvent]:
    if event.event == EventType.ERROR:
        raise HTTPException(status_code=502, detail=str(event.data.get("message") or "LLM stream failed."))
    if event.event == EventType.MESSAGE_DONE:
        state.message_done = event
        return
    if event.event != EventType.MESSAGE_DELTA:
        return

    text = str(event.data.get("text") or "")
    if not text:
        return
    state.message_parts.append(text)
    yield _gateway_event(request, EventType.MESSAGE_DELTA, {"text": text})
    if not request.output_audio:
        return

    state.tts_buffer += text
    segments, state.tts_buffer = _extract_tts_segments(state.tts_buffer)
    async for audio_event in _stream_voice_tts_segments(request, segments, state):
        yield audio_event


async def _stream_voice_response(
    request: RealtimeVoiceRequest,
    current_message: Message,
) -> AsyncIterator[StreamEvent]:
    llm_request = LLMRequest(
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        user_id=request.user_id,
        messages=[current_message],
        stream=True,
        metadata=request.metadata,
    )
    event_queue: asyncio.Queue[StreamEvent | Exception | None] = asyncio.Queue(
        maxsize=_read_int_env("GATEWAY_VOICE_EVENT_QUEUE_SIZE", 64)
    )
    producer = asyncio.create_task(_produce_voice_llm_events(llm_request, event_queue))
    state = _VoiceStreamState()
    try:
        while True:
            item = await event_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            async for event in _handle_voice_llm_event(request, item, state):
                yield event

        if request.output_audio and state.tts_buffer.strip():
            segments, state.tts_buffer = _extract_tts_segments(state.tts_buffer, flush=True)
            async for audio_event in _stream_voice_tts_segments(request, segments, state):
                yield audio_event

        yield state.message_done or _gateway_event(
            request,
            EventType.MESSAGE_DONE,
            {"message_id": request.request_id, "finish_reason": "stop", "usage": {}},
        )
        if state.audio_started:
            yield _gateway_event(
                request,
                EventType.AUDIO_DONE,
                {"segments": state.segment_index, "format": request.output_audio_config.format},
            )
    finally:
        if not producer.done():
            producer.cancel()
        try:
            await producer
        except asyncio.CancelledError:
            pass


async def _stream_tts_segment(
    request: RealtimeVoiceRequest,
    text: str,
    segment_index: int,
    *,
    voice: Optional[VoiceConfig] = None,
    metadata: Optional[Metadata] = None,
) -> AsyncIterator[StreamEvent]:
    if orchestrator.tts_provider is None:
        raise HTTPException(status_code=503, detail="TTS provider is not configured.")
    tts_request = TTSRequest(
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        text=text,
        stream=True,
        voice=voice or request.voice,
        audio_config=request.output_audio_config,
        metadata=metadata or request.metadata,
    )
    async for event in orchestrator.tts_provider.stream_synthesize(tts_request):
        if event.event != EventType.AUDIO_DELTA:
            continue
        data = dict(event.data)
        data["segment_index"] = segment_index
        data["text"] = text
        yield _gateway_event(request, EventType.AUDIO_DELTA, data)


_TTS_SENTENCE_BOUNDARY = re.compile(r"[。.！？!?；;\n]+")
_TTS_SEGMENT_MAX_CHARS = 450


def _extract_tts_segments(buffer: str, *, flush: bool = False) -> tuple[list[str], str]:
    segments: list[str] = []
    start = 0
    for match in _TTS_SENTENCE_BOUNDARY.finditer(buffer):
        segments.extend(_split_long_tts_text(buffer[start:match.end()]))
        start = match.end()
    remainder = buffer[start:]

    while not flush and len(remainder) > _TTS_SEGMENT_MAX_CHARS:
        segment, remainder = _take_tts_prefix(remainder)
        segments.append(segment)
    if flush and remainder.strip():
        segments.extend(_split_long_tts_text(remainder))
        remainder = ""
    return [segment for segment in segments if segment], remainder


def _split_long_tts_text(text: str) -> list[str]:
    remaining = text.strip()
    parts: list[str] = []
    while len(remaining) > _TTS_SEGMENT_MAX_CHARS:
        part, remaining = _take_tts_prefix(remaining)
        parts.append(part)
    if remaining:
        parts.append(remaining)
    return parts


def _take_tts_prefix(text: str) -> tuple[str, str]:
    cut = text.rfind(" ", 0, _TTS_SEGMENT_MAX_CHARS + 1)
    if cut <= 0:
        cut = _TTS_SEGMENT_MAX_CHARS
    return text[:cut].strip(), text[cut:].lstrip()


def _gateway_event(
    request: RealtimeVoiceRequest,
    event: EventType,
    data: Optional[dict[str, Any]] = None,
) -> StreamEvent:
    return StreamEvent(
        event=event,
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        data=data or {},
    )


async def _iter_sse_json(response: Any) -> AsyncIterator[dict[str, Any]]:
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("SSE data must be a JSON object.")
        yield body


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def _authorize_request_identity(user_id: str, request: Request) -> None:
    authorize_request_user(
        user_id,
        request.headers.get("Authorization"),
        request.headers.get("X-User-ID"),
    )


def _asr_document_to_response(request: ASRRequest, body: dict[str, Any]) -> ASRResponse:
    results = body["result"]
    if not results:
        raise ValueError("ASR response result is empty")

    first = results[0]
    text = str(first.get("clean_text") or first.get("text") or "")
    return ASRResponse(
        request_id=request.request_id,
        text=text,
        language=str(first.get("language") or request.language),
        asr=ASRDetails(
            key=first.get("key"),
            emotion=first.get("emotion"),
            emotion_label=first.get("emotion_label"),
            emotion_emoji=first.get("emotion_emoji"),
            events=first.get("events") or [],
            emotion_hint=first.get("emotion_hint"),
            raw_text=first.get("raw_text"),
            clean_text=first.get("clean_text") or text,
        ),
        metadata={"provider": "http-asr"},
    )


def _extract_upstream_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text
    if isinstance(body, dict):
        detail = body.get("detail")
        if detail:
            return str(detail)
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    return str(body)
