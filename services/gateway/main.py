"""Gateway/BFF composing ASR, LLM, and TTS through app_protocols."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, NoReturn, Optional, Sequence

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import Field
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

from app_protocols import (
    ASRDetails,
    ASRRequest,
    ASRResponse,
    AudioConfig,
    AudioFormat,
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
    sse_encode,
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
TEST_PAGE_PATH = Path(__file__).resolve().parent / "static" / "test.html"
TEST_PCM_WORKLET_PATH = Path(__file__).resolve().parent / "static" / "pcm_capture_worklet.js"
DIAGNOSTIC_PATHS = frozenset({"/test", "/docs", "/redoc", "/openapi.json"})
DIAGNOSTIC_PREFIXES = ("/test/",)
logger = logging.getLogger(__name__)


def _diagnostics_enabled() -> bool:
    environment = os.getenv("NAWEI_ENV", "development").strip().lower()
    if environment in {"production", "prod"}:
        return False
    raw = os.getenv("GATEWAY_ENABLE_DIAGNOSTICS")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


_initial_diagnostics_enabled = _diagnostics_enabled()
app = FastAPI(
    title="Navi Gateway Service",
    version="0.1.0",
    lifespan=_gateway_lifespan,
    docs_url="/docs" if _initial_diagnostics_enabled else None,
    redoc_url="/redoc" if _initial_diagnostics_enabled else None,
    openapi_url="/openapi.json" if _initial_diagnostics_enabled else None,
)


@app.middleware("http")
async def protect_diagnostic_routes(request: Request, call_next: Any) -> Response:
    diagnostic_path = request.url.path in DIAGNOSTIC_PATHS or request.url.path.startswith(DIAGNOSTIC_PREFIXES)
    if diagnostic_path and not _diagnostics_enabled():
        return Response(status_code=404)
    return await call_next(request)


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
    output_audio_config: AudioConfig = Field(
        default_factory=lambda: AudioConfig(format=AudioFormat.WAV, sample_rate=24000, channels=1)
    )
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
        default_factory=lambda: AudioConfig(format=AudioFormat.PCM16, sample_rate=16000, channels=1)
    )
    output_audio_config: AudioConfig = Field(
        default_factory=lambda: AudioConfig(format=AudioFormat.WAV, sample_rate=24000, channels=1)
    )
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
            _raise_upstream_http_status("ASR", exc)
        except httpx.HTTPError as exc:
            _raise_upstream_transport_error("ASR", exc)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            _raise_invalid_upstream_response("ASR", exc)

    def stream_transcribe(self, request: ASRRequest) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError("Streaming ASR is not implemented in the HTTP provider.")


_INLINE_REFERENCE_MARKER = re.compile(r"[\[【]\s*资料\s*\d+\s*[\]】]")
_INLINE_REFERENCE_OPEN_TO_CLOSE = {"[": "]", "【": "】"}
_INLINE_REFERENCE_MAX_CHARS = 20


def _strip_inline_reference_markers(text: str) -> str:
    return _INLINE_REFERENCE_MARKER.sub("", text)


def _could_be_inline_reference_marker(value: str) -> bool:
    if not value or value[0] not in _INLINE_REFERENCE_OPEN_TO_CLOSE:
        return False
    close = _INLINE_REFERENCE_OPEN_TO_CLOSE[value[0]]
    content = value[1:]
    if content.endswith(close):
        content = content[:-1]
    elif "]" in content or "】" in content:
        return False
    content = content.strip()
    if not content:
        return True
    if "资料".startswith(content):
        return True
    if not content.startswith("资料"):
        return False
    number = content[2:].strip()
    return not number or number.isdigit()


@dataclass
class _InlineReferenceMarkerFilter:
    pending: str = ""

    def feed(self, text: str, *, flush: bool = False) -> str:
        output: list[str] = []
        for character in text:
            if not self.pending:
                if character in _INLINE_REFERENCE_OPEN_TO_CLOSE:
                    self.pending = character
                else:
                    output.append(character)
                continue

            self.pending += character
            close = _INLINE_REFERENCE_OPEN_TO_CLOSE[self.pending[0]]
            if character == close:
                if not _INLINE_REFERENCE_MARKER.fullmatch(self.pending):
                    output.append(self.pending)
                self.pending = ""
                continue
            if (
                len(self.pending) >= _INLINE_REFERENCE_MAX_CHARS
                or not _could_be_inline_reference_marker(self.pending)
            ):
                output.append(self.pending)
                self.pending = ""

        if flush and self.pending:
            output.append(self.pending)
            self.pending = ""
        return "".join(output)


def _stream_event_with_text(event: StreamEvent, text: str) -> StreamEvent:
    payload = _model_to_dict(event)
    data = dict(payload.get("data") or {})
    data["text"] = text
    payload["data"] = data
    return StreamEvent(**payload)


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
                body = response.json()
                body["text"] = _strip_inline_reference_markers(str(body.get("text") or ""))
                return LLMResponse(**body)
        except httpx.HTTPStatusError as exc:
            _raise_upstream_http_status("LLM", exc)
        except httpx.HTTPError as exc:
            _raise_upstream_transport_error("LLM", exc)
        except (TypeError, ValueError) as exc:
            _raise_invalid_upstream_response("LLM", exc)

    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        payload = _model_to_dict(request)
        headers: dict[str, str] = {}
        internal_api_key = os.getenv("LLM_INTERNAL_API_KEY", "").strip()
        if internal_api_key:
            headers["X-Internal-API-Key"] = internal_api_key
        reference_filter = _InlineReferenceMarkerFilter()
        last_delta: StreamEvent | None = None
        try:
            async with _gateway_http_client(self.timeout_seconds) as client:
                request_kwargs: dict[str, Any] = {"json": payload}
                if headers:
                    request_kwargs["headers"] = headers
                async with client.stream("POST", self.stream_url, **request_kwargs) as response:
                    await _raise_for_status_with_body(response)
                    async for body in _iter_sse_json(response):
                        event = StreamEvent(**body)
                        if event.event == EventType.MESSAGE_DELTA:
                            last_delta = event
                            text = reference_filter.feed(str(event.data.get("text") or ""))
                            if text:
                                yield _stream_event_with_text(event, text)
                            continue
                        if event.event in {EventType.MESSAGE_DONE, EventType.ERROR}:
                            tail = reference_filter.feed("", flush=True)
                            if tail and last_delta is not None:
                                yield _stream_event_with_text(last_delta, tail)
                        yield event
                    tail = reference_filter.feed("", flush=True)
                    if tail and last_delta is not None:
                        yield _stream_event_with_text(last_delta, tail)
        except httpx.HTTPStatusError as exc:
            _raise_upstream_http_status("LLM stream", exc)
        except httpx.HTTPError as exc:
            _raise_upstream_transport_error("LLM stream", exc)
        except (TypeError, ValueError) as exc:
            _raise_invalid_upstream_response("LLM stream", exc)


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
            _raise_upstream_http_status("TTS", exc)
        except httpx.HTTPError as exc:
            _raise_upstream_transport_error("TTS", exc)
        except (TypeError, ValueError) as exc:
            _raise_invalid_upstream_response("TTS", exc)

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
                    await _raise_for_status_with_body(response)
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
                                    "ttfa_ms": body.get("ttfa_ms"),
                                    "voice_clone_ms": body.get("voice_clone_ms"),
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
            _raise_upstream_http_status("TTS stream", exc)
        except httpx.HTTPError as exc:
            _raise_upstream_transport_error("TTS stream", exc)
        except (TypeError, ValueError) as exc:
            _raise_invalid_upstream_response("TTS stream", exc)


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
gateway_realtime_handshakes = asyncio.BoundedSemaphore(
    _read_int_env("GATEWAY_MAX_PENDING_REALTIME_HANDSHAKES", 50)
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

    routes = {
        "service": "gateway",
        "health": "/health",
        "status": "/api/v1/status",
        "realtime_voice": "/api/v1/voice/realtime",
    }
    if _diagnostics_enabled():
        routes["docs"] = "/docs"
        routes["test"] = "/test"
    return routes


@app.get("/test", include_in_schema=False)
async def test_page() -> FileResponse:
    """Serve the local end-to-end audio test console."""

    if not _diagnostics_enabled():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(
        TEST_PAGE_PATH,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/test/pcm-capture-worklet.js", include_in_schema=False)
async def test_pcm_capture_worklet() -> FileResponse:
    """Serve the local realtime PCM AudioWorklet module."""

    if not _diagnostics_enabled():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(
        TEST_PCM_WORKLET_PATH,
        media_type="application/javascript; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


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
    """Return public dependency readiness without exposing internal metadata."""

    return await _dependency_status(include_details=False)


@app.get("/api/v1/status/details")
async def detailed_dependency_status(
    authenticated_user_id: str = Depends(authenticate_user),
) -> dict[str, object]:
    """Return authenticated dependency diagnostics."""

    return await _dependency_status(include_details=True)


async def _dependency_status(*, include_details: bool) -> dict[str, object]:
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
    services = {
        str(result["name"]): result if include_details else _public_dependency_result(result)
        for result in results
    }
    response: dict[str, object] = {
        "status": "ready" if all(result["ready"] for result in results) else "degraded",
        "services": services,
    }
    if include_details:
        response["concurrency"] = vars(gateway_concurrency.snapshot())
    return response


def _public_dependency_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "ready": bool(result.get("ready")),
        "latency_ms": result.get("latency_ms"),
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


@app.post("/api/v1/text/completions/stream")
async def stream_text(request: ChatCompletionRequest, http_request: Request) -> StreamingResponse:
    """Stream a text turn as canonical Server-Sent Events."""

    _authorize_request_identity(request.user_id, http_request)
    lease = await _acquire_gateway_capacity(request.user_id)
    stream_request = ChatCompletionRequest(
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        user_id=request.user_id,
        message=request.message,
        stream=True,
        output_audio=False,
        voice=request.voice,
        audio_config=request.audio_config,
        metadata=request.metadata,
    )

    async def event_source() -> AsyncIterator[str]:
        async with lease:
            try:
                async for event in orchestrator.stream(stream_request):
                    yield sse_encode(event)
            except HTTPException as exc:
                yield sse_encode(
                    StreamEvent(
                        event=EventType.ERROR,
                        request_id=request.request_id,
                        trace_id=request.trace_id,
                        session_id=request.session_id,
                        data={
                            "code": ErrorCode.INVALID_ARGUMENT
                            if exc.status_code < 500
                            else ErrorCode.MODEL_UNAVAILABLE,
                            "message": str(exc.detail),
                            "retryable": exc.status_code >= 500,
                        },
                    )
                )
            except Exception as exc:
                logger.exception("Streaming text completion failed", exc_info=exc)
                yield sse_encode(
                    StreamEvent(
                        event=EventType.ERROR,
                        request_id=request.request_id,
                        trace_id=request.trace_id,
                        session_id=request.session_id,
                        data={
                            "code": ErrorCode.INTERNAL_ERROR,
                            "message": "Streaming text completion failed.",
                            "retryable": True,
                        },
                    )
                )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
            logger.warning(
                "TTS stream upstream returned HTTP %s: %s",
                exc.response.status_code,
                _extract_upstream_error(exc.response),
            )
            yield b'data: {"type": "error", "message": "TTS streaming failed."}\n\n'
        except httpx.HTTPError as exc:
            logger.warning("TTS stream transport failed: %s", exc)
            yield b'data: {"type": "error", "message": "TTS streaming failed."}\n\n'

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
    handshake_acquired = False
    try:
        if gateway_realtime_handshakes.locked():
            await websocket.close(code=4429)
            return
        await gateway_realtime_handshakes.acquire()
        handshake_acquired = True
        try:
            start_payload = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=_read_float_env("GATEWAY_REALTIME_START_TIMEOUT_SECONDS", 10.0),
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=408, detail="Realtime start message timed out.") from exc
        start = _parse_realtime_voice_start(start_payload)
        authorization = f"Bearer {start.access_token}" if start.access_token else None
        authorize_request_user(start.user_id, authorization, start.user_id)
        request = _realtime_voice_request(start)
        lease = await _acquire_gateway_capacity(request.user_id)
        gateway_realtime_handshakes.release()
        handshake_acquired = False
        async with lease:
            lease = None
            segments = await _relay_realtime_asr(websocket, request)
            if not _continuous_realtime_turns(request):
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
        logger.exception("Realtime voice request failed", exc_info=exc)
        await _safe_websocket_json(
            websocket,
            {
                "event": EventType.ERROR,
                "data": {
                    "code": ErrorCode.INTERNAL_ERROR,
                    "message": "Realtime voice processing failed.",
                    "retryable": True,
                },
            },
        )
        await _safe_websocket_close(websocket, 1011)
    finally:
        if handshake_acquired:
            gateway_realtime_handshakes.release()
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
        input_audio_config=AudioConfig(format=AudioFormat.PCM16, sample_rate=16000, channels=1),
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
            asr_task = asyncio.create_task(_collect_realtime_asr(websocket, upstream, request))
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


async def _collect_realtime_asr(
    websocket: WebSocket,
    upstream: Any,
    request: RealtimeVoiceRequest,
) -> list[dict[str, Any]]:
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
            if _continuous_realtime_turns(request):
                turn_request = _realtime_turn_request(request, len(segments))
                await _complete_realtime_voice_turn(websocket, turn_request, [body])
        if event_type == "session.done":
            return segments
    raise HTTPException(status_code=502, detail="ASR realtime connection ended before session.done.")


def _continuous_realtime_turns(request: RealtimeVoiceRequest) -> bool:
    return bool(request.metadata.get("continuous_turns", False))


def _realtime_turn_request(request: RealtimeVoiceRequest, turn_number: int) -> RealtimeVoiceRequest:
    suffix = f"turn-{turn_number}"
    updates = {
        "request_id": f"{request.request_id}-{suffix}",
        "trace_id": f"{request.trace_id}-{suffix}",
        "metadata": {**request.metadata, "turn_number": turn_number},
    }
    if hasattr(request, "model_copy"):
        return request.model_copy(update=updates)  # type: ignore[attr-defined]
    return request.copy(update=updates)


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


@dataclass(frozen=True)
class _VoiceProducerDone:
    name: str


_TTS_REFERENCE_LINE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:#{1,6}\s*)?"
    r"(?:参考(?:资料|来源|文献)|(?:资料|来源)\s*\d+|sources?|references?|source\s*\d+)"
    r"\s*(?:[：:]|$)",
    re.IGNORECASE,
)
_TTS_REFERENCE_CANDIDATE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:#{1,6}\s*)?(?:[参资来sr]|$)",
    re.IGNORECASE,
)
_TTS_REFERENCE_PREFIX_MAX_CHARS = 24
_TTS_MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)]\((?:https?://|www\.)[^)\s]+\)", re.IGNORECASE)
_TTS_BARE_URL = re.compile(r"(?:https?://|www\.)[^\s，。！？；、]+", re.IGNORECASE)
_TTS_LINE_MARKUP = re.compile(r"(?m)^\s*(?:#{1,6}\s+|[-*•]\s+|\d+[.)、]\s+)")
_TTS_INLINE_MARKUP = re.compile(r"(?:\*\*|__|~~|`+)")
_VOICE_REFERENCE_METADATA: Metadata = {
    "include_references": True,
}


@dataclass
class _TtsReferenceTextFilter:
    pending_prefix: str = ""
    mode: str = "detect"

    def feed(self, text: str, *, flush: bool = False) -> str:
        output: list[str] = []
        for character in text:
            if self.mode == "skip":
                if character in "\r\n":
                    output.append(character)
                    self.mode = "detect"
                continue
            if self.mode == "pass":
                output.append(character)
                if character in "\r\n":
                    self.mode = "detect"
                continue

            if character in "\r\n":
                if not _TTS_REFERENCE_LINE.match(self.pending_prefix):
                    output.append(self.pending_prefix)
                output.append(character)
                self.pending_prefix = ""
                continue

            self.pending_prefix += character
            if _TTS_REFERENCE_LINE.match(self.pending_prefix):
                self.pending_prefix = ""
                self.mode = "skip"
                continue
            if (
                not _TTS_REFERENCE_CANDIDATE.match(self.pending_prefix)
                or len(self.pending_prefix) >= _TTS_REFERENCE_PREFIX_MAX_CHARS
            ):
                output.append(self.pending_prefix)
                self.pending_prefix = ""
                self.mode = "pass"

        if flush and self.mode == "detect" and self.pending_prefix:
            if not _TTS_REFERENCE_LINE.match(self.pending_prefix):
                output.append(self.pending_prefix)
            self.pending_prefix = ""
        return "".join(output)


def _normalize_tts_spoken_text(text: str) -> str:
    normalized = _TTS_MARKDOWN_LINK.sub(r"\1", text)
    normalized = _TTS_BARE_URL.sub("", normalized)
    normalized = _TTS_LINE_MARKUP.sub("", normalized)
    normalized = _TTS_INLINE_MARKUP.sub("", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"([\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", r"\1", normalized)
    normalized = re.sub(r"\s+([，。！？；、])", r"\1", normalized)
    normalized = re.sub(r"\s*\r?\n+\s*", "，", normalized)
    return normalized.strip(" ，")


async def _produce_voice_llm_events(
    request: RealtimeVoiceRequest,
    llm_request: LLMRequest,
    event_queue: asyncio.Queue[StreamEvent | Exception | _VoiceProducerDone],
    segment_queue: asyncio.Queue[str | None],
) -> None:
    tts_buffer = ""
    tts_segment_count = 0
    tts_reference_filter = _TtsReferenceTextFilter()
    message_done: Optional[StreamEvent] = None
    try:
        async for event in orchestrator.llm_provider.stream_complete(llm_request):
            if event.event == EventType.ERROR:
                raise HTTPException(status_code=502, detail=str(event.data.get("message") or "LLM stream failed."))
            if event.event == EventType.MESSAGE_DONE:
                message_done = event
                continue
            if event.event != EventType.MESSAGE_DELTA:
                continue

            text = str(event.data.get("text") or "")
            if not text:
                continue
            await event_queue.put(_gateway_event(request, EventType.MESSAGE_DELTA, {"text": text}))
            if request.output_audio:
                tts_buffer += tts_reference_filter.feed(text)
                segments, tts_buffer = _extract_tts_segments(
                    tts_buffer,
                    first=tts_segment_count == 0,
                )
                for segment in segments:
                    await segment_queue.put(segment)
                tts_segment_count += len(segments)

        if request.output_audio:
            tts_buffer += tts_reference_filter.feed("", flush=True)
            if tts_buffer.strip():
                segments, _ = _extract_tts_segments(tts_buffer, flush=True)
                for segment in segments:
                    await segment_queue.put(segment)
        await event_queue.put(
            message_done
            or _gateway_event(
                request,
                EventType.MESSAGE_DONE,
                {"message_id": request.request_id, "finish_reason": "stop", "usage": {}},
            )
        )
    except Exception as exc:
        await event_queue.put(exc)
    finally:
        await segment_queue.put(None)
        await event_queue.put(_VoiceProducerDone("llm"))


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


async def _produce_voice_tts_events(
    request: RealtimeVoiceRequest,
    segment_queue: asyncio.Queue[str | None],
    event_queue: asyncio.Queue[StreamEvent | Exception | _VoiceProducerDone],
) -> None:
    audio_started = False
    segment_index = 0
    try:
        while True:
            segment = await segment_queue.get()
            if segment is None:
                break
            segment = _normalize_tts_spoken_text(segment)
            if not segment:
                continue
            if not audio_started:
                audio_started = True
                await event_queue.put(_voice_audio_started_event(request))
            async for audio_event in _stream_tts_segment(request, segment, segment_index):
                await event_queue.put(audio_event)
            segment_index += 1
        if audio_started:
            await event_queue.put(
                _gateway_event(
                    request,
                    EventType.AUDIO_DONE,
                    {"segments": segment_index, "format": request.output_audio_config.format},
                )
            )
    except Exception as exc:
        await event_queue.put(exc)
    finally:
        await event_queue.put(_VoiceProducerDone("tts"))


async def _stream_voice_response(
    request: RealtimeVoiceRequest,
    current_message: Message,
) -> AsyncIterator[StreamEvent]:
    llm_metadata = {**request.metadata, **_VOICE_REFERENCE_METADATA}
    llm_request = LLMRequest(
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        user_id=request.user_id,
        messages=[current_message],
        stream=True,
        metadata=llm_metadata,
    )
    event_queue: asyncio.Queue[StreamEvent | Exception | _VoiceProducerDone] = asyncio.Queue(
        maxsize=_read_int_env("GATEWAY_VOICE_EVENT_QUEUE_SIZE", 64)
    )
    segment_queue: asyncio.Queue[str | None] = asyncio.Queue(
        maxsize=_read_int_env("GATEWAY_VOICE_TTS_SEGMENT_QUEUE_SIZE", 8)
    )
    producers = (
        asyncio.create_task(_produce_voice_llm_events(request, llm_request, event_queue, segment_queue)),
        asyncio.create_task(_produce_voice_tts_events(request, segment_queue, event_queue)),
    )
    completed: set[str] = set()
    try:
        while len(completed) < len(producers):
            item = await event_queue.get()
            if isinstance(item, _VoiceProducerDone):
                completed.add(item.name)
                continue
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        for producer in producers:
            if not producer.done():
                producer.cancel()
        await asyncio.gather(*producers, return_exceptions=True)


async def _stream_tts_segment(
    request: RealtimeVoiceRequest,
    text: str,
    segment_index: int,
) -> AsyncIterator[StreamEvent]:
    if orchestrator.tts_provider is None:
        raise HTTPException(status_code=503, detail="TTS provider is not configured.")
    tts_metadata = dict(request.metadata)
    tts_metadata.setdefault("chunk_size", _read_int_env("GATEWAY_TTS_STREAM_CHUNK_SIZE", 8))
    tts_request = TTSRequest(
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        text=text,
        stream=True,
        voice=request.voice,
        audio_config=request.output_audio_config,
        metadata=tts_metadata,
    )
    async for event in orchestrator.tts_provider.stream_synthesize(tts_request):
        if event.event != EventType.AUDIO_DELTA:
            continue
        data = dict(event.data)
        data["segment_index"] = segment_index
        data["text"] = text
        yield _gateway_event(request, EventType.AUDIO_DELTA, data)


_TTS_SENTENCE_BOUNDARY = re.compile(r"[。.！？!?；;\n]+")
_TTS_SOFT_BOUNDARY = re.compile(r"[，,、：:]\s*")
_TTS_SOFT_SEGMENT_MIN_CHARS = max(4, _read_int_env("GATEWAY_TTS_SOFT_SEGMENT_MIN_CHARS", 8))
_TTS_FIRST_SEGMENT_CHARS = max(4, _read_int_env("GATEWAY_TTS_FIRST_SEGMENT_CHARS", 6))
_TTS_EAGER_SEGMENT_CHARS = max(
    _TTS_SOFT_SEGMENT_MIN_CHARS + 1,
    _read_int_env("GATEWAY_TTS_EAGER_SEGMENT_CHARS", 48),
)
_TTS_SEGMENT_MAX_CHARS = 450


def _extract_tts_segments(
    buffer: str,
    *,
    flush: bool = False,
    first: bool = False,
) -> tuple[list[str], str]:
    segments: list[str] = []
    start = 0
    for match in _TTS_SENTENCE_BOUNDARY.finditer(buffer):
        segments.extend(_split_long_tts_text(buffer[start:match.end()]))
        start = match.end()
    remainder = buffer[start:]
    if segments:
        first = False

    while not flush:
        soft_segment_min_chars = (
            min(_TTS_SOFT_SEGMENT_MIN_CHARS, _TTS_FIRST_SEGMENT_CHARS)
            if first
            else _TTS_SOFT_SEGMENT_MIN_CHARS
        )
        soft_boundary = next(
            (match for match in _TTS_SOFT_BOUNDARY.finditer(remainder) if match.end() >= soft_segment_min_chars),
            None,
        )
        if soft_boundary is not None:
            segments.append(remainder[:soft_boundary.end()].strip())
            remainder = remainder[soft_boundary.end():]
            first = False
            continue
        eager_segment_chars = _TTS_FIRST_SEGMENT_CHARS if first else _TTS_EAGER_SEGMENT_CHARS
        if len(remainder) >= eager_segment_chars:
            segment, remainder = _take_tts_prefix(remainder, eager_segment_chars)
            segments.append(segment)
            first = False
            continue
        break
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


def _take_tts_prefix(text: str, limit: int = _TTS_SEGMENT_MAX_CHARS) -> tuple[str, str]:
    cut = text.rfind(" ", 0, limit + 1)
    if cut <= 0:
        cut = limit
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
        confidence=first.get("confidence"),
        duration_ms=first.get("duration_ms"),
        segments=first.get("segments") or [],
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
        metadata={"provider": "http-asr", **(first.get("metadata") or {})},
    )


def _extract_upstream_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except httpx.ResponseNotRead:
        return f"Upstream returned HTTP {response.status_code}"
    except ValueError:
        try:
            return response.text
        except httpx.ResponseNotRead:
            return f"Upstream returned HTTP {response.status_code}"
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


async def _raise_for_status_with_body(response: httpx.Response) -> None:
    if response.is_error:
        await response.aread()
    response.raise_for_status()


def _raise_upstream_http_status(service: str, exc: httpx.HTTPStatusError) -> NoReturn:
    status_code = exc.response.status_code
    detail = _extract_upstream_error(exc.response)
    logger.warning(
        "%s upstream returned HTTP %s: %s",
        service,
        status_code,
        detail,
    )
    if status_code in {400, 408, 409, 413, 415, 422, 429}:
        raise HTTPException(status_code=status_code, detail=detail) from exc
    raise HTTPException(status_code=502, detail=f"{service} service request failed.") from exc


def _raise_upstream_transport_error(service: str, exc: httpx.HTTPError) -> NoReturn:
    logger.warning("%s upstream transport failed: %s", service, exc)
    raise HTTPException(status_code=502, detail=f"{service} service request failed.") from exc


def _raise_invalid_upstream_response(service: str, exc: Exception) -> NoReturn:
    logger.warning("%s upstream returned an invalid response: %s", service, exc)
    raise HTTPException(status_code=502, detail=f"{service} service returned an invalid response.") from exc
