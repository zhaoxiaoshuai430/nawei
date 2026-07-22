"""Gateway/BFF composing ASR, LLM, and TTS through app_protocols."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, AsyncIterator, Optional, Sequence

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import Field

try:  # Pydantic 2.x
    from pydantic import model_validator
except ImportError:  # pragma: no cover - Pydantic 1.x
    model_validator = None  # type: ignore[assignment]

try:  # Pydantic 1.x
    from pydantic import root_validator
except ImportError:  # pragma: no cover - Pydantic 2.x
    root_validator = None  # type: ignore[assignment]

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
    sse_encode,
)
from app_protocols.base import Metadata, ProtocolModel, metadata_field
from app_protocols.orchestrator import ConversationOrchestrator
from services.gateway_service.conversation_store import ConversationSession, InMemoryConversationStore


DEFAULT_LLM_URL = "http://127.0.0.1:8002/api/v1/llm"
DEFAULT_LLM_STREAM_URL = "http://127.0.0.1:8002/api/v1/llm/stream"
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_ASR_URL = "http://127.0.0.1:8001/api/v1/audio/transcriptions/base64"
DEFAULT_ASR_TIMEOUT_SECONDS = 300.0
DEFAULT_TTS_URL = "http://127.0.0.1:19000/api/v1/audio/speech"
DEFAULT_TTS_STREAM_URL = "http://127.0.0.1:19000/api/v1/audio/speech/stream"
DEFAULT_TTS_TIMEOUT_SECONDS = 300.0

app = FastAPI(title="Navi Gateway Service", version="0.1.0")


def _default_input_audio_config() -> AudioConfig:
    return AudioConfig(sample_rate=16000)


class VoiceCompletionRequest(ProtocolModel):
    """Frontend request for a full voice turn."""

    request_id: str = Field(..., min_length=1)
    trace_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    language: str = "auto"
    output_audio: bool = True
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    input_audio_config: AudioConfig = Field(default_factory=_default_input_audio_config)
    output_audio_config: AudioConfig = Field(default_factory=AudioConfig)
    audio_config: Optional[AudioConfig] = Field(
        None,
        description="Deprecated compatibility field applied to both input and output audio.",
    )
    metadata: Metadata = Field(default_factory=metadata_field)

    if model_validator is not None:
        @model_validator(mode="after")
        def validate_audio_source(self) -> "VoiceCompletionRequest":
            _validate_single_audio_source(self.audio_url, self.audio_base64)
            if self.audio_config is not None:
                split_fields = {"input_audio_config", "output_audio_config"}
                if split_fields.intersection(self.model_fields_set):
                    raise ValueError("audio_config cannot be combined with split audio configuration fields.")
                self.input_audio_config = self.audio_config
                self.output_audio_config = self.audio_config
            return self
    elif root_validator is not None:  # pragma: no cover - Pydantic 1.x
        @root_validator(pre=True)
        def validate_audio_source(cls, values: Metadata) -> Metadata:
            _validate_single_audio_source(values.get("audio_url"), values.get("audio_base64"))
            legacy = values.get("audio_config")
            if legacy is not None:
                if "input_audio_config" in values or "output_audio_config" in values:
                    raise ValueError("audio_config cannot be combined with split audio configuration fields.")
                values["input_audio_config"] = legacy
                values["output_audio_config"] = legacy
            return values


class VoiceCompletionResponse(ProtocolModel):
    """Unified full voice turn response."""

    request_id: str
    trace_id: str
    session_id: str
    transcription: AudioTranscriptionResponse
    completion: ChatCompletionResponse
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
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(self.url, json=payload)
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

    async def stream_transcribe(self, request: ASRRequest) -> AsyncIterator[StreamEvent]:
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
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(self.url, json=payload)
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
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                async with client.stream("POST", self.stream_url, json=payload) as response:
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
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(self.url, json=payload)
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
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                async with client.stream("POST", self.stream_url, json=payload) as response:
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

conversation_store = InMemoryConversationStore(
    max_sessions=int(os.getenv("GATEWAY_SESSION_MAX_COUNT", "1000")),
    max_messages_per_session=int(os.getenv("GATEWAY_SESSION_MAX_MESSAGES", "40")),
    ttl_seconds=_read_float_env("GATEWAY_SESSION_TTL_SECONDS", 3600.0),
)


async def _complete_with_session_history(request: ChatCompletionRequest) -> ChatCompletionResponse:
    async with conversation_store.session(request.session_id) as conversation:
        response = await orchestrator.complete(request, history=conversation.history)
        if response.text.strip():
            conversation.append_turn(
                request.message,
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.text,
                    metadata={"finish_reason": "stop"},
                ),
            )
        return response


def _messages_with_history(history: Sequence[Message], current: Message) -> list[Message]:
    messages = list(history)
    if not any(message.role == MessageRole.SYSTEM for message in messages):
        messages.insert(0, Message(role=MessageRole.SYSTEM, content="You are a helpful voice assistant."))
    messages.append(current)
    return messages


@app.get("/health")
async def health() -> dict[str, object]:
    """Return Gateway health and dependency mode."""

    return {
        "status": "ok",
        "services": {
            "asr": "http",
            "llm": "http",
            "tts": "http",
        },
    }


@app.post("/api/v1/audio/transcriptions/base64", response_model=AudioTranscriptionResponse)
async def transcribe_base64(request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
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
async def complete_text(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Run an LLM-only text completion through the unified protocol."""

    return await _complete_with_session_history(
        ChatCompletionRequest(
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            message=request.message,
            stream=False,
            output_audio=False,
            voice=request.voice,
            audio_config=request.audio_config,
            metadata=request.metadata,
        )
    )


@app.post("/api/v1/text/speech-completions", response_model=ChatCompletionResponse)
async def complete_text_with_speech(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Run LLM then synthesize the LLM reply through TTS."""

    return await _complete_with_session_history(
        ChatCompletionRequest(
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            message=request.message,
            stream=False,
            output_audio=True,
            voice=request.voice,
            audio_config=request.audio_config,
            metadata=request.metadata,
        )
    )


@app.post("/api/v1/audio/speech", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest) -> TTSResponse:
    """Run a TTS-only speech synthesis through the unified protocol."""

    if orchestrator.tts_provider is None:
        raise HTTPException(status_code=503, detail="TTS provider is not configured.")
    return await orchestrator.tts_provider.synthesize(request)


@app.post("/api/v1/audio/speech/stream")
async def stream_speech(request: TTSRequest) -> StreamingResponse:
    """Proxy streaming TTS SSE events from the standalone TTS service."""

    url = os.getenv("GATEWAY_TTS_STREAM_URL", DEFAULT_TTS_STREAM_URL)
    payload = _model_to_dict(request)

    async def event_source() -> AsyncIterator[bytes]:
        try:
            timeout_seconds = _read_float_env(
                "GATEWAY_TTS_TIMEOUT_SECONDS",
                DEFAULT_TTS_TIMEOUT_SECONDS,
            )
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                async with client.stream("POST", url, json=payload) as response:
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


@app.post("/api/v1/voice/completions", response_model=VoiceCompletionResponse)
async def complete_voice(request: VoiceCompletionRequest) -> VoiceCompletionResponse:
    """Run ASR, LLM, and optional TTS as one voice conversation turn."""

    transcription = await orchestrator.transcribe(
        ASRRequest(
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            audio_url=request.audio_url,
            audio_base64=request.audio_base64,
            language=request.language,
            format=request.input_audio_config.format,
            sample_rate=request.input_audio_config.sample_rate,
            channels=request.input_audio_config.channels,
            options=request.metadata,
        )
    )
    completion = await _complete_with_session_history(
        ChatCompletionRequest(
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            message=Message(
                role=MessageRole.USER,
                content=transcription.text,
                metadata={"asr": _model_to_dict(transcription.asr)},
            ),
            stream=False,
            output_audio=request.output_audio,
            voice=request.voice,
            audio_config=request.output_audio_config,
            metadata=request.metadata,
        )
    )
    return VoiceCompletionResponse(
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        transcription=transcription,
        completion=completion,
        metadata={"path": "asr+llm+tts" if request.output_audio else "asr+llm"},
    )


@app.post("/api/v1/voice/completions/stream")
async def stream_voice_completion(request: VoiceCompletionRequest) -> StreamingResponse:
    """Run ASR, streaming LLM, and sentence-buffered streaming TTS over SSE."""

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in _voice_completion_events(request):
                yield sse_encode(event)
        except asyncio.CancelledError:
            raise
        except HTTPException as exc:
            yield sse_encode(
                _gateway_event(
                    request,
                    EventType.ERROR,
                    {
                        "code": ErrorCode.INVALID_ARGUMENT if exc.status_code < 500 else ErrorCode.MODEL_UNAVAILABLE,
                        "message": str(exc.detail),
                        "retryable": exc.status_code >= 500,
                    },
                )
            )
        except Exception as exc:
            yield sse_encode(
                _gateway_event(
                    request,
                    EventType.ERROR,
                    {"code": ErrorCode.INTERNAL_ERROR, "message": str(exc), "retryable": True},
                )
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _voice_completion_events(request: VoiceCompletionRequest) -> AsyncIterator[StreamEvent]:
    yield _gateway_event(request, EventType.ASR_STARTED)
    transcription = await orchestrator.transcribe(
        ASRRequest(
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            audio_url=request.audio_url,
            audio_base64=request.audio_base64,
            language=request.language,
            format=request.input_audio_config.format,
            sample_rate=request.input_audio_config.sample_rate,
            channels=request.input_audio_config.channels,
            options=request.metadata,
        )
    )
    if not transcription.text.strip():
        raise HTTPException(status_code=422, detail="ASR returned an empty transcription.")

    yield _gateway_event(
        request,
        EventType.ASR_COMPLETED,
        {
            "text": transcription.text,
            "language": transcription.language,
            "confidence": transcription.confidence,
            "duration_ms": transcription.duration_ms,
            "asr": _model_to_dict(transcription.asr),
        },
    )
    yield _gateway_event(
        request,
        EventType.MESSAGE_STARTED,
        {"message_id": request.request_id},
    )

    current_message = Message(
        role=MessageRole.USER,
        content=transcription.text,
        metadata={"asr": _model_to_dict(transcription.asr)},
    )
    async with conversation_store.session(request.session_id) as conversation:
        async for event in _stream_voice_response(request, current_message, conversation):
            yield event


async def _stream_voice_response(
    request: VoiceCompletionRequest,
    current_message: Message,
    conversation: ConversationSession,
) -> AsyncIterator[StreamEvent]:
    llm_request = LLMRequest(
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        messages=_messages_with_history(conversation.history, current_message),
        stream=True,
    )

    event_queue: asyncio.Queue[StreamEvent | Exception | None] = asyncio.Queue()

    async def produce_llm_events() -> None:
        try:
            async for event in orchestrator.llm_provider.stream_complete(llm_request):
                await event_queue.put(event)
        except Exception as exc:
            await event_queue.put(exc)
        finally:
            await event_queue.put(None)

    producer = asyncio.create_task(produce_llm_events())
    tts_buffer = ""
    audio_started = False
    segment_index = 0
    message_done: Optional[StreamEvent] = None
    message_parts: list[str] = []
    try:
        while True:
            item = await event_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if item.event == EventType.ERROR:
                raise HTTPException(status_code=502, detail=str(item.data.get("message") or "LLM stream failed."))
            if item.event == EventType.MESSAGE_DELTA:
                text = str(item.data.get("text") or "")
                if not text:
                    continue
                message_parts.append(text)
                yield _gateway_event(request, EventType.MESSAGE_DELTA, {"text": text})
                if request.output_audio:
                    tts_buffer += text
                    segments, tts_buffer = _extract_tts_segments(tts_buffer)
                    for segment in segments:
                        if not audio_started:
                            audio_started = True
                            yield _gateway_event(
                                request,
                                EventType.AUDIO_STARTED,
                                {
                                    "format": request.output_audio_config.format,
                                    "sample_rate": request.output_audio_config.sample_rate,
                                    "channels": request.output_audio_config.channels,
                                },
                            )
                        async for audio_event in _stream_tts_segment(request, segment, segment_index):
                            yield audio_event
                        segment_index += 1
            elif item.event == EventType.MESSAGE_DONE:
                message_done = item

        if request.output_audio and tts_buffer.strip():
            segments, _ = _extract_tts_segments(tts_buffer, flush=True)
            for segment in segments:
                if not audio_started:
                    audio_started = True
                    yield _gateway_event(
                        request,
                        EventType.AUDIO_STARTED,
                        {
                            "format": request.output_audio_config.format,
                            "sample_rate": request.output_audio_config.sample_rate,
                            "channels": request.output_audio_config.channels,
                        },
                    )
                async for audio_event in _stream_tts_segment(request, segment, segment_index):
                    yield audio_event
                segment_index += 1

        assistant_text = "".join(message_parts).strip()
        if assistant_text:
            conversation.append_turn(
                current_message,
                Message(role=MessageRole.ASSISTANT, content=assistant_text),
            )

        yield message_done or _gateway_event(
            request,
            EventType.MESSAGE_DONE,
            {"message_id": request.request_id, "finish_reason": "stop", "usage": {}},
        )
        if audio_started:
            yield _gateway_event(
                request,
                EventType.AUDIO_DONE,
                {"segments": segment_index, "format": request.output_audio_config.format},
            )
    finally:
        if not producer.done():
            producer.cancel()
        try:
            await producer
        except asyncio.CancelledError:
            pass


async def _stream_tts_segment(
    request: VoiceCompletionRequest,
    text: str,
    segment_index: int,
) -> AsyncIterator[StreamEvent]:
    if orchestrator.tts_provider is None:
        raise HTTPException(status_code=503, detail="TTS provider is not configured.")
    tts_request = TTSRequest(
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        text=text,
        stream=True,
        voice=request.voice,
        audio_config=request.output_audio_config,
        metadata=request.metadata,
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
    request: VoiceCompletionRequest,
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


def _validate_single_audio_source(audio_url: Optional[str], audio_base64: Optional[str]) -> None:
    if bool(audio_url) == bool(audio_base64):
        raise ValueError("Exactly one of audio_url or audio_base64 must be provided.")


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
