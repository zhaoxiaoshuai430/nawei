"""Pydantic DTOs for client-facing and internal service protocols."""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field
try:  # Pydantic 2.x
    from pydantic import model_validator
except ImportError:  # pragma: no cover - Pydantic 1.x
    model_validator = None  # type: ignore[assignment]

try:  # Pydantic 1.x
    from pydantic import root_validator
except ImportError:  # pragma: no cover - Pydantic 2.x
    root_validator = None  # type: ignore[assignment]

from .base import Metadata, ProtocolModel, RequestId, SessionId, TraceId, metadata_field
from .enums import AudioFormat, MessageRole, ServiceName, VoiceMode


MAX_MESSAGE_CHARS = 20_000
MAX_AUDIO_URL_CHARS = 2_000
MAX_AUDIO_BASE64_CHARS = 36 * 1024 * 1024
MAX_TTS_TEXT_CHARS = 500


class ToolFunctionCall(ProtocolModel):
    """OpenAI-compatible function payload attached to a tool call."""

    name: str = Field(..., min_length=1)
    arguments: str = "{}"


class ToolCall(ProtocolModel):
    """Normalized LLM request to execute one registered tool."""

    id: str = Field(..., min_length=1)
    type: str = "function"
    function: ToolFunctionCall


class Message(ProtocolModel):
    """A normalized chat message."""

    role: MessageRole
    content: str = Field("", max_length=MAX_MESSAGE_CHARS)
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=metadata_field)

    if model_validator is not None:
        @model_validator(mode="after")
        def validate_message_body(self) -> "Message":
            if not self.content and not self.tool_calls:
                raise ValueError("Message requires content or tool_calls.")
            if self.role == "tool" and not self.tool_call_id:
                raise ValueError("Tool messages require tool_call_id.")
            return self
    elif root_validator is not None:  # pragma: no cover - Pydantic 1.x
        @root_validator
        def validate_message_body(cls, values: Metadata) -> Metadata:
            if not values.get("content") and not values.get("tool_calls"):
                raise ValueError("Message requires content or tool_calls.")
            if values.get("role") == "tool" and not values.get("tool_call_id"):
                raise ValueError("Tool messages require tool_call_id.")
            return values


class Usage(ProtocolModel):
    """Token usage returned by an LLM provider."""

    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)


class VoiceConfig(ProtocolModel):
    """Voice selection for TTS generation."""

    mode: VoiceMode = VoiceMode.CUSTOM
    speaker_id: Optional[str] = None
    language: str = "auto"
    instruct: Optional[str] = None


class AudioConfig(ProtocolModel):
    """Audio output configuration."""

    format: AudioFormat = AudioFormat.WAV
    sample_rate: int = Field(24000, gt=0, le=192000)
    channels: int = Field(1, ge=1, le=8)


class CreateSessionRequest(ProtocolModel):
    """Frontend request for creating a conversation session."""

    user_id: str = Field(..., min_length=1)
    title: Optional[str] = None
    metadata: Metadata = Field(default_factory=metadata_field)


class CreateSessionResponse(ProtocolModel):
    """Created session response."""

    session_id: str = SessionId
    created_at: str


class AudioTranscriptionRequest(ProtocolModel):
    """Frontend-facing request metadata for audio transcription."""

    request_id: str = RequestId
    trace_id: str = TraceId
    session_id: str = SessionId
    audio_url: Optional[str] = Field(None, max_length=MAX_AUDIO_URL_CHARS)
    audio_base64: Optional[str] = Field(None, max_length=MAX_AUDIO_BASE64_CHARS)
    language: str = "auto"
    return_timestamps: bool = False
    format: AudioFormat = AudioFormat.WAV
    sample_rate: int = Field(16000, gt=0)
    channels: int = Field(1, ge=1)
    metadata: Metadata = Field(default_factory=metadata_field)

    if model_validator is not None:
        @model_validator(mode="after")
        def validate_audio_source(self) -> "AudioTranscriptionRequest":
            _validate_single_audio_source(self.audio_url, self.audio_base64)
            return self
    elif root_validator is not None:  # pragma: no cover - Pydantic 1.x
        @root_validator
        def validate_audio_source(cls, values: Metadata) -> Metadata:
            _validate_single_audio_source(values.get("audio_url"), values.get("audio_base64"))
            return values


class ASRRequest(ProtocolModel):
    """Internal request for ASR transcription."""

    request_id: str = RequestId
    trace_id: str = TraceId
    session_id: str = SessionId
    audio_url: Optional[str] = Field(None, max_length=MAX_AUDIO_URL_CHARS)
    audio_base64: Optional[str] = Field(None, max_length=MAX_AUDIO_BASE64_CHARS)
    language: str = "auto"
    return_timestamps: bool = False
    format: AudioFormat = AudioFormat.WAV
    sample_rate: int = Field(16000, gt=0)
    channels: int = Field(1, ge=1)
    options: Metadata = Field(default_factory=metadata_field)

    if model_validator is not None:
        @model_validator(mode="after")
        def validate_audio_source(self) -> "ASRRequest":
            _validate_single_audio_source(self.audio_url, self.audio_base64)
            return self
    elif root_validator is not None:  # pragma: no cover - Pydantic 1.x
        @root_validator
        def validate_audio_source(cls, values: Metadata) -> Metadata:
            _validate_single_audio_source(values.get("audio_url"), values.get("audio_base64"))
            return values


class ASRSegment(ProtocolModel):
    """Timestamped ASR segment."""

    start_ms: int = Field(0, ge=0)
    end_ms: int = Field(0, ge=0)
    text: str


class ASREvent(ProtocolModel):
    """Non-speech event detected by an ASR provider."""

    tag: str
    label: Optional[str] = None
    emoji: Optional[str] = None


class ASRDetails(ProtocolModel):
    """Provider-specific ASR details normalized for platform responses."""

    key: Optional[str] = None
    emotion: Optional[str] = None
    emotion_label: Optional[str] = None
    emotion_emoji: Optional[str] = None
    events: List[ASREvent] = Field(default_factory=list)
    emotion_hint: Optional[str] = None
    raw_text: Optional[str] = None
    clean_text: Optional[str] = None
    corrections: List[dict[str, str]] = Field(default_factory=list)


class ASRResponse(ProtocolModel):
    """Internal ASR transcription response."""

    request_id: str = RequestId
    text: str
    language: str = "auto"
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    duration_ms: Optional[int] = Field(None, ge=0)
    segments: List[ASRSegment] = Field(default_factory=list)
    asr: ASRDetails = Field(default_factory=ASRDetails)
    metadata: Metadata = Field(default_factory=metadata_field)


class AudioTranscriptionResponse(ProtocolModel):
    """Frontend-facing transcription response."""

    request_id: str = RequestId
    trace_id: Optional[str] = None
    session_id: str = SessionId
    text: str
    language: str = "auto"
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    duration_ms: Optional[int] = Field(None, ge=0)
    segments: List[ASRSegment] = Field(default_factory=list)
    asr: ASRDetails = Field(default_factory=ASRDetails)
    metadata: Metadata = Field(default_factory=metadata_field)


class LLMRequest(ProtocolModel):
    """Internal request for LLM inference."""

    request_id: str = RequestId
    trace_id: str = TraceId
    session_id: str = SessionId
    user_id: str = Field("anonymous", min_length=1)
    messages: List[Message] = Field(..., min_length=1)
    stream: bool = True
    defer_persistence: bool = False
    voice_preview_token: Optional[str] = Field(default=None, max_length=128)
    tools: List[Metadata] = Field(default_factory=list)
    generation_config: Metadata = Field(default_factory=metadata_field)
    metadata: Metadata = Field(default_factory=metadata_field)


class LLMResponse(ProtocolModel):
    """Non-streaming LLM response."""

    request_id: str = RequestId
    text: str
    finish_reason: str = "stop"
    tool_calls: List[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    metadata: Metadata = Field(default_factory=metadata_field)


class TTSRequest(ProtocolModel):
    """Internal request for TTS generation."""

    request_id: str = RequestId
    trace_id: str = TraceId
    session_id: str = SessionId
    text: str = Field(..., min_length=1, max_length=MAX_TTS_TEXT_CHARS)
    stream: bool = True
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    audio_config: AudioConfig = Field(
        default_factory=lambda: AudioConfig(format=AudioFormat.WAV, sample_rate=24000, channels=1)
    )
    metadata: Metadata = Field(default_factory=metadata_field)


class TTSResponse(ProtocolModel):
    """Non-streaming TTS response."""

    request_id: str = RequestId
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    duration_ms: Optional[int] = Field(None, ge=0)
    format: AudioFormat = AudioFormat.WAV
    sample_rate: int = Field(24000, gt=0)
    rtf: Optional[float] = Field(None, ge=0.0)
    metadata: Metadata = Field(default_factory=metadata_field)

    if model_validator is not None:
        @model_validator(mode="after")
        def validate_audio_output(self) -> "TTSResponse":
            _validate_audio_output(self.audio_url, self.audio_base64)
            return self
    elif root_validator is not None:  # pragma: no cover - Pydantic 1.x
        @root_validator
        def validate_audio_output(cls, values: Metadata) -> Metadata:
            _validate_audio_output(values.get("audio_url"), values.get("audio_base64"))
            return values


class ChatCompletionRequest(ProtocolModel):
    """Frontend request for a direct Gateway-to-LLM conversation turn."""

    request_id: str = RequestId
    trace_id: str = TraceId
    session_id: str = SessionId
    user_id: str = Field("anonymous", min_length=1)
    message: Message
    stream: bool = True
    output_audio: bool = False
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    audio_config: AudioConfig = Field(
        default_factory=lambda: AudioConfig(format=AudioFormat.WAV, sample_rate=24000, channels=1)
    )
    metadata: Metadata = Field(default_factory=metadata_field)


class ChatCompletionResponse(ProtocolModel):
    """Frontend non-streaming chat completion response."""

    message_id: str
    session_id: str = SessionId
    text: str
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    duration_ms: Optional[int] = Field(None, ge=0)
    format: AudioFormat = AudioFormat.WAV
    sample_rate: int = Field(24000, gt=0)
    usage: Usage = Field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    metadata: Metadata = Field(default_factory=metadata_field)


class StreamChatRequest(ProtocolModel):
    """Query-state model for an SSE chat stream."""

    request_id: str = RequestId
    trace_id: str = TraceId
    session_id: str = SessionId
    message: Message
    output_audio: bool = False
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    audio_config: AudioConfig = Field(
        default_factory=lambda: AudioConfig(format=AudioFormat.WAV, sample_rate=24000, channels=1)
    )


class RealtimeAudioChunk(ProtocolModel):
    """WebSocket payload for a realtime audio chunk."""

    request_id: str = RequestId
    trace_id: str = TraceId
    session_id: str = SessionId
    format: AudioFormat = AudioFormat.PCM16
    sample_rate: int = Field(16000, gt=0)
    audio_base64: str = Field(..., min_length=1, max_length=MAX_AUDIO_BASE64_CHARS)
    metadata: Metadata = Field(default_factory=metadata_field)


class RealtimeAudioEnd(ProtocolModel):
    """WebSocket payload marking the end of realtime audio input."""

    request_id: str = RequestId
    trace_id: str = TraceId
    session_id: str = SessionId
    metadata: Metadata = Field(default_factory=metadata_field)


class ModelInvocation(ProtocolModel):
    """Audit record shape for model service calls."""

    request_id: str = RequestId
    trace_id: str = TraceId
    session_id: str = SessionId
    service: ServiceName
    model: Optional[str] = None
    latency_ms: Optional[int] = Field(None, ge=0)
    status: str = "success"
    error_code: Optional[str] = None
    metadata: Metadata = Field(default_factory=metadata_field)


def _validate_single_audio_source(audio_url: Optional[str], audio_base64: Optional[str]) -> None:
    """Require exactly one complete audio source for transcription requests."""

    has_url = bool(audio_url)
    has_base64 = bool(audio_base64)
    if has_url == has_base64:
        raise ValueError("Exactly one of audio_url or audio_base64 must be provided.")


def _validate_audio_output(audio_url: Optional[str], audio_base64: Optional[str]) -> None:
    if not audio_url and not audio_base64:
        raise ValueError("At least one of audio_url or audio_base64 must be provided.")
