"""Streaming event DTOs for SSE and WebSocket payloads."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from .base import Metadata, ProtocolModel, RequestId, SessionId, TraceId, dump_json, metadata_field
from .enums import AudioFormat, ErrorCode, EventType
from .schemas import ASRDetails, Usage


class StreamEvent(ProtocolModel):
    """Generic stream event envelope."""

    event: EventType
    request_id: str = RequestId
    trace_id: str = TraceId
    session_id: str = SessionId
    data: Metadata = Field(default_factory=metadata_field)


class AudioDeltaData(ProtocolModel):
    """Payload for audio delta events."""

    format: AudioFormat = AudioFormat.WAV
    sample_rate: int = Field(24000, gt=0)
    audio_base64: str


class AudioDoneData(ProtocolModel):
    """Payload for audio completion events."""

    audio_url: Optional[str] = None
    duration_ms: Optional[int] = Field(None, ge=0)


class TextDeltaData(ProtocolModel):
    """Payload for text delta events."""

    text: str


class ASRDeltaData(ProtocolModel):
    """Payload for partial ASR transcript events."""

    text: str
    is_final: bool = False


class ASRCompletedData(ProtocolModel):
    """Payload for completed ASR transcript events."""

    text: str
    language: str = "auto"
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    asr: ASRDetails = Field(default_factory=ASRDetails)


class MessageDoneData(ProtocolModel):
    """Payload for message completion events."""

    message_id: str
    finish_reason: str = "stop"
    usage: Usage = Field(default_factory=Usage)


class ErrorEventData(ProtocolModel):
    """Payload for stream error events."""

    code: ErrorCode
    message: str
    retryable: bool = False


def sse_encode(event: StreamEvent) -> str:
    """Encode a stream event as a Server-Sent Events frame."""

    payload = dump_json(event)
    return f"event: {event.event}\ndata: {payload}\n\n"
