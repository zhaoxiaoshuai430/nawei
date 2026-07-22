"""Protocol enums shared by DTOs and provider contracts."""

from __future__ import annotations

from enum import Enum


class MessageRole(str, Enum):
    """Allowed chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class VoiceMode(str, Enum):
    """Supported TTS voice selection modes."""

    CUSTOM = "custom"
    CLONE = "voice_clone"
    DESIGN = "voice_design"


class AudioFormat(str, Enum):
    """Audio formats exposed by platform protocols."""

    WAV = "wav"
    PCM16 = "pcm16"
    MP3 = "mp3"


class ServiceName(str, Enum):
    """Internal AI service names for tracing and invocation logs."""

    ASR = "asr"
    LLM = "llm"
    TTS = "tts"
    ORCHESTRATOR = "orchestrator"
    GATEWAY = "gateway"


class EventType(str, Enum):
    """Canonical streaming event names."""

    AUDIO_CHUNK = "audio.chunk"
    AUDIO_END = "audio.end"
    ASR_STARTED = "asr.started"
    ASR_DELTA = "asr.delta"
    ASR_COMPLETED = "asr.completed"
    MESSAGE_STARTED = "message.started"
    MESSAGE_DELTA = "message.delta"
    MESSAGE_DONE = "message.done"
    AUDIO_STARTED = "audio.started"
    AUDIO_DELTA = "audio.delta"
    AUDIO_DONE = "audio.done"
    ERROR = "error"


class ErrorCode(str, Enum):
    """Unified platform error codes."""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    AUDIO_TOO_LARGE = "AUDIO_TOO_LARGE"
    UNSUPPORTED_AUDIO_FORMAT = "UNSUPPORTED_AUDIO_FORMAT"
    ASR_TIMEOUT = "ASR_TIMEOUT"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    TTS_TIMEOUT = "TTS_TIMEOUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
