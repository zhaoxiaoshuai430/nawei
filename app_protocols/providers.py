"""Provider contracts for ASR, LLM, and TTS implementations."""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from .events import StreamEvent
from .schemas import ASRRequest, ASRResponse, LLMRequest, LLMResponse, TTSRequest, TTSResponse


class ASRProvider(Protocol):
    """Contract for speech-to-text providers."""

    async def transcribe(self, request: ASRRequest) -> ASRResponse:
        """Transcribe a complete audio input."""

    def stream_transcribe(self, request: ASRRequest) -> AsyncIterator[StreamEvent]:
        """Transcribe audio as a stream of ASR events."""


class LLMProvider(Protocol):
    """Contract for LLM providers."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a non-streaming LLM response."""

    def stream_complete(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Return text deltas as stream events."""


class TTSProvider(Protocol):
    """Contract for text-to-speech providers."""

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Return a non-streaming TTS result."""

    def stream_synthesize(self, request: TTSRequest) -> AsyncIterator[StreamEvent]:
        """Return audio deltas as stream events."""
