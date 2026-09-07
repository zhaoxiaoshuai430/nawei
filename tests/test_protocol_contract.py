"""Contract tests for shared protocols and orchestration."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from app_protocols import (
    ASRDetails,
    ASRRequest,
    ASRResponse,
    ChatCompletionRequest,
    LLMRequest,
    LLMResponse,
    Message,
    StreamEvent,
    TTSRequest,
    TTSResponse,
    Usage,
)
from app_protocols.orchestrator import ConversationOrchestrator


class FakeASRProvider:
    async def transcribe(self, request: ASRRequest) -> ASRResponse:
        return ASRResponse(
            request_id=request.request_id,
            text="你好",
            language="zh",
            confidence=0.98,
            duration_ms=800,
            asr=ASRDetails(emotion="HAPPY", emotion_label="开心", clean_text="你好"),
        )

    async def stream_transcribe(self, request: ASRRequest) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


class FakeLLMProvider:
    async def complete(self, request: LLMRequest) -> LLMResponse:
        text = request.messages[-1].content
        return LLMResponse(
            request_id=request.request_id,
            text=f"回复：{text}",
            usage=Usage(input_tokens=len(text), output_tokens=len(text) + 3),
        )

    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


class FakeTTSProvider:
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        return TTSResponse(
            request_id=request.request_id,
            audio_base64="UklGRg==",
            duration_ms=1000,
            format=request.audio_config.format,
            sample_rate=request.audio_config.sample_rate,
            rtf=1.6,
        )

    async def stream_synthesize(self, request: TTSRequest) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


def test_asr_request_requires_exactly_one_audio_source() -> None:
    ASRRequest(request_id="r", trace_id="t", session_id="s", audio_base64="abc")

    with pytest.raises(ValidationError):
        ASRRequest(request_id="r", trace_id="t", session_id="s")

    with pytest.raises(ValidationError):
        ASRRequest(
            request_id="r",
            trace_id="t",
            session_id="s",
            audio_url="upload://a.wav",
            audio_base64="abc",
        )


@pytest.mark.asyncio
async def test_transcribe_maps_asr_details_to_frontend_response() -> None:
    orchestrator = ConversationOrchestrator(
        asr_provider=FakeASRProvider(),
        llm_provider=FakeLLMProvider(),
        tts_provider=FakeTTSProvider(),
    )

    response = await orchestrator.transcribe(
        ASRRequest(
            request_id="r",
            trace_id="t",
            session_id="s",
            audio_base64="abc",
        )
    )

    assert response.text == "你好"
    assert response.confidence == 0.98
    assert response.duration_ms == 800
    assert response.asr.emotion == "HAPPY"


@pytest.mark.asyncio
async def test_complete_can_attach_tts_audio() -> None:
    orchestrator = ConversationOrchestrator(
        asr_provider=FakeASRProvider(),
        llm_provider=FakeLLMProvider(),
        tts_provider=FakeTTSProvider(),
    )

    response = await orchestrator.complete(
        ChatCompletionRequest(
            request_id="r",
            trace_id="t",
            session_id="s",
            user_id="user-1",
            message=Message(role="user", content="你好"),
            output_audio=True,
        )
    )

    assert response.text == "回复：你好"
    assert response.audio_base64 == "UklGRg=="
    assert response.duration_ms == 1000
    assert response.usage.input_tokens == 2
