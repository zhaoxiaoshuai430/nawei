"""External keyword-filter integration tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from app_protocols import EventType, Message, StreamEvent, TTSRequest
from services.gateway import main as gateway_main
from services.gateway.keyword_filter import (
    KeywordFilterDecision,
    KeywordFilterError,
    KeywordFilterSettings,
    check_keyword_filter,
)


def _settings(**overrides: object) -> KeywordFilterSettings:
    values = {
        "enabled": True,
        "url": "https://filter.example/check",
        "timeout_seconds": 1.0,
        "failure_mode": "closed",
        "fallback_text": "Please choose another topic.",
        "text_field": "input.content",
        "blocked_field": "data.hit",
        "allowed_field": "",
        "matched_keywords_field": "data.keywords",
        "reason_field": "data.reason",
        "api_key": "secret",
        "api_key_header": "X-API-Key",
        "api_key_prefix": "",
    }
    values.update(overrides)
    return KeywordFilterSettings(**values)


@pytest.mark.asyncio
async def test_keyword_filter_supports_configurable_request_and_response_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "secret"
        assert json.loads(request.content) == {"input": {"content": "blocked transcript"}}
        return httpx.Response(
            200,
            json={"data": {"hit": True, "keywords": ["term-a", "term-b"], "reason": "policy"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        decision = await check_keyword_filter(client, "blocked transcript", _settings())

    assert decision.blocked is True
    assert decision.matched_keywords == ("term-a", "term-b")
    assert decision.reason == "policy"


@pytest.mark.asyncio
async def test_keyword_filter_can_invert_an_allowed_field() -> None:
    settings = _settings(blocked_field="", allowed_field="result.safe")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"safe": False}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        decision = await check_keyword_filter(client, "blocked transcript", settings)

    assert decision.blocked is True


class FailingLLMProvider:
    async def stream_complete(self, request: object) -> AsyncIterator[StreamEvent]:
        raise AssertionError("LLM must not be called for a blocked ASR transcript")
        yield  # pragma: no cover


class RecordingTTSProvider:
    def __init__(self) -> None:
        self.requests: list[TTSRequest] = []

    async def stream_synthesize(self, request: TTSRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        yield StreamEvent(
            event=EventType.AUDIO_DELTA,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"audio_base64": "AA==", "format": "pcm16", "sample_rate": 24000, "channels": 1},
        )


@pytest.mark.asyncio
async def test_blocked_asr_transcript_bypasses_llm_and_uses_fallback_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_text = "Please choose another topic."
    tts_provider = RecordingTTSProvider()

    async def blocked(*args: object, **kwargs: object) -> KeywordFilterDecision:
        return KeywordFilterDecision(blocked=True, matched_keywords=("term-a",), reason="policy")

    monkeypatch.setenv("GATEWAY_KEYWORD_FILTER_FALLBACK_TEXT", fallback_text)
    monkeypatch.setattr(gateway_main, "_evaluate_asr_keyword_filter", blocked)
    monkeypatch.setattr(gateway_main.orchestrator, "llm_provider", FailingLLMProvider())
    monkeypatch.setattr(gateway_main.orchestrator, "tts_provider", tts_provider)
    request = gateway_main.RealtimeVoiceRequest(
        request_id="request-filtered",
        trace_id="trace-filtered",
        session_id="session-filtered",
        user_id="user-filtered",
    )

    events = [
        event
        async for event in gateway_main._stream_voice_response(
            request,
            Message(role="user", content="blocked transcript"),
        )
    ]

    assert [event.event for event in events] == [
        EventType.MESSAGE_DELTA,
        EventType.MESSAGE_DONE,
        EventType.AUDIO_STARTED,
        EventType.AUDIO_DELTA,
        EventType.AUDIO_DONE,
    ]
    assert events[0].data["text"] == fallback_text
    assert events[1].data["finish_reason"] == "content_filter"
    assert events[1].data["moderation"]["matched_keyword_count"] == 1
    assert [request.text for request in tts_provider.requests] == [fallback_text]


@pytest.mark.asyncio
@pytest.mark.parametrize(("failure_mode", "blocked"), [("closed", True), ("open", False)])
async def test_keyword_filter_failure_mode_controls_llm_bypass(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    blocked: bool,
) -> None:
    async def fail(*args: object, **kwargs: object) -> KeywordFilterDecision:
        raise KeywordFilterError("unavailable")

    monkeypatch.setattr(gateway_main, "check_keyword_filter", fail)
    decision = await gateway_main._evaluate_asr_keyword_filter(
        gateway_main.RealtimeVoiceRequest(
            request_id="request-failure",
            trace_id="trace-failure",
            session_id="session-failure",
            user_id="user-failure",
        ),
        "transcript",
        _settings(failure_mode=failure_mode),
    )

    assert decision.blocked is blocked
    assert decision.failed is True
