"""Regression tests for unread HTTP streaming error responses."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import HTTPException

from app_protocols import EventType, LLMRequest, Message
from services.gateway import main as gateway_main
from services.llm import main as llm_main


def _llm_request() -> LLMRequest:
    return LLMRequest(
        request_id="request-1",
        trace_id="trace-1",
        session_id="session-1",
        messages=[Message(role="user", content="test")],
    )


@pytest.mark.asyncio
async def test_llm_stream_reads_error_body_before_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, request=request, json={"error": {"message": "rate limited"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(llm_main, "_deepseek_client", lambda timeout_seconds: client)
        settings = llm_main.LLMSettings(
            provider=llm_main.PROVIDER_DEEPSEEK,
            base_url="https://llm.example/v1",
            model="test-model",
            fallback_enabled=False,
        )

        events = [event async for event in llm_main._stream_backend_with_fallback(_llm_request(), settings)]

    assert len(events) == 1
    assert events[0].event == EventType.ERROR
    assert events[0].data["message"] == "rate limited"


@pytest.mark.asyncio
async def test_gateway_llm_stream_preserves_read_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, request=request, json={"detail": "Too many requests."})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        @asynccontextmanager
        async def fake_client(timeout_seconds: float) -> AsyncIterator[httpx.AsyncClient]:
            yield client

        monkeypatch.setattr(gateway_main, "_gateway_http_client", fake_client)
        provider = gateway_main.HTTPLLMProvider(stream_url="https://llm.example/v1/stream")

        with pytest.raises(HTTPException) as captured:
            _ = [event async for event in provider.stream_complete(_llm_request())]

    assert captured.value.status_code == 429
    assert captured.value.detail == "Too many requests."


@pytest.mark.asyncio
async def test_partial_llm_stream_failure_does_not_append_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def partial_stream(request: LLMRequest, settings: llm_main.LLMSettings) -> AsyncIterator[llm_main.StreamEvent]:
        yield llm_main.StreamEvent(
            event=EventType.MESSAGE_DELTA,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"text": "partial"},
        )
        raise httpx.ReadError("connection lost")

    monkeypatch.setattr(llm_main, "stream_deepseek", partial_stream)
    settings = llm_main.LLMSettings(
        provider=llm_main.PROVIDER_DEEPSEEK,
        base_url="https://llm.example/v1",
        model="test-model",
        fallback_enabled=True,
    )

    events = [event async for event in llm_main._stream_backend_with_fallback(_llm_request(), settings)]

    assert [event.event for event in events] == [EventType.MESSAGE_DELTA, EventType.ERROR]
    assert events[0].data["text"] == "partial"
    assert "connection lost" in events[1].data["message"]
