"""Regression tests for unread HTTP streaming error responses."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import HTTPException

from app_protocols import EventType, LLMRequest, Message, StreamEvent, TTSRequest
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


class SSEByteStream(httpx.AsyncByteStream):
    def __init__(self, frames, *, hold_open=False):
        self.frames = frames
        self.hold_open = hold_open
        self.closed = False
        self.read_past_end = False

    async def __aiter__(self):
        for frame in self.frames:
            raw = frame if isinstance(frame, str) else json.dumps(frame)
            yield f"data: {raw}\n\n".encode()
        self.read_past_end = True
        if self.hold_open:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def mocked_gateway_stream(monkeypatch, stream):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=stream),
    )) as client:
        @asynccontextmanager
        async def fake_client(timeout):
            yield client
        monkeypatch.setattr(gateway_main, "_gateway_http_client", fake_client)
        yield


def llm_frame(kind, **data):
    return StreamEvent(event=kind, request_id="request-1", trace_id="trace-1",
                       session_id="session-1", data=data).model_dump(mode="json")


def provider_events(kind):
    if kind == "llm":
        return gateway_main.HTTPLLMProvider().stream_complete(_llm_request())
    request = TTSRequest(request_id="request-1", trace_id="trace-1", session_id="session-1", text="Hello.")
    return gateway_main.HTTPTTSProvider().stream_synthesize(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "frames"), [
    ("llm", []),
    ("llm", [llm_frame(EventType.MESSAGE_DELTA, text="Partial answer")]),
    ("tts", []),
    ("tts", [{"type": "chunk", "audio_b64": "AAAA"}]),
    ("tts", [{"type": "done"}]),
    ("tts", [{"type": "chunk", "audio_b64": ""}, {"type": "done"}]),
])
async def test_gateway_rejects_incomplete_or_empty_streams(monkeypatch, kind, frames):
    stream = SSEByteStream(frames)
    async with mocked_gateway_stream(monkeypatch, stream):
        with pytest.raises(HTTPException) as error:
            _ = [event async for event in provider_events(kind)]
    assert error.value.status_code == 502
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "frames", "expected"), [
    ("llm", [llm_frame(EventType.MESSAGE_DELTA, text="Answer"), llm_frame(EventType.MESSAGE_DONE)],
     [EventType.MESSAGE_DELTA, EventType.MESSAGE_DONE]),
    ("llm", [llm_frame(EventType.ERROR, message="failed")], [EventType.ERROR]),
    ("tts", [{"type": "chunk", "audio_b64": "AAAA"}, {"type": "done"}], [EventType.AUDIO_DELTA]),
])
async def test_gateway_terminal_event_closes_connection_without_waiting(monkeypatch, kind, frames, expected):
    stream = SSEByteStream(frames, hold_open=True)
    async with mocked_gateway_stream(monkeypatch, stream):
        async with asyncio.timeout(1):
            events = [event async for event in provider_events(kind)]
    assert [event.event for event in events] == expected
    assert stream.closed
    assert not stream.read_past_end


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["llm", "tts"])
async def test_gateway_cancelled_stream_releases_connection(monkeypatch, kind):
    stream = SSEByteStream([], hold_open=True)
    async with mocked_gateway_stream(monkeypatch, stream):
        async def consume():
            return [event async for event in provider_events(kind)]
        task = asyncio.create_task(consume())
        try:
            async with asyncio.timeout(1):
                while not stream.read_past_end:
                    await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stream.closed
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_model_done_marker_does_not_wait_for_connection_close(monkeypatch):
    stream = SSEByteStream([{"choices": [{"delta": {"content": "Answer"}}]}, "[DONE]"], hold_open=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=stream),
    )) as client:
        monkeypatch.setattr(llm_main, "_deepseek_client", lambda timeout: client)
        settings = llm_main.LLMSettings(base_url="https://llm.example/v1")
        async with asyncio.timeout(1):
            bodies = [body async for body in llm_main._iter_deepseek_stream_bodies({}, settings)]
    assert len(bodies) == 1
    assert stream.closed
    assert not stream.read_past_end


@pytest.mark.asyncio
async def test_truncated_tts_cancels_llm_without_audio_done(monkeypatch):
    llm_closed = asyncio.Event()

    async def llm_events(request):
        try:
            yield StreamEvent(**llm_frame(EventType.MESSAGE_DELTA, text="Hello."))
            await asyncio.Event().wait()
        finally:
            llm_closed.set()

    monkeypatch.setattr(gateway_main.orchestrator.llm_provider, "stream_complete", llm_events)
    monkeypatch.setattr(gateway_main.orchestrator, "tts_provider", gateway_main.HTTPTTSProvider())
    monkeypatch.setenv("GATEWAY_KEYWORD_FILTER_ENABLED", "false")
    stream = SSEByteStream([{"type": "chunk", "audio_b64": "AAAA"}])
    events = []
    request = gateway_main.RealtimeVoiceRequest(
        request_id="request-1", trace_id="trace-1", session_id="session-1", user_id="owner",
    )
    async with mocked_gateway_stream(monkeypatch, stream):
        async with asyncio.timeout(2):
            with pytest.raises(HTTPException) as error:
                async for event in gateway_main._stream_voice_response(request, Message(role="user", content="Hello")):
                    events.append(event)

    assert error.value.status_code == 502
    assert not any(event.event == EventType.AUDIO_DONE for event in events)
    assert llm_closed.is_set()
    assert stream.closed


@pytest.mark.asyncio
async def test_tts_provider_error_event_is_not_ignored(monkeypatch):
    async def failed_tts(request):
        yield StreamEvent(**llm_frame(EventType.ERROR, message="failed"))

    monkeypatch.setattr(gateway_main.orchestrator.tts_provider, "stream_synthesize", failed_tts)
    request = gateway_main.RealtimeVoiceRequest(
        request_id="request-1", trace_id="trace-1", session_id="session-1", user_id="owner",
    )
    with pytest.raises(HTTPException) as error:
        _ = [event async for event in gateway_main._stream_tts_segment(request, "Hello", 0)]
    assert error.value.status_code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize("native_tools", [False, True])
async def test_model_eof_after_partial_text_never_reports_success(monkeypatch, native_tools):
    stream = SSEByteStream([{"choices": [{"delta": {"content": "partial answer"}}]}])
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(llm_main, "_deepseek_client", lambda timeout: client)
        request = _llm_request()
        if native_tools:
            request = request.model_copy(update={"tools": [{"type": "function", "function": {"name": "calculator"}}]})
        settings = llm_main.LLMSettings(provider="deepseek", base_url="https://llm.example/v1", fallback_enabled=True)
        events = [event async for event in llm_main._stream_backend_with_fallback(request, settings)]

    assert [event.event for event in events] == [EventType.MESSAGE_DELTA, EventType.ERROR]
    assert len(calls) == 1
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["[DONE]", {"choices": [{"delta": {}, "finish_reason": "stop"}]}])
async def test_model_explicit_completion_allows_clean_eof(monkeypatch, terminal):
    stream = SSEByteStream([{"choices": [{"delta": {"content": "complete"}}]}, terminal])
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=stream),
    )) as client:
        monkeypatch.setattr(llm_main, "_deepseek_client", lambda timeout: client)
        settings = llm_main.LLMSettings(base_url="https://llm.example/v1")
        events = [event async for event in llm_main.stream_deepseek(_llm_request(), settings)]
    assert [event.event for event in events] == [EventType.MESSAGE_DELTA, EventType.MESSAGE_DONE]
    assert stream.closed
