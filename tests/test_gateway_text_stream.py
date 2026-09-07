"""Gateway text streaming regression tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app_protocols import EventType, LLMRequest, StreamEvent
from services.gateway import main as gateway_main


class TextStreamingLLMProvider:
    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        for text in ("first ", "second"):
            yield StreamEvent(
                event=EventType.MESSAGE_DELTA,
                request_id=request.request_id,
                trace_id=request.trace_id,
                session_id=request.session_id,
                data={"text": text},
            )
        yield StreamEvent(
            event=EventType.MESSAGE_DONE,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"message_id": request.request_id, "finish_reason": "stop"},
        )


def _request_body() -> dict[str, object]:
    return {
        "request_id": "request-stream",
        "trace_id": "trace-stream",
        "session_id": "session-stream",
        "user_id": "user-stream",
        "message": {"role": "user", "content": "hello", "metadata": {}},
        "stream": True,
        "output_audio": False,
        "metadata": {"source": "test"},
    }


def test_text_completion_streams_canonical_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NAWEI_ENV", "development")
    monkeypatch.setenv("GATEWAY_AUTH_MODE", "local")
    monkeypatch.setattr(gateway_main.orchestrator, "llm_provider", TextStreamingLLMProvider())

    with TestClient(gateway_main.app) as client:
        response = client.post(
            "/api/v1/text/completions/stream",
            headers={"X-User-ID": "user-stream"},
            json=_request_body(),
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.index('"text":"first "') < response.text.index('"text":"second"')
    assert response.text.count("event: message.started") == 1
    assert response.text.count("event: message.delta") == 2
    assert response.text.count("event: message.done") == 1
    assert gateway_main.gateway_concurrency.snapshot().active == 0


def test_text_completion_stream_requires_matching_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NAWEI_ENV", "development")
    monkeypatch.setenv("GATEWAY_AUTH_MODE", "local")

    with TestClient(gateway_main.app) as client:
        response = client.post("/api/v1/text/completions/stream", json=_request_body())

    assert response.status_code == 400
