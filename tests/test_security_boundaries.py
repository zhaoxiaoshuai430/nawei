"""Regression tests for request limits and service boundaries."""

from __future__ import annotations

import base64
import asyncio

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from services.asr.src.asr_protocol import decode_base64_audio, read_upload_limited
from services.gateway import main as gateway_main
from services.llm.memory import MemoryStore
from app_protocols import ASRRequest


class MemoryUpload:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.requested_size: int | None = None

    async def read(self, size: int = -1) -> bytes:
        self.requested_size = size
        return self.content[:size]


def test_base64_audio_limit_is_enforced_after_decode() -> None:
    payload = base64.b64encode(b"12345").decode("ascii")

    with pytest.raises(ValueError, match="upload limit"):
        decode_base64_audio(payload, max_bytes=4)


@pytest.mark.asyncio
async def test_upload_reader_stops_after_limit_probe() -> None:
    upload = MemoryUpload(b"12345")

    with pytest.raises(ValueError, match="upload limit"):
        await read_upload_limited(upload, max_bytes=4)

    assert upload.requested_size == 5


def test_public_status_hides_upstream_details(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_probe(client: object, name: str, url: str) -> dict[str, object]:
        return {
            "name": name,
            "ready": True,
            "latency_ms": 3,
            "detail": {"model": "private-model", "runtime": {"postgres": "ok"}},
        }

    monkeypatch.setattr(gateway_main, "_probe_dependency", fake_probe)
    with TestClient(gateway_main.app) as client:
        response = client.get("/api/v1/status")

    assert response.status_code == 200
    body = response.json()
    assert "concurrency" not in body
    assert body["services"]["llm"] == {"ready": True, "latency_ms": 3}


def test_detailed_status_requires_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_probe(client: object, name: str, url: str) -> dict[str, object]:
        return {"name": name, "ready": True, "latency_ms": 3, "detail": {"model": "internal"}}

    monkeypatch.setattr(gateway_main, "_probe_dependency", fake_probe)
    monkeypatch.setenv("GATEWAY_AUTH_MODE", "local")
    monkeypatch.setenv("NAWEI_ENV", "development")

    with TestClient(gateway_main.app) as client:
        unauthorized = client.get("/api/v1/status/details")
        authorized = client.get("/api/v1/status/details", headers={"X-User-ID": "operator-1"})

    assert unauthorized.status_code == 400
    assert authorized.status_code == 200
    assert authorized.json()["services"]["llm"]["detail"]["model"] == "internal"
    assert "concurrency" in authorized.json()


async def test_health_probe_bounds_total_wait_and_reports_timeout(monkeypatch):
    monkeypatch.setenv("GATEWAY_HEALTH_TIMEOUT_SECONDS", "0.02")
    cancelled = asyncio.Event()

    async def stalled(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled)) as client:
        result = await asyncio.wait_for(
            gateway_main._probe_dependency(client, "llm", "http://test/health"), timeout=1,
        )
    assert not result["ready"]
    assert result["error_type"] == "TimeoutError"
    assert cancelled.is_set()


async def test_health_probe_allows_slow_healthy_dependency(monkeypatch):
    monkeypatch.delenv("GATEWAY_HEALTH_TIMEOUT_SECONDS", raising=False)
    finished = asyncio.Event()

    async def respond(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            await asyncio.sleep(3.2)
            body = b'{"status":"ok"}'
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        async with httpx.AsyncClient(timeout=gateway_main.DEFAULT_HEALTH_TIMEOUT_SECONDS, trust_env=False) as client:
            result = await gateway_main._probe_dependency(client, "llm", f"http://127.0.0.1:{port}/health")
        assert result["ready"]
    finally:
        server.close()
        await server.wait_closed()
        await asyncio.wait_for(finished.wait(), timeout=5)


@pytest.mark.parametrize("body", [[], {"status": "degraded"}])
async def test_health_probe_rejects_invalid_or_unready_responses(body):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        result = await gateway_main._probe_dependency(client, "llm", "http://test/health")
    assert not result["ready"]
    if isinstance(body, list):
        assert result["error_type"] == "ValueError"


def test_legacy_asr_document_preserves_timing_and_confidence() -> None:
    request = ASRRequest(
        request_id="r",
        trace_id="t",
        session_id="s",
        audio_base64="abc",
    )

    response = gateway_main._asr_document_to_response(
        request,
        {
            "result": [
                {
                    "text": "hello",
                    "language": "en",
                    "confidence": 0.91,
                    "duration_ms": 420,
                    "segments": [{"start_ms": 0, "end_ms": 420, "text": "hello"}],
                    "metadata": {"decoder": "legacy"},
                    "corrections": [{"source": "helo", "replacement": "hello", "occurrences": "1"}],
                }
            ]
        },
    )

    assert response.confidence == 0.91
    assert response.duration_ms == 420
    assert response.segments[0].end_ms == 420
    assert response.metadata["decoder"] == "legacy"
    assert response.asr.corrections == [{"source": "helo", "replacement": "hello", "occurrences": "1"}]


@pytest.mark.parametrize("status_code", [400, 408, 409, 413, 415, 422, 429])
def test_gateway_preserves_safe_upstream_client_errors(status_code: int) -> None:
    request = httpx.Request("POST", "http://asr/api/v1/audio/transcriptions/base64")
    response = httpx.Response(status_code, request=request, json={"detail": "Invalid audio input."})
    upstream_error = httpx.HTTPStatusError("client error", request=request, response=response)

    with pytest.raises(HTTPException) as captured:
        gateway_main._raise_upstream_http_status("ASR", upstream_error)

    assert captured.value.status_code == status_code
    assert captured.value.detail == "Invalid audio input."


def test_gateway_hides_upstream_server_errors() -> None:
    request = httpx.Request("POST", "http://asr/api/v1/audio/transcriptions/base64")
    response = httpx.Response(500, request=request, json={"detail": "Internal model path leaked."})
    upstream_error = httpx.HTTPStatusError("server error", request=request, response=response)

    with pytest.raises(HTTPException) as captured:
        gateway_main._raise_upstream_http_status("ASR", upstream_error)

    assert captured.value.status_code == 502
    assert captured.value.detail == "ASR service request failed."


def test_memory_history_keys_encode_user_and_session_separately() -> None:
    first = MemoryStore._history_key("alice", "foo:bar")
    second = MemoryStore._history_key("alice:foo", "bar")

    assert first != second
    assert "alice" not in first
    assert "foo" not in first
