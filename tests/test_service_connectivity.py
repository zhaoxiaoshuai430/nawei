"""Connectivity tests for Gateway, ASR, LLM, and TTS service boundaries."""

from __future__ import annotations

import base64
import json
import os
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from services.gateway import main as gateway_main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT
RUN_LIVE_TESTS = os.getenv("NAWEI_RUN_LIVE_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


class FakeHTTPResponse:
    """Minimal JSON response used by the routed service client."""

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


class FakeSSEHTTPResponse(FakeHTTPResponse):
    """Minimal streaming response exposing SSE lines."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__({})
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeStreamContext:
    def __init__(self, response: FakeSSEHTTPResponse) -> None:
        self.response = response

    async def __aenter__(self) -> FakeSSEHTTPResponse:
        return self.response

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class RoutedServiceAsyncClient:
    """Route Gateway HTTP calls to deterministic ASR, LLM, and TTS responses."""

    calls: list[dict[str, Any]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "RoutedServiceAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def get(self, url: str) -> FakeHTTPResponse:
        self.calls.append({"method": "GET", "url": url, "timeout": self.timeout})
        if url.endswith(":19000/api/v1/status"):
            return FakeHTTPResponse({"service": "tts", "loaded": True, "model": "voice_lab/nana"})
        return FakeHTTPResponse({"status": "ok"})

    async def post(
        self,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> FakeHTTPResponse:
        self.calls.append(
            {"method": "POST", "url": url, "json": json, "headers": headers or {}, "timeout": self.timeout}
        )
        if url.endswith("/api/v1/audio/transcriptions/base64"):
            return FakeHTTPResponse(
                {
                    "request_id": json["request_id"],
                    "text": "模块互通测试",
                    "language": "zh",
                    "confidence": 0.98,
                    "duration_ms": 800,
                    "segments": [],
                    "asr": {
                        "key": json.get("options", {}).get("filename"),
                        "emotion": "NEUTRAL",
                        "emotion_label": "平静",
                        "clean_text": "模块互通测试",
                    },
                    "metadata": {"provider": "connectivity-asr"},
                }
            )
        if url.endswith("/api/v1/llm"):
            text = json["messages"][-1]["content"]
            return FakeHTTPResponse(
                {
                    "request_id": json["request_id"],
                    "text": f"LLM 已收到：{text}",
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 4, "output_tokens": 8},
                    "metadata": {"provider": "connectivity-llm"},
                }
            )
        if url.endswith("/api/v1/audio/speech"):
            return FakeHTTPResponse(
                {
                    "request_id": json["request_id"],
                    "audio_base64": "UklGRg==",
                    "duration_ms": 900,
                    "format": json["audio_config"]["format"],
                    "sample_rate": json["audio_config"]["sample_rate"],
                    "rtf": 0.8,
                    "metadata": {"provider": "connectivity-tts"},
                }
            )
        raise AssertionError(f"Unexpected POST URL: {url}")

    def stream(
        self,
        method: str,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> FakeStreamContext:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "json": json,
                "headers": headers or {},
                "timeout": self.timeout,
                "stream": True,
            }
        )
        if url.endswith("/api/v1/llm/stream"):
            envelope = {
                "request_id": json["request_id"],
                "trace_id": json["trace_id"],
                "session_id": json["session_id"],
            }
            lines = [
                _sse_data({**envelope, "event": "message.delta", "data": {"text": "流式互通测试。"}}),
                _sse_data(
                    {
                        **envelope,
                        "event": "message.done",
                        "data": {"message_id": json["request_id"], "finish_reason": "stop", "usage": {}},
                    }
                ),
            ]
            return FakeStreamContext(FakeSSEHTTPResponse(lines))
        if url.endswith("/api/v1/audio/speech/stream"):
            lines = [
                _sse_data(
                    {
                        "type": "chunk",
                        "audio_b64": "UklGRg==",
                        "format": "wav",
                        "sample_rate": 24000,
                        "channels": 1,
                    }
                ),
                _sse_data({"type": "done"}),
            ]
            return FakeStreamContext(FakeSSEHTTPResponse(lines))
        raise AssertionError(f"Unexpected stream URL: {url}")


def _sse_data(body: dict[str, Any]) -> str:
    return f"data: {json.dumps(body, ensure_ascii=False)}"


def _request_ids(prefix: str) -> dict[str, str]:
    suffix = uuid.uuid4().hex
    return {
        "request_id": f"{prefix}-{suffix}",
        "trace_id": f"trace-{suffix}",
        "session_id": f"session-{suffix}",
    }


class GatewayConnectivityContractTest(unittest.TestCase):
    """Verify every Gateway boundary without requiring model processes."""

    def setUp(self) -> None:
        RoutedServiceAsyncClient.calls.clear()
        self.client = TestClient(gateway_main.app)

    def test_gateway_health_probe_reaches_asr_llm_and_tts(self) -> None:
        with patch("services.gateway.main.httpx.AsyncClient", RoutedServiceAsyncClient):
            response = self.client.get("/api/v1/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        called_urls = {call["url"] for call in RoutedServiceAsyncClient.calls}
        self.assertEqual(
            called_urls,
            {
                gateway_main.DEFAULT_ASR_HEALTH_URL,
                gateway_main.DEFAULT_LLM_HEALTH_URL,
                gateway_main.DEFAULT_TTS_HEALTH_URL,
            },
        )

    def test_gateway_asr_route_reaches_asr_service(self) -> None:
        request = {
            **_request_ids("asr"),
            "audio_base64": "UklGRg==",
            "language": "zh",
            "format": "wav",
            "sample_rate": 16000,
            "channels": 1,
            "metadata": {"filename": "asr-connectivity.wav", "use_itn": True},
        }
        with patch("services.gateway.main.httpx.AsyncClient", RoutedServiceAsyncClient):
            response = self.client.post(
                "/api/v1/audio/transcriptions/base64",
                headers={"X-User-ID": "connectivity-user"},
                json=request,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "模块互通测试")
        call = _single_post_call()
        self.assertEqual(call["url"], gateway_main.DEFAULT_ASR_URL)
        self.assertEqual(call["json"]["options"]["filename"], "asr-connectivity.wav")

    def test_gateway_llm_route_reaches_llm_service(self) -> None:
        request = {
            **_request_ids("llm"),
            "user_id": "connectivity-user",
            "message": {"role": "user", "content": "检查 LLM 连接"},
            "stream": False,
        }
        with patch("services.gateway.main.httpx.AsyncClient", RoutedServiceAsyncClient):
            response = self.client.post(
                "/api/v1/text/completions",
                headers={"X-User-ID": "connectivity-user"},
                json=request,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "LLM 已收到：检查 LLM 连接")
        call = _single_post_call()
        self.assertEqual(call["url"], gateway_main.DEFAULT_LLM_URL)
        self.assertEqual(call["json"]["messages"][-1]["content"], "检查 LLM 连接")

    def test_gateway_tts_route_reaches_tts_service(self) -> None:
        request = {
            **_request_ids("tts"),
            "text": "检查 TTS 连接",
            "stream": False,
            "voice": {"mode": "custom", "speaker_id": "nana", "language": "Chinese"},
            "audio_config": {"format": "wav", "sample_rate": 24000, "channels": 1},
        }
        with patch("services.gateway.main.httpx.AsyncClient", RoutedServiceAsyncClient):
            response = self.client.post(
                "/api/v1/audio/speech",
                headers={"X-User-ID": "connectivity-user"},
                json=request,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["audio_base64"], "UklGRg==")
        call = _single_post_call()
        self.assertEqual(call["url"], gateway_main.DEFAULT_TTS_URL)
        self.assertEqual(call["json"]["voice"]["speaker_id"], "nana")

    def test_gateway_forwards_internal_service_keys(self) -> None:
        env = {
            "ASR_INTERNAL_API_KEY": "a" * 32,
            "TTS_INTERNAL_API_KEY": "t" * 32,
            "LLM_INTERNAL_API_KEY": "l" * 32,
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "services.gateway.main.httpx.AsyncClient",
            RoutedServiceAsyncClient,
        ):
            asr_response = self.client.post(
                "/api/v1/audio/transcriptions/base64",
                headers={"X-User-ID": "connectivity-user"},
                json={
                    **_request_ids("asr-key"),
                    "audio_base64": "UklGRg==",
                    "format": "wav",
                },
            )
            tts_response = self.client.post(
                "/api/v1/audio/speech",
                headers={"X-User-ID": "connectivity-user"},
                json={
                    **_request_ids("tts-key"),
                    "text": "内部认证测试",
                    "stream": False,
                },
            )
            llm_response = self.client.post(
                "/api/v1/diagnostics/llm",
                headers={"X-User-ID": "connectivity-user"},
                json={
                    **_request_ids("llm-key"),
                    "user_id": "connectivity-user",
                    "message": {"role": "user", "content": "LLM 密钥测试"},
                    "stream": False,
                },
            )
            text_response = self.client.post(
                "/api/v1/text/completions",
                headers={"X-User-ID": "connectivity-user"},
                json={
                    **_request_ids("text-key"),
                    "user_id": "connectivity-user",
                    "message": {"role": "user", "content": "文本链路密钥测试"},
                    "stream": False,
                },
            )

        self.assertEqual(
            [asr_response.status_code, tts_response.status_code, llm_response.status_code, text_response.status_code],
            [200, 200, 200, 200],
        )
        calls = [call for call in RoutedServiceAsyncClient.calls if call["method"] == "POST"]
        self.assertEqual(calls[0]["headers"]["X-Internal-API-Key"], "a" * 32)
        self.assertEqual(calls[1]["headers"]["X-Internal-API-Key"], "t" * 32)
        self.assertEqual(calls[2]["headers"]["X-Internal-API-Key"], "l" * 32)
        self.assertEqual(calls[3]["headers"]["X-Internal-API-Key"], "l" * 32)

def _single_post_call() -> dict[str, Any]:
    calls = [call for call in RoutedServiceAsyncClient.calls if call["method"] == "POST"]
    if len(calls) != 1:
        raise AssertionError(f"Expected one POST call, got {len(calls)}")
    return calls[0]


class GatewayHTTPPoolTest(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_http_client_reuses_lifespan_client(self) -> None:
        shared_client = object()
        previous = getattr(gateway_main.app.state, "http_clients", {})
        gateway_main.app.state.http_clients = {12.0: shared_client}
        try:
            async with gateway_main._gateway_http_client(12.0) as client:
                self.assertIs(client, shared_client)
        finally:
            gateway_main.app.state.http_clients = previous


@unittest.skipUnless(RUN_LIVE_TESTS, "Set NAWEI_RUN_LIVE_TESTS=1 to test running model services.")
class LiveServiceConnectivityTest(unittest.TestCase):
    """Exercise real running services through Gateway; explicitly opt in."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.gateway_url = os.getenv("NAWEI_GATEWAY_URL", "http://127.0.0.1:8000").rstrip("/")
        cls.asr_url = os.getenv("NAWEI_ASR_URL", "http://127.0.0.1:8001").rstrip("/")
        cls.llm_url = os.getenv("NAWEI_LLM_URL", "http://127.0.0.1:8002").rstrip("/")
        cls.tts_url = os.getenv("NAWEI_TTS_URL", "http://127.0.0.1:19000").rstrip("/")
        cls.client = httpx.Client(
            timeout=float(os.getenv("NAWEI_LIVE_TEST_TIMEOUT_SECONDS", "300")),
            trust_env=False,
        )
        audio_path = Path(
            os.getenv(
                "NAWEI_TEST_AUDIO",
                str(WORKSPACE_ROOT / "services" / "asr" / "tests" / "fixtures" / "test.wav"),
            )
        )
        if not audio_path.is_file():
            raise unittest.SkipTest(f"Live test audio does not exist: {audio_path}")
        cls.audio_name = audio_path.name
        cls.audio_base64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        cls.user_id = os.getenv("NAWEI_TEST_USER_ID", "live-test-user")
        cls.auth_headers = {"X-User-ID": cls.user_id}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_live_services_are_healthy_and_gateway_reports_ready(self) -> None:
        checks = (
            (f"{self.gateway_url}/health", "status", "ok"),
            (f"{self.asr_url}/health", "status", "ok"),
            (f"{self.llm_url}/health", "status", "ok"),
            (f"{self.tts_url}/api/v1/health", "status", "ok"),
        )
        for url, field, expected in checks:
            with self.subTest(url=url):
                response = self.client.get(url)
                response.raise_for_status()
                self.assertEqual(response.json().get(field), expected)

        status = self.client.get(f"{self.gateway_url}/api/v1/status")
        status.raise_for_status()
        self.assertEqual(status.json()["status"], "ready")

    def test_live_gateway_reaches_asr(self) -> None:
        response = self.client.post(
            f"{self.gateway_url}/api/v1/audio/transcriptions/base64",
            headers=self.auth_headers,
            json=self._audio_request("live-asr"),
        )
        response.raise_for_status()
        self.assertTrue(response.json()["text"].strip())

    def test_live_gateway_reaches_llm(self) -> None:
        response = self.client.post(
            f"{self.gateway_url}/api/v1/diagnostics/llm",
            headers=self.auth_headers,
            json={
                **_request_ids("live-llm"),
                "user_id": self.user_id,
                "message": {"role": "user", "content": "只回复：LLM互通成功"},
                "stream": False,
            },
        )
        response.raise_for_status()
        self.assertTrue(response.json()["text"].strip())

    def test_live_gateway_text_route_reaches_llm(self) -> None:
        response = self.client.post(
            f"{self.gateway_url}/api/v1/text/completions",
            headers=self.auth_headers,
            json={
                **_request_ids("live-text"),
                "user_id": self.user_id,
                "message": {"role": "user", "content": "只回复：文本链路互通成功"},
                "stream": False,
            },
        )
        response.raise_for_status()
        body = response.json()
        self.assertTrue(body["text"].strip())
        self.assertIn(body["metadata"].get("provider", "stub"), {"stub", "deepseek"})

    def test_live_gateway_reaches_tts(self) -> None:
        response = self.client.post(
            f"{self.gateway_url}/api/v1/audio/speech",
            headers=self.auth_headers,
            json={
                **_request_ids("live-tts"),
                "text": "语音模块互通测试成功。",
                "stream": False,
                "voice": {"mode": "custom", "speaker_id": "nana", "language": "Chinese"},
                "audio_config": {"format": "wav", "sample_rate": 24000, "channels": 1},
            },
        )
        response.raise_for_status()
        self.assertTrue(response.json().get("audio_base64") or response.json().get("audio_url"))

    def _audio_request(self, prefix: str) -> dict[str, Any]:
        return {
            **_request_ids(prefix),
            "audio_base64": self.audio_base64,
            "language": "auto",
            "format": "wav",
            "sample_rate": 16000,
            "channels": 1,
            "metadata": {"filename": self.audio_name, "use_itn": True},
        }

if __name__ == "__main__":
    unittest.main()
