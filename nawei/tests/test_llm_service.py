"""Tests for the configurable DeepSeek LLM service backend."""

from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.llm_service.main import _load_dotenv_file, app


class FakeDeepSeekResponse:
    """Minimal fake httpx response for DeepSeek tests."""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "id": "chatcmpl-test",
            "model": "real-model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "真实回复"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }


class FakeAsyncClient:
    """Capture outbound calls without making network requests."""

    calls: list[dict[str, object]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def post(self, url: str, json: dict[str, object], headers: dict[str, str]) -> FakeDeepSeekResponse:
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": self.timeout})
        return FakeDeepSeekResponse()


class FakeDeepSeekStreamResponse:
    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        chunks = (
            {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "world."}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        )
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"


class FakeStreamContext:
    async def __aenter__(self) -> FakeDeepSeekStreamResponse:
        return FakeDeepSeekStreamResponse()

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class FakeStreamingAsyncClient:
    calls: list[dict[str, object]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeStreamingAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def stream(self, method: str, url: str, json: dict[str, object], headers: dict[str, str]) -> FakeStreamContext:
        self.calls.append(
            {"method": method, "url": url, "json": json, "headers": headers, "timeout": self.timeout}
        )
        return FakeStreamContext()


class LLMServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeAsyncClient.calls.clear()
        FakeStreamingAsyncClient.calls.clear()

    def test_dotenv_loader_reads_local_file_without_overriding_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text(
                "LLM_PROVIDER=deepseek\n"
                "LLM_API_BASE_URL=https://api.deepseek.com\n"
                "LLM_API_KEY=from-file\n"
                "LLM_MODEL=deepseek-v4-pro\n",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"LLM_API_KEY": "from-env"}, clear=True):
                _load_dotenv_file(dotenv_path)

                self.assertEqual(os.getenv("LLM_PROVIDER"), "deepseek")
                self.assertEqual(os.getenv("LLM_API_BASE_URL"), "https://api.deepseek.com")
                self.assertEqual(os.getenv("LLM_API_KEY"), "from-env")
                self.assertEqual(os.getenv("LLM_MODEL"), "deepseek-v4-pro")

    def test_stub_backend_is_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = TestClient(app)

            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json(), {"status": "ok", "model": "stub-llm", "provider": "stub"})

            response = client.post(
                "/api/v1/llm",
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["text"], "我听到了：你好")
        self.assertEqual(body["metadata"]["model"], "stub-llm")

    def test_placeholder_api_key_returns_configuration_error(self) -> None:
        env = {
            "LLM_PROVIDER": "deepseek",
            "LLM_API_BASE_URL": "https://api.deepseek.com",
            "LLM_API_KEY": "REPLACE_WITH_YOUR_DEEPSEEK_API_KEY",
            "LLM_MODEL": "deepseek-v4-pro",
        }
        with patch.dict("os.environ", env, clear=True):
            client = TestClient(app)
            response = client.post(
                "/api/v1/llm",
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": False,
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertIn("Replace the LLM_API_KEY placeholder", response.json()["detail"])

    def test_deepseek_backend_maps_request_and_response(self) -> None:
        env = {
            "LLM_PROVIDER": "deepseek",
            "LLM_API_BASE_URL": "https://api.deepseek.com",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "real-model",
            "LLM_TIMEOUT_SECONDS": "12",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("services.llm_service.main.httpx.AsyncClient", FakeAsyncClient):
                client = TestClient(app)
                response = client.post(
                    "/api/v1/llm",
                    json={
                        "request_id": "r",
                        "trace_id": "t",
                        "session_id": "s",
                        "messages": [{"role": "user", "content": "你好"}],
                        "stream": False,
                        "generation_config": {"temperature": 0.2, "max_tokens": 64},
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["text"], "真实回复")
        self.assertEqual(body["usage"], {"input_tokens": 3, "output_tokens": 4})
        self.assertEqual(body["metadata"]["provider"], "deepseek")
        self.assertEqual(body["metadata"]["upstream_id"], "chatcmpl-test")

        self.assertEqual(len(FakeAsyncClient.calls), 1)
        call = FakeAsyncClient.calls[0]
        self.assertEqual(call["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(call["timeout"], 12.0)
        payload = call["json"]
        self.assertEqual(payload["model"], "real-model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "你好"}])
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 64)

    def test_deepseek_stream_maps_sse_events(self) -> None:
        env = {
            "LLM_PROVIDER": "deepseek",
            "LLM_API_BASE_URL": "https://api.deepseek.com",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "real-model",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("services.llm_service.main.httpx.AsyncClient", FakeStreamingAsyncClient):
                response = TestClient(app).post(
                    "/api/v1/llm/stream",
                    json={
                        "request_id": "r",
                        "trace_id": "t",
                        "session_id": "s",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: message.delta", response.text)
        self.assertIn('"text":"Hello "', response.text)
        self.assertIn('"text":"world."', response.text)
        self.assertIn("event: message.done", response.text)
        self.assertIn('"input_tokens":2', response.text)
        self.assertTrue(FakeStreamingAsyncClient.calls[0]["json"]["stream"])


if __name__ == "__main__":
    unittest.main()
