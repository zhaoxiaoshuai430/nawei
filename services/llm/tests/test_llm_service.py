"""Tests for the configurable DeepSeek LLM service backend."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from services.llm.main import _load_dotenv_file, app


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


class FakeEmptyContentResponse(FakeDeepSeekResponse):
    def json(self) -> dict[str, object]:
        return {
            "id": "chatcmpl-empty",
            "model": "real-model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 120},
        }


class FakeEmptyContentAsyncClient(FakeAsyncClient):
    async def post(self, url: str, json: dict[str, object], headers: dict[str, str]) -> FakeEmptyContentResponse:
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": self.timeout})
        return FakeEmptyContentResponse()


class FakeEmptyStreamResponse:
    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        yield "data: [DONE]"


class FakeEmptyStreamContext:
    async def __aenter__(self) -> FakeEmptyStreamResponse:
        return FakeEmptyStreamResponse()

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class FakeEmptyStreamAsyncClient(FakeAsyncClient):
    stream_calls: list[dict[str, object]] = []

    def stream(self, method: str, url: str, json: dict[str, object], headers: dict[str, str]) -> FakeEmptyStreamContext:
        self.stream_calls.append({"method": method, "url": url, "json": json, "headers": headers, "timeout": self.timeout})
        return FakeEmptyStreamContext()


class FakeTimeoutAsyncClient(FakeAsyncClient):
    """Fake client that simulates upstream LLM timeouts."""

    async def post(self, url: str, json: dict[str, object], headers: dict[str, str]) -> FakeDeepSeekResponse:
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": self.timeout})
        raise httpx.ReadTimeout("timed out")

    def stream(self, method: str, url: str, json: dict[str, object], headers: dict[str, str]) -> FakeEmptyStreamContext:
        raise httpx.ReadTimeout("stream timed out")


class LLMServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeAsyncClient.calls.clear()
        FakeEmptyContentAsyncClient.calls.clear()
        FakeEmptyStreamAsyncClient.stream_calls.clear()

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
            health_body = health.json()
            self.assertEqual(health_body["status"], "ok")
            self.assertEqual(health_body["model"], "stub-llm")
            self.assertEqual(health_body["provider"], "stub")
            self.assertTrue(health_body["runtime"]["agent"]["enabled"])
            self.assertEqual(health_body["runtime"]["skills"], ["memory", "concept", "tutor"])

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
        self.assertIn("我听到了", body["text"])
        self.assertIn("学习辅导", body["text"])
        self.assertEqual(body["metadata"]["model"], "stub-llm")

    def test_stub_backend_streams_sse(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = TestClient(app)
            with client.stream(
                "POST",
                "/api/v1/llm/stream",
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": True,
                },
            ) as response:
                body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: message.delta", body)
        self.assertIn("我听到了", body)

    def test_stub_backend_treats_learning_statement_as_normal_input(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = TestClient(app)
            response = client.post(
                "/api/v1/llm",
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
                    "messages": [{"role": "user", "content": "我叫小明，我最近在学二次函数"}],
                    "stream": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("二次函数", body["text"])
        self.assertNotIn("还没有读到稳定的个人记忆", body["text"])

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
            with patch("services.llm.main.httpx.AsyncClient", FakeAsyncClient):
                client = TestClient(app)
                response = client.post(
                    "/api/v1/llm",
                    json={
                        "request_id": "r",
                        "trace_id": "t",
                        "session_id": "s",
                        "user_id": "student-1",
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
        self.assertEqual(body["metadata"]["runtime"]["memory"]["user_id"], "student-1")
        self.assertEqual(body["metadata"]["runtime"]["skills"]["active"], ["memory", "concept", "tutor"])

        self.assertEqual(len(FakeAsyncClient.calls), 1)
        call = FakeAsyncClient.calls[0]
        self.assertEqual(call["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(call["timeout"], 12.0)
        payload = call["json"]
        self.assertEqual(payload["model"], "real-model")
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "\u4f60\u597d"})
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("小维", payload["messages"][0]["content"])
        self.assertIn("本轮 Agent 判断", payload["messages"][0]["content"])
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 64)

    def test_deepseek_empty_stream_falls_back_to_non_stream_response(self) -> None:
        env = {
            "LLM_PROVIDER": "deepseek",
            "LLM_API_BASE_URL": "https://api.deepseek.com",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "real-model",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("services.llm.main.httpx.AsyncClient", FakeEmptyStreamAsyncClient):
                client = TestClient(app)
                with client.stream(
                    "POST",
                    "/api/v1/llm/stream",
                    json={
                        "request_id": "r",
                        "trace_id": "t",
                        "session_id": "s",
                        "messages": [{"role": "user", "content": "你好"}],
                        "stream": True,
                    },
                ) as response:
                    body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: message.delta", body)
        self.assertIn("真实回复", body)
        self.assertIn("non_stream_after_empty_stream", body)
        self.assertEqual(len(FakeEmptyStreamAsyncClient.stream_calls), 1)
        self.assertEqual(len(FakeEmptyStreamAsyncClient.calls), 1)

    def test_deepseek_empty_content_returns_fallback_response(self) -> None:
        env = {
            "LLM_PROVIDER": "deepseek",
            "LLM_API_BASE_URL": "https://api.deepseek.com",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "real-model",
            "LLM_FALLBACK_ENABLED": "true",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("services.llm.main.httpx.AsyncClient", FakeEmptyContentAsyncClient):
                client = TestClient(app)
                response = client.post(
                    "/api/v1/llm",
                    json={
                        "request_id": "r",
                        "trace_id": "t",
                        "session_id": "s",
                        "messages": [{"role": "user", "content": "这道题怎么做"}],
                        "stream": False,
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["finish_reason"], "fallback")
        self.assertTrue(body["metadata"]["fallback"])
        self.assertIn("empty content", body["metadata"]["fallback_reason"])
        self.assertTrue(body["text"].strip())

    def test_deepseek_timeout_returns_fallback_response(self) -> None:
        env = {
            "LLM_PROVIDER": "deepseek",
            "LLM_API_BASE_URL": "https://api.deepseek.com",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "real-model",
            "LLM_TIMEOUT_SECONDS": "0.1",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("services.llm.main.httpx.AsyncClient", FakeTimeoutAsyncClient):
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

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["finish_reason"], "fallback")
        self.assertTrue(body["metadata"]["fallback"])
        self.assertEqual(body["metadata"]["provider"], "deepseek")
        self.assertIn("timed out", body["metadata"]["fallback_reason"])
        self.assertTrue(body["text"])

    def test_deepseek_timeout_returns_502_when_fallback_is_disabled(self) -> None:
        env = {
            "LLM_PROVIDER": "deepseek",
            "LLM_API_BASE_URL": "https://api.deepseek.com",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "real-model",
            "LLM_FALLBACK_ENABLED": "false",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("services.llm.main.httpx.AsyncClient", FakeTimeoutAsyncClient):
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

        self.assertEqual(response.status_code, 502)
        self.assertIn("LLM upstream request failed", response.json()["detail"])

    def test_deepseek_stream_timeout_streams_fallback_and_done(self) -> None:
        env = {
            "LLM_PROVIDER": "deepseek",
            "LLM_API_BASE_URL": "https://api.deepseek.com",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "real-model",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("services.llm.main.httpx.AsyncClient", FakeTimeoutAsyncClient):
                client = TestClient(app)
                with client.stream(
                    "POST",
                    "/api/v1/llm/stream",
                    json={
                        "request_id": "r",
                        "trace_id": "t",
                        "session_id": "s",
                        "messages": [{"role": "user", "content": "你好"}],
                        "stream": True,
                    },
                ) as response:
                    body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: message.delta", body)
        self.assertIn("event: message.done", body)
        self.assertIn('"finish_reason":"fallback"', body)
        self.assertIn('"fallback":true', body)

    def test_llm_memory_management_lifecycle(self) -> None:
        user_id = "memory-api-student"
        with patch.dict("os.environ", {"MEMORY_ENABLED": "true"}, clear=True):
            client = TestClient(app)
            stored = client.post(
                f"/api/v1/llm/memory/{user_id}/remember",
                json={
                    "session_id": "memory-session",
                    "content": "我正在学习二次函数，并且需要分步骤讲解",
                    "memory_type": "learning_state",
                    "importance": 8,
                    "confidence": 0.95,
                    "tags": ["数学", "二次函数"],
                },
            )
            snapshot = client.get(
                f"/api/v1/llm/memory/{user_id}",
                params={"session_id": "memory-session"},
            )

            self.assertEqual(stored.status_code, 200)
            self.assertTrue(stored.json()["stored"])
            memory_id = stored.json()["memory_id"]
            self.assertEqual(snapshot.status_code, 200)
            self.assertTrue(any(item["id"] == memory_id for item in snapshot.json()["memories"]))

            deleted = client.delete(f"/api/v1/llm/memory/{user_id}/items/{memory_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertTrue(deleted.json()["deleted"])


if __name__ == "__main__":
    unittest.main()
