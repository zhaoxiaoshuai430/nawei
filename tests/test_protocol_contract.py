"""Contract tests for app_protocols and service orchestration."""

from __future__ import annotations

import asyncio
import unittest
from typing import AsyncIterator
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app_protocols import (
    ASRDetails,
    ASRRequest,
    ASRResponse,
    AudioConfig,
    ChatCompletionRequest,
    EventType,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    StreamEvent,
    TTSRequest,
    TTSResponse,
    Usage,
)
from app_protocols.orchestrator import ConversationOrchestrator
from services.gateway import main as gateway_main


class FakeASRProvider:
    async def transcribe(self, request: ASRRequest) -> ASRResponse:
        return ASRResponse(
            request_id=request.request_id,
            text="你好",
            language="zh",
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


class RecordingLLMProvider(FakeLLMProvider):
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return await super().complete(request)


class FakeVoiceStreamingLLMProvider(FakeLLMProvider):
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        for text in ("First sentence.", "Second sentence!"):
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
            data={"message_id": request.request_id, "finish_reason": "stop", "usage": {}},
        )


class FakeVoiceStreamingTTSProvider(FakeTTSProvider):
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def stream_synthesize(self, request: TTSRequest) -> AsyncIterator[StreamEvent]:
        self.texts.append(request.text)
        yield StreamEvent(
            event=EventType.AUDIO_DELTA,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={
                "audio_base64": "UklGRg==",
                "format": request.audio_config.format,
                "sample_rate": request.audio_config.sample_rate,
                "channels": request.audio_config.channels,
            },
        )


class FakeGatewayLLMResponse:
    """Minimal fake HTTP response returned by the standalone LLM service."""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "request_id": "r",
            "text": "reply: hello",
            "finish_reason": "stop",
            "usage": {"input_tokens": 2, "output_tokens": 5},
            "metadata": {"provider": "http-test"},
        }


class FakeGatewayHTTPResponse:
    """Minimal fake HTTP response with a configurable JSON body."""

    def __init__(self, body: dict[str, object]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._body


class FakeGatewayAsyncClient:
    """Capture gateway outbound LLM HTTP calls."""

    calls: list[dict[str, object]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeGatewayAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def post(self, url: str, json: dict[str, object]) -> FakeGatewayLLMResponse:
        self.calls.append({"url": url, "json": json, "timeout": self.timeout})
        return FakeGatewayLLMResponse()


class FakeGatewayHealthAsyncClient:
    """Return configurable health documents for Gateway status probes."""

    ready = True

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeGatewayHealthAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def get(self, url: str) -> FakeGatewayHTTPResponse:
        if url.endswith(":19000/api/v1/status"):
            return FakeGatewayHTTPResponse({"loaded": self.ready})
        if url.endswith(":8002/health") and not self.ready:
            return FakeGatewayHTTPResponse({"status": "loading"})
        return FakeGatewayHTTPResponse({"status": "ok"})


class FakeGatewayTTSAsyncClient:
    """Capture gateway outbound TTS HTTP JSON calls."""

    calls: list[dict[str, object]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeGatewayTTSAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def post(self, url: str, json: dict[str, object]) -> FakeGatewayHTTPResponse:
        self.calls.append({"url": url, "json": json, "timeout": self.timeout})
        return FakeGatewayHTTPResponse(
            {
                "request_id": json["request_id"],
                "audio_url": None,
                "audio_base64": "UklGRg==",
                "duration_ms": 1250,
                "format": "wav",
                "sample_rate": 24000,
                "rtf": 1.7,
                "metadata": {"provider": "http-test"},
            }
        )


class FakeGatewayLLMTTSAsyncClient:
    """Capture and route Gateway outbound LLM then TTS HTTP calls."""

    calls: list[dict[str, object]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeGatewayLLMTTSAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def post(self, url: str, json: dict[str, object]) -> FakeGatewayHTTPResponse:
        self.calls.append({"url": url, "json": json, "timeout": self.timeout})
        if url.endswith("/api/v1/llm"):
            user_text = json["messages"][-1]["content"]  # type: ignore[index]
            return FakeGatewayHTTPResponse(
                {
                    "request_id": json["request_id"],
                    "text": f"reply: {user_text}",
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 2, "output_tokens": 5},
                    "metadata": {"provider": "http-test"},
                }
            )

        return FakeGatewayHTTPResponse(
            {
                "request_id": json["request_id"],
                "audio_url": None,
                "audio_base64": "UklGRg==",
                "duration_ms": 1250,
                "format": "wav",
                "sample_rate": 24000,
                "rtf": 1.7,
                "metadata": {"provider": "tts-test"},
            }
        )


class ProtocolContractTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeGatewayAsyncClient.calls.clear()
        FakeGatewayTTSAsyncClient.calls.clear()
        FakeGatewayLLMTTSAsyncClient.calls.clear()

    def test_gateway_root_is_api_only(self) -> None:
        client = TestClient(gateway_main.app)

        page = client.get("/")
        stylesheet = client.get("/static/css/voice_app.css")
        script = client.get("/static/js/voice_app.js")

        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.json()["service"], "gateway")
        self.assertEqual(page.json()["realtime_voice"], "/api/v1/voice/realtime")
        self.assertEqual(stylesheet.status_code, 404)
        self.assertEqual(script.status_code, 404)

    def test_gateway_dependency_status_reports_ready_and_degraded(self) -> None:
        client = TestClient(gateway_main.app)
        with patch("services.gateway.main.httpx.AsyncClient", FakeGatewayHealthAsyncClient):
            FakeGatewayHealthAsyncClient.ready = True
            ready_response = client.get("/api/v1/status")
            FakeGatewayHealthAsyncClient.ready = False
            degraded_response = client.get("/api/v1/status")

        self.assertEqual(ready_response.status_code, 200)
        self.assertEqual(ready_response.json()["status"], "ready")
        self.assertTrue(all(item["ready"] for item in ready_response.json()["services"].values()))
        self.assertEqual(degraded_response.status_code, 200)
        self.assertEqual(degraded_response.json()["status"], "degraded")
        self.assertFalse(degraded_response.json()["services"]["llm"]["ready"])
        self.assertFalse(degraded_response.json()["services"]["tts"]["ready"])

    def test_asr_request_requires_exactly_one_audio_source(self) -> None:
        ASRRequest(request_id="r", trace_id="t", session_id="s", audio_base64="abc")

        with self.assertRaises(ValidationError):
            ASRRequest(request_id="r", trace_id="t", session_id="s")

        with self.assertRaises(ValidationError):
            ASRRequest(
                request_id="r",
                trace_id="t",
                session_id="s",
                audio_url="upload://a.wav",
                audio_base64="abc",
            )

    def test_gateway_exposes_llm_only_and_tts_only_paths(self) -> None:
        client = TestClient(gateway_main.app)

        with patch("services.gateway.main.httpx.AsyncClient", FakeGatewayAsyncClient):
            llm_response = client.post(
                "/api/v1/text/completions",
                headers={"X-User-ID": "test-user"},
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
                    "user_id": "test-user",
                    "message": {"role": "user", "content": "hello"},
                    "stream": False,
                    "output_audio": True,
                },
            )
        self.assertEqual(llm_response.status_code, 200)
        llm_body = llm_response.json()
        self.assertEqual(llm_body["text"], "reply: hello")
        self.assertIsNone(llm_body["audio_base64"])
        self.assertEqual(len(FakeGatewayAsyncClient.calls), 1)
        call = FakeGatewayAsyncClient.calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:8002/api/v1/llm")
        self.assertEqual(call["json"]["request_id"], "r")
        self.assertEqual(
            call["json"]["messages"][-1],
            {
                "role": "user",
                "content": "hello",
                "name": None,
                "tool_call_id": None,
                "tool_calls": [],
                "metadata": {},
            },
        )

        with patch("services.gateway.main.httpx.AsyncClient", FakeGatewayTTSAsyncClient):
            tts_response = client.post(
                "/api/v1/audio/speech",
                headers={"X-User-ID": "test-user"},
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
                    "text": "hello",
                    "stream": False,
                    "metadata": {"temperature": 0.8, "top_k": 40},
                },
            )
        self.assertEqual(tts_response.status_code, 200)
        tts_body = tts_response.json()
        self.assertEqual(tts_body["audio_base64"], "UklGRg==")
        self.assertEqual(tts_body["duration_ms"], 1250)
        self.assertEqual(tts_body["sample_rate"], 24000)
        self.assertEqual(tts_body["metadata"]["provider"], "http-test")

        self.assertEqual(len(FakeGatewayTTSAsyncClient.calls), 1)
        tts_call = FakeGatewayTTSAsyncClient.calls[0]
        self.assertEqual(tts_call["url"], "http://127.0.0.1:19000/api/v1/audio/speech")
        self.assertEqual(tts_call["json"]["voice"], {"mode": "custom", "speaker_id": None, "language": "auto", "instruct": None})
        self.assertEqual(tts_call["json"]["metadata"], {"temperature": 0.8, "top_k": 40})

    def test_gateway_exposes_llm_then_tts_path(self) -> None:
        client = TestClient(gateway_main.app)

        with patch("services.gateway.main.httpx.AsyncClient", FakeGatewayLLMTTSAsyncClient):
            response = client.post(
                "/api/v1/text/speech-completions",
                headers={"X-User-ID": "test-user"},
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
                    "user_id": "test-user",
                    "message": {"role": "user", "content": "hello"},
                    "stream": False,
                    "output_audio": False,
                    "voice": {"mode": "custom", "speaker_id": "nana", "language": "Chinese", "instruct": ""},
                    "audio_config": {"format": "wav", "sample_rate": 24000, "channels": 1},
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["text"], "reply: hello")
        self.assertEqual(body["audio_base64"], "UklGRg==")
        self.assertEqual(body["duration_ms"], 1250)
        self.assertEqual(body["metadata"]["provider"], "http-test")
        self.assertEqual(body["metadata"]["tts"]["provider"], "tts-test")

        self.assertEqual(len(FakeGatewayLLMTTSAsyncClient.calls), 2)
        llm_call, tts_call = FakeGatewayLLMTTSAsyncClient.calls
        self.assertEqual(llm_call["url"], "http://127.0.0.1:8002/api/v1/llm")
        self.assertEqual(llm_call["json"]["messages"][-1]["content"], "hello")
        self.assertEqual(tts_call["url"], "http://127.0.0.1:19000/api/v1/audio/speech")
        self.assertEqual(tts_call["json"]["text"], "reply: hello")
        self.assertEqual(tts_call["json"]["voice"]["speaker_id"], "nana")

    def test_complete_voice_http_routes_are_removed(self) -> None:
        client = TestClient(gateway_main.app)
        payload = {"request_id": "r", "trace_id": "t", "session_id": "s"}

        self.assertEqual(client.post("/api/v1/voice/completions", json=payload).status_code, 404)
        self.assertEqual(client.post("/api/v1/voice/completions/stream", json=payload).status_code, 404)
        self.assertNotIn("/api/v1/voice/completions", gateway_main.app.openapi()["paths"])
        self.assertNotIn("/api/v1/voice/completions/stream", gateway_main.app.openapi()["paths"])

    def test_gateway_delegates_user_identity_and_history_to_llm(self) -> None:
        llm = RecordingLLMProvider()
        previous = gateway_main.orchestrator.llm_provider
        gateway_main.orchestrator.llm_provider = llm
        client = TestClient(gateway_main.app)
        try:
            for request_id, session_id, content in (
                ("one", "session-a", "first"),
                ("two", "session-a", "second"),
                ("other", "session-b", "isolated"),
            ):
                response = client.post(
                    "/api/v1/text/completions",
                    headers={"X-User-ID": "test-user"},
                    json={
                        "request_id": request_id,
                        "trace_id": "trace",
                        "session_id": session_id,
                        "user_id": "test-user",
                        "message": {"role": "user", "content": content},
                        "stream": False,
                    },
                )
                self.assertEqual(response.status_code, 200)
        finally:
            gateway_main.orchestrator.llm_provider = previous

        self.assertEqual(
            [message.content for message in llm.requests[1].messages],
            ["second"],
        )
        self.assertEqual(
            [message.content for message in llm.requests[2].messages],
            ["isolated"],
        )
        self.assertTrue(all(request.user_id == "test-user" for request in llm.requests))

    def test_realtime_voice_delegates_identity_and_message_to_llm(self) -> None:
        llm = FakeVoiceStreamingLLMProvider()
        previous = gateway_main.orchestrator.llm_provider
        gateway_main.orchestrator.llm_provider = llm
        request = gateway_main.RealtimeVoiceRequest(
            request_id="stream-one",
            trace_id="trace",
            session_id="stream-session",
            user_id="test-user",
            output_audio=False,
        )

        async def collect_events() -> list[StreamEvent]:
            message = Message(role=MessageRole.USER, content="你好")
            return [event async for event in gateway_main._stream_voice_response(request, message)]

        try:
            events = asyncio.run(collect_events())
        finally:
            gateway_main.orchestrator.llm_provider = previous

        self.assertEqual([message.content for message in llm.requests[0].messages], ["你好"])
        self.assertEqual(llm.requests[0].user_id, "test-user")
        self.assertEqual(events[-1].event, EventType.MESSAGE_DONE)

    def test_realtime_voice_interleaves_text_and_sentence_audio(self) -> None:
        streaming_tts = FakeVoiceStreamingTTSProvider()
        previous = (
            gateway_main.orchestrator.llm_provider,
            gateway_main.orchestrator.tts_provider,
        )
        gateway_main.orchestrator.llm_provider = FakeVoiceStreamingLLMProvider()
        gateway_main.orchestrator.tts_provider = streaming_tts
        request = gateway_main.RealtimeVoiceRequest(
            request_id="r",
            trace_id="t",
            session_id="s",
            user_id="test-user",
            output_audio=True,
            voice={"mode": "custom", "speaker_id": "nana", "language": "Chinese"},
        )

        async def collect_events() -> list[StreamEvent]:
            message = Message(role=MessageRole.USER, content="你好")
            return [event async for event in gateway_main._stream_voice_response(request, message)]

        try:
            events = asyncio.run(collect_events())
        finally:
            (
                gateway_main.orchestrator.llm_provider,
                gateway_main.orchestrator.tts_provider,
            ) = previous

        self.assertEqual(streaming_tts.texts, ["First sentence.", "Second sentence!"])
        self.assertEqual(
            [event.event for event in events],
            [
                EventType.MESSAGE_DELTA,
                EventType.AUDIO_STARTED,
                EventType.AUDIO_DELTA,
                EventType.MESSAGE_DELTA,
                EventType.AUDIO_DELTA,
                EventType.MESSAGE_DONE,
                EventType.AUDIO_DONE,
            ],
        )


class OrchestratorContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_maps_asr_details_to_client_response(self) -> None:
        orchestrator = ConversationOrchestrator(
            asr_provider=FakeASRProvider(),
            llm_provider=FakeLLMProvider(),
            tts_provider=FakeTTSProvider(),
        )

        response = await orchestrator.transcribe(
            ASRRequest(request_id="r", trace_id="t", session_id="s", audio_base64="abc")
        )

        self.assertEqual(response.trace_id, "t")
        self.assertEqual(response.asr.emotion, "HAPPY")
        self.assertEqual(response.asr.clean_text, "你好")

    async def test_complete_preserves_tts_audio_config(self) -> None:
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
                message=Message(role=MessageRole.USER, content="你好"),
                output_audio=True,
                audio_config=AudioConfig(sample_rate=16000),
                stream=False,
            )
        )

        self.assertEqual(response.audio_base64, "UklGRg==")
        self.assertEqual(response.sample_rate, 16000)
        self.assertEqual(response.duration_ms, 1000)


if __name__ == "__main__":
    unittest.main()
