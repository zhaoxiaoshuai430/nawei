"""Contract tests for app_protocols and service orchestration."""

from __future__ import annotations

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
from services.gateway_service import main as gateway_main
from services.gateway_service.conversation_store import InMemoryConversationStore
from services.tts_service.main import app as tts_app


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


class RecordingASRProvider(FakeASRProvider):
    def __init__(self) -> None:
        self.requests: list[ASRRequest] = []

    async def transcribe(self, request: ASRRequest) -> ASRResponse:
        self.requests.append(request)
        return await super().transcribe(request)


class RecordingLLMProvider(FakeLLMProvider):
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return await super().complete(request)


class RecordingTTSProvider(FakeTTSProvider):
    def __init__(self) -> None:
        self.requests: list[TTSRequest] = []

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        self.requests.append(request)
        return await super().synthesize(request)


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


class FakeGatewayRouteAsyncClient:
    """Capture and route Gateway outbound ASR and LLM HTTP calls."""

    calls: list[dict[str, object]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeGatewayRouteAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def post(self, url: str, json: dict[str, object]) -> FakeGatewayHTTPResponse:
        self.calls.append({"url": url, "json": json, "timeout": self.timeout})
        if url.endswith("/api/v1/audio/transcriptions/base64"):
            options = json.get("options") or {}
            return FakeGatewayHTTPResponse(
                {
                    "request_id": json["request_id"],
                    "text": "recognized hello",
                    "language": "zh",
                    "confidence": None,
                    "duration_ms": None,
                    "segments": [],
                    "asr": {
                        "key": options.get("filename"),  # type: ignore[union-attr]
                        "emotion": "NEUTRAL",
                        "emotion_label": "calm",
                        "emotion_emoji": "",
                        "events": [{"tag": "speech", "label": "Speech", "emoji": ""}],
                        "emotion_hint": "calm",
                        "raw_text": "<|zh|>recognized hello",
                        "clean_text": "recognized hello",
                    },
                    "metadata": {"provider": "http-test"},
                }
            )
            return FakeGatewayHTTPResponse(
                {
                    "result": [
                        {
                            "key": json["filename"],
                            "text": "recognized hello",
                            "language": "zh",
                            "emotion": "NEUTRAL",
                            "emotion_label": "平静",
                            "emotion_emoji": "",
                            "events": [{"tag": "speech", "label": "Speech", "emoji": ""}],
                            "emotion_hint": "calm",
                            "raw_text": "<|zh|>recognized hello",
                            "clean_text": "recognized hello",
                        }
                    ]
                }
            )

        text = json["messages"][-1]["content"]  # type: ignore[index]
        return FakeGatewayHTTPResponse(
            {
                "request_id": json["request_id"],
                "text": f"reply: {text}",
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 5},
                "metadata": {"provider": "http-test"},
            }
        )


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
        FakeGatewayRouteAsyncClient.calls.clear()
        FakeGatewayTTSAsyncClient.calls.clear()
        FakeGatewayLLMTTSAsyncClient.calls.clear()
        gateway_main.conversation_store = gateway_main.InMemoryConversationStore()

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

    def test_tts_service_matches_api_md_generate_shape(self) -> None:
        client = TestClient(tts_app)

        health = client.get("/api/v1/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok", "service": "tts"})

        response = client.post(
            "/api/v1/tts/generate",
            data={
                "text": "你好",
                "mode": "custom",
                "speaker": "liliya",
                "language": "Chinese",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("audio_b64", body)
        self.assertEqual(body["sample_rate"], 24000)
        self.assertIn("metrics", body)
        self.assertIn("rtf", body["metrics"])

    def test_gateway_exposes_llm_only_and_tts_only_paths(self) -> None:
        client = TestClient(gateway_main.app)

        with patch("services.gateway_service.main.httpx.AsyncClient", FakeGatewayAsyncClient):
            llm_response = client.post(
                "/api/v1/text/completions",
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
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
        self.assertEqual(call["json"]["messages"][-1], {"role": "user", "content": "hello", "name": None, "metadata": {}})

        with patch("services.gateway_service.main.httpx.AsyncClient", FakeGatewayTTSAsyncClient):
            tts_response = client.post(
                "/api/v1/audio/speech",
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

        with patch("services.gateway_service.main.httpx.AsyncClient", FakeGatewayLLMTTSAsyncClient):
            response = client.post(
                "/api/v1/text/speech-completions",
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
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

    def test_gateway_voice_completion_uses_http_asr_then_llm(self) -> None:
        client = TestClient(gateway_main.app)

        with patch("services.gateway_service.main.httpx.AsyncClient", FakeGatewayRouteAsyncClient):
            response = client.post(
                "/api/v1/voice/completions",
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
                    "audio_base64": "UklGRg==",
                    "output_audio": False,
                    "metadata": {"filename": "sample.wav"},
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["transcription"]["text"], "recognized hello")
        self.assertEqual(body["transcription"]["asr"]["key"], "sample.wav")
        self.assertEqual(body["completion"]["text"], "reply: recognized hello")

        self.assertEqual(len(FakeGatewayRouteAsyncClient.calls), 2)
        asr_call, llm_call = FakeGatewayRouteAsyncClient.calls
        self.assertEqual(asr_call["url"], "http://127.0.0.1:8001/api/v1/audio/transcriptions/base64")
        self.assertEqual(asr_call["json"]["audio_base64"], "UklGRg==")
        self.assertEqual(asr_call["json"]["options"]["filename"], "sample.wav")
        self.assertEqual(llm_call["url"], "http://127.0.0.1:8002/api/v1/llm")
        self.assertEqual(llm_call["json"]["messages"][-1]["content"], "recognized hello")

    def test_gateway_voice_completion_rejects_invalid_audio_sources(self) -> None:
        client = TestClient(gateway_main.app, raise_server_exceptions=False)
        required = {"request_id": "r", "trace_id": "t", "session_id": "s"}

        missing = client.post("/api/v1/voice/completions", json=required)
        duplicate = client.post(
            "/api/v1/voice/completions",
            json={**required, "audio_url": "https://example.test/a.wav", "audio_base64": "UklGRg=="},
        )

        self.assertEqual(missing.status_code, 422)
        self.assertEqual(duplicate.status_code, 422)

    def test_gateway_voice_completion_separates_input_and_output_audio_config(self) -> None:
        asr = RecordingASRProvider()
        llm = RecordingLLMProvider()
        tts = RecordingTTSProvider()
        previous = (
            gateway_main.orchestrator.asr_provider,
            gateway_main.orchestrator.llm_provider,
            gateway_main.orchestrator.tts_provider,
        )
        gateway_main.orchestrator.asr_provider = asr
        gateway_main.orchestrator.llm_provider = llm
        gateway_main.orchestrator.tts_provider = tts
        try:
            response = TestClient(gateway_main.app).post(
                "/api/v1/voice/completions",
                json={
                    "request_id": "audio-split",
                    "trace_id": "trace",
                    "session_id": "audio-session",
                    "audio_base64": "UklGRg==",
                    "output_audio": True,
                    "input_audio_config": {"format": "wav", "sample_rate": 16000, "channels": 1},
                    "output_audio_config": {"format": "mp3", "sample_rate": 24000, "channels": 2},
                },
            )
        finally:
            (
                gateway_main.orchestrator.asr_provider,
                gateway_main.orchestrator.llm_provider,
                gateway_main.orchestrator.tts_provider,
            ) = previous

        self.assertEqual(response.status_code, 200)
        self.assertEqual(asr.requests[0].sample_rate, 16000)
        self.assertEqual(asr.requests[0].channels, 1)
        self.assertEqual(tts.requests[0].audio_config.format, "mp3")
        self.assertEqual(tts.requests[0].audio_config.sample_rate, 24000)
        self.assertEqual(tts.requests[0].audio_config.channels, 2)

        legacy = gateway_main.VoiceCompletionRequest(
            request_id="legacy",
            trace_id="trace",
            session_id="legacy-session",
            audio_base64="UklGRg==",
            audio_config=AudioConfig(sample_rate=22050),
        )
        self.assertEqual(legacy.input_audio_config.sample_rate, 22050)
        self.assertEqual(legacy.output_audio_config.sample_rate, 22050)

        with self.assertRaises(ValidationError):
            gateway_main.VoiceCompletionRequest(
                request_id="conflict",
                trace_id="trace",
                session_id="legacy-session",
                audio_base64="UklGRg==",
                audio_config=AudioConfig(sample_rate=22050),
                output_audio_config=AudioConfig(sample_rate=24000),
            )

    def test_gateway_keeps_bounded_history_per_session(self) -> None:
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
                    json={
                        "request_id": request_id,
                        "trace_id": "trace",
                        "session_id": session_id,
                        "message": {"role": "user", "content": content},
                        "stream": False,
                    },
                )
                self.assertEqual(response.status_code, 200)
        finally:
            gateway_main.orchestrator.llm_provider = previous

        self.assertEqual(
            [message.content for message in llm.requests[1].messages],
            ["You are a helpful voice assistant.", "first", "回复：first", "second"],
        )
        self.assertEqual(
            [message.content for message in llm.requests[2].messages],
            ["You are a helpful voice assistant.", "isolated"],
        )

    def test_gateway_voice_stream_persists_assistant_history(self) -> None:
        llm = FakeVoiceStreamingLLMProvider()
        previous = (
            gateway_main.orchestrator.asr_provider,
            gateway_main.orchestrator.llm_provider,
        )
        gateway_main.orchestrator.asr_provider = FakeASRProvider()
        gateway_main.orchestrator.llm_provider = llm
        client = TestClient(gateway_main.app)
        try:
            for request_id in ("stream-one", "stream-two"):
                response = client.post(
                    "/api/v1/voice/completions/stream",
                    json={
                        "request_id": request_id,
                        "trace_id": "trace",
                        "session_id": "stream-session",
                        "audio_base64": "UklGRg==",
                        "output_audio": False,
                    },
                )
                self.assertEqual(response.status_code, 200)
        finally:
            (
                gateway_main.orchestrator.asr_provider,
                gateway_main.orchestrator.llm_provider,
            ) = previous

        self.assertEqual(
            [message.content for message in llm.requests[1].messages],
            [
                "You are a helpful voice assistant.",
                "你好",
                "First sentence.Second sentence!",
                "你好",
            ],
        )

    def test_gateway_voice_stream_interleaves_text_and_sentence_audio(self) -> None:
        streaming_tts = FakeVoiceStreamingTTSProvider()
        previous = (
            gateway_main.orchestrator.asr_provider,
            gateway_main.orchestrator.llm_provider,
            gateway_main.orchestrator.tts_provider,
        )
        gateway_main.orchestrator.asr_provider = FakeASRProvider()
        gateway_main.orchestrator.llm_provider = FakeVoiceStreamingLLMProvider()
        gateway_main.orchestrator.tts_provider = streaming_tts
        try:
            response = TestClient(gateway_main.app).post(
                "/api/v1/voice/completions/stream",
                json={
                    "request_id": "r",
                    "trace_id": "t",
                    "session_id": "s",
                    "audio_base64": "UklGRg==",
                    "output_audio": True,
                    "voice": {"mode": "custom", "speaker_id": "nana", "language": "Chinese"},
                },
            )
        finally:
            (
                gateway_main.orchestrator.asr_provider,
                gateway_main.orchestrator.llm_provider,
                gateway_main.orchestrator.tts_provider,
            ) = previous

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(streaming_tts.texts, ["First sentence.", "Second sentence!"])
        self.assertEqual(response.text.count("event: audio.started"), 1)
        self.assertEqual(response.text.count("event: audio.delta"), 2)
        self.assertEqual(response.text.count("event: audio.done"), 1)
        expected_order = (
            "event: asr.started",
            "event: asr.completed",
            "event: message.started",
            "event: message.delta",
            "event: audio.started",
            "event: audio.delta",
            "event: message.done",
            "event: audio.done",
        )
        positions = [response.text.index(event) for event in expected_order]
        self.assertEqual(positions, sorted(positions))


class OrchestratorContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_maps_asr_details_to_frontend_response(self) -> None:
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


class ConversationStoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_store_keeps_complete_turns_and_evicts_old_sessions(self) -> None:
        store = InMemoryConversationStore(max_sessions=1, max_messages_per_session=3)
        async with store.session("first") as conversation:
            conversation.append_turn(
                Message(role=MessageRole.USER, content="one"),
                Message(role=MessageRole.ASSISTANT, content="reply one"),
            )
            conversation.append_turn(
                Message(role=MessageRole.USER, content="two"),
                Message(role=MessageRole.ASSISTANT, content="reply two"),
            )
            self.assertEqual(
                [message.content for message in conversation.history],
                ["two", "reply two"],
            )

        async with store.session("second") as conversation:
            self.assertEqual(list(conversation.history), [])

        async with store.session("first") as conversation:
            self.assertEqual(list(conversation.history), [])


if __name__ == "__main__":
    unittest.main()
