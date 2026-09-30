import asyncio
import base64
import json

import httpx
import pytest

from app_protocols import ASRRequest
from scripts import warm_voice


async def test_warmup_exercises_http_and_all_streaming_workers(monkeypatch):
    pcm = bytes(48000)
    requests = []
    streams = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/api/v1/audio/speech":
            assert request.headers["X-Internal-API-Key"] == "tts-test"
            return httpx.Response(200, json={"format": "pcm16", "sample_rate": 16000, "channels": 1,
                                            "audio_base64": base64.b64encode(pcm).decode()})
        if request.url.path == "/api/v1/audio/transcriptions/base64":
            assert request.headers["X-Internal-API-Key"] == "asr-test"
            asr_request = ASRRequest.model_validate_json(request.content)
            assert asr_request.options == {"use_itn": True}
            assert base64.b64decode(asr_request.audio_base64) == pcm
            return httpx.Response(200, json={"text": "recognized"})
        assert request.url.path == "/health"
        return httpx.Response(200, json={"streaming_concurrency": {"limit": 2}})

    class Socket:
        def __init__(self, url):
            self.audio = bytearray()
            self.closed = False
            streams.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

        async def send(self, message):
            if isinstance(message, bytes):
                assert len(streams) == 2
                assert len(message) <= 32000
                self.audio.extend(message)
            elif json.loads(message)["type"] == "start":
                assert json.loads(message)["internal_api_key"] == "asr-test"

        async def recv(self):
            return json.dumps({"type": "ready"})

        async def __aiter__(self):
            yield json.dumps({"type": "transcript.final", "text": "recognized"})
            yield json.dumps({"type": "session.done"})

    monkeypatch.setattr(warm_voice, "connect", Socket)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with asyncio.timeout(2):
            result = await warm_voice.warm_voice(client, {"TTS_INTERNAL_API_KEY": "tts-test", "ASR_INTERNAL_API_KEY": "asr-test"})
    assert result["ready"] and result["streaming_workers"] == 2
    assert all(socket.closed and socket.audio == pcm for socket in streams)
    assert len(requests) == 3


@pytest.mark.parametrize("audio", [
    {"format": "wav", "sample_rate": 16000, "audio_base64": ""},
    {"format": "pcm16", "sample_rate": 16000, "audio_base64": ""},
    {"format": "pcm16", "sample_rate": 16000, "audio_base64": "AA=="},
])
async def test_invalid_warmup_audio_does_not_report_ready(audio):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=audio))) as client:
        with pytest.raises(ValueError):
            await warm_voice.warm_voice(client, {})
