"""Gateway realtime voice protocol tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from services.gateway import main as gateway_main


class GatewayRealtimeTest(unittest.TestCase):
    def test_realtime_start_builds_pcm_voice_request(self) -> None:
        start = gateway_main._parse_realtime_voice_start(self._start_payload())
        request = gateway_main._realtime_voice_request(start)

        self.assertEqual(request.input_audio_config.format, "pcm16")
        self.assertEqual(request.input_audio_config.sample_rate, 16000)
        self.assertEqual(request.user_id, "user-a")
        self.assertTrue(request.output_audio)

    def test_transcript_join_preserves_chinese_and_spaces_english(self) -> None:
        self.assertEqual(gateway_main._join_realtime_transcripts(["你好，", "世界！"]), "你好，世界！")
        self.assertEqual(gateway_main._join_realtime_transcripts(["hello", "world"]), "hello world")

    def test_websocket_reuses_downstream_completion(self) -> None:
        async def complete(websocket, request, segments) -> None:
            await websocket.send_json({"event": "test.done", "data": {"text": segments[0]["text"]}})

        env = {"GATEWAY_AUTH_MODE": "local"}
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(
                gateway_main,
                "_relay_realtime_asr",
                new=AsyncMock(return_value=[{"text": "你好", "language": "zh", "duration_ms": 500}]),
            ) as relay,
            patch.object(gateway_main, "_complete_realtime_voice_turn", new=complete),
            TestClient(gateway_main.app) as client,
        ):
            with client.websocket_connect("/api/v1/voice/realtime") as websocket:
                websocket.send_json(self._start_payload())
                event = websocket.receive_json()

        self.assertEqual(event["event"], "test.done")
        self.assertEqual(event["data"]["text"], "你好")
        relay.assert_awaited_once()

    @staticmethod
    def _start_payload() -> dict[str, object]:
        return {
            "type": "start",
            "request_id": "request-1",
            "trace_id": "trace-1",
            "session_id": "session-1",
            "user_id": "user-a",
            "language": "auto",
            "output_audio": True,
            "voice": {"mode": "custom", "speaker_id": "Nana", "language": "Chinese"},
            "output_audio_config": {"format": "wav", "sample_rate": 24000, "channels": 1},
            "metadata": {"use_itn": True},
        }


if __name__ == "__main__":
    unittest.main()
