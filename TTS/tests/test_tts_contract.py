from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import unittest
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.tts import state
from app.services.gateway import handlers as tts_handlers
from app.web import tts_routes


class FakeTTSModel:
    def generate_custom_voice(self, **kwargs):
        _ = kwargs
        samples = np.sin(2 * np.pi * 440 * np.arange(2400) / 24000).astype(np.float32) * 0.1
        return [samples], 24000


class SlowStreamingTTSModel(FakeTTSModel):
    def generate_custom_voice_streaming(self, **kwargs):
        _ = kwargs
        chunk = np.zeros(240, dtype=np.float32)
        timing = {"prefill_ms": 1, "decode_ms": 1, "codec_decode_ms": 0}
        for _ in range(3):
            time.sleep(0.1)
            yield chunk, 24000, timing


class TTSContractTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.previous_cache = state.model_cache
        self.previous_active = state.active_model_name
        state.model_cache = OrderedDict({"fake": FakeTTSModel()})
        state.active_model_name = "fake"

    async def asyncTearDown(self) -> None:
        state.model_cache = self.previous_cache
        state.active_model_name = self.previous_active

    async def test_standard_endpoint_encodes_requested_audio_config(self) -> None:
        cases = (
            ("wav", 16000, 1),
            ("mp3", 16000, 2),
            ("pcm16", 8000, 2),
        )
        for audio_format, sample_rate, channels in cases:
            with self.subTest(audio_format=audio_format):
                response = await tts_routes.tts_standard_generate(
                    tts_routes.TTSRequest(
                        request_id="r",
                        trace_id="t",
                        session_id="s",
                        text="hello",
                        stream=False,
                        voice=tts_routes.VoiceConfig(speaker_id="nana", language="Chinese"),
                        audio_config=tts_routes.AudioConfig(
                            format=audio_format,
                            sample_rate=sample_rate,
                            channels=channels,
                        ),
                    )
                )
                raw = base64.b64decode(response.audio_base64)
                self.assertEqual(response.format, audio_format)
                self.assertEqual(response.sample_rate, sample_rate)
                self.assertEqual(response.metadata["channels"], channels)
                if audio_format == "pcm16":
                    self.assertEqual(len(raw), round(0.1 * sample_rate) * channels * 2)
                else:
                    info = sf.info(io.BytesIO(raw))
                    self.assertEqual(info.samplerate, sample_rate)
                    self.assertEqual(info.channels, channels)
                    self.assertEqual(info.format, audio_format.upper())

    async def test_stream_close_waits_for_generation_worker_before_unlocking(self) -> None:
        state.model_cache["fake"] = SlowStreamingTTSModel()
        response = await tts_handlers.generate_stream(
            text="hello",
            language="Chinese",
            mode="custom",
            ref_text="",
            speaker="nana",
            instruct="",
            xvec_only=True,
            chunk_size=25,
            temperature=0.9,
            top_k=50,
            repetition_penalty=1.05,
            ref_preset="",
            ref_audio=None,
        )

        stream = response.body_iterator
        first = await anext(stream)
        self.assertIn('"type": "chunk"', first)
        self.assertTrue(state.generation_lock.locked())

        close_task = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0.01)
        self.assertTrue(state.generation_lock.locked())
        await close_task
        self.assertFalse(state.generation_lock.locked())


class ModelLoadSecurityTest(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(tts_routes.router)
        self.client = TestClient(app)

    def test_remote_model_load_requires_key_and_allowlisted_model(self) -> None:
        with patch.dict(os.environ, {"TTS_ADMIN_API_KEY": "test-secret"}, clear=False):
            denied = self.client.post("/api/v1/models/load", data={"model_id": "allowed"})
            self.assertEqual(denied.status_code, 403)

            with (
                patch.object(tts_routes, "list_available_models", return_value=["allowed"]),
                patch.object(tts_routes, "load_model_into_cache") as load_model,
            ):
                unknown = self.client.post(
                    "/api/v1/models/load",
                    data={"model_id": "unknown"},
                    headers={"X-TTS-Admin-Key": "test-secret"},
                )
                accepted = self.client.post(
                    "/api/v1/models/load",
                    data={"model_id": "allowed"},
                    headers={"X-TTS-Admin-Key": "test-secret"},
                )

            self.assertEqual(unknown.status_code, 400)
            self.assertEqual(accepted.status_code, 200)
            load_model.assert_called_once_with("allowed")


if __name__ == "__main__":
    unittest.main()
