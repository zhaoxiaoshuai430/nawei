"""Realtime ASR endpointing regression tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

try:
    from services.asr.src.realtime import PCM16VadSegmenter, RealtimeVadConfig
except ModuleNotFoundError:
    from src.realtime import PCM16VadSegmenter, RealtimeVadConfig


class RealtimeVadTests(unittest.TestCase):
    def test_default_endpoint_silence_is_low_latency(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = RealtimeVadConfig.from_env()

        self.assertEqual(config.end_silence_ms, 420)
        self.assertEqual(config.end_silence_frames, 21)

    def test_segment_finalizes_after_configured_silence(self) -> None:
        config = RealtimeVadConfig(
            frame_ms=20,
            start_ms=40,
            end_silence_ms=60,
            pre_roll_ms=40,
            min_speech_ms=40,
            rms_threshold=100,
        )
        segmenter = PCM16VadSegmenter(config)
        voiced_frame = (1000).to_bytes(2, "little", signed=True) * 320
        silent_frame = b"\x00\x00" * 320

        self.assertEqual(segmenter.feed(voiced_frame), [])
        self.assertEqual(segmenter.feed(voiced_frame)[0].kind, "speech_started")
        self.assertEqual(segmenter.feed(silent_frame), [])
        self.assertEqual(segmenter.feed(silent_frame), [])
        events = segmenter.feed(silent_frame)

        self.assertEqual([event.kind for event in events], ["segment"])


if __name__ == "__main__":
    unittest.main()
