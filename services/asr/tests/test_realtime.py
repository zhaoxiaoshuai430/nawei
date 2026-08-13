"""Realtime PCM segmentation tests."""

from __future__ import annotations

import array
import unittest

from realtime import PCM16VadSegmenter, RealtimeVadConfig


def _frame(amplitude: int, samples: int) -> bytes:
    return array.array("h", [amplitude] * samples).tobytes()


class PCM16VadSegmenterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RealtimeVadConfig(
            sample_rate=1000,
            frame_ms=20,
            start_ms=40,
            end_silence_ms=40,
            pre_roll_ms=20,
            min_speech_ms=40,
            max_segment_seconds=2,
            rms_threshold=100,
        )

    def test_segments_voiced_audio_after_silence(self) -> None:
        segmenter = PCM16VadSegmenter(self.config)
        samples = self.config.sample_rate * self.config.frame_ms // 1000
        silence = _frame(0, samples)
        speech = _frame(1000, samples)

        self.assertEqual(segmenter.feed(silence), [])
        self.assertEqual(segmenter.feed(speech), [])
        started = segmenter.feed(speech)
        self.assertEqual([event.kind for event in started], ["speech_started"])
        self.assertTrue(segmenter.in_speech)

        self.assertEqual(segmenter.feed(silence), [])
        finished = segmenter.feed(silence)
        self.assertEqual([event.kind for event in finished], ["segment"])
        self.assertGreaterEqual(len(finished[0].pcm), self.config.min_segment_bytes)
        self.assertFalse(segmenter.in_speech)

    def test_flush_returns_active_utterance(self) -> None:
        segmenter = PCM16VadSegmenter(self.config)
        samples = self.config.sample_rate * self.config.frame_ms // 1000
        speech = _frame(1200, samples)
        segmenter.feed(speech)
        segmenter.feed(speech)

        events = segmenter.flush()

        self.assertEqual([event.kind for event in events], ["segment"])

    def test_rejects_odd_pcm_payload(self) -> None:
        segmenter = PCM16VadSegmenter(self.config)
        with self.assertRaises(ValueError):
            segmenter.feed(b"\x00")


if __name__ == "__main__":
    unittest.main()
