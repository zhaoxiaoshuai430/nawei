"""TTS timing metric regression tests."""

from __future__ import annotations

import unittest

try:
    from services.tts.app.services.tts.streaming import real_time_factor
except ModuleNotFoundError:
    from app.services.tts.streaming import real_time_factor


class TTSMetricsTests(unittest.TestCase):
    def test_real_time_factor_uses_generation_over_audio_duration(self) -> None:
        self.assertEqual(real_time_factor(generation_ms=2000, audio_seconds=4), 0.5)
        self.assertEqual(real_time_factor(generation_ms=4000, audio_seconds=2), 2.0)

    def test_real_time_factor_handles_empty_measurements(self) -> None:
        self.assertEqual(real_time_factor(generation_ms=0, audio_seconds=2), 0.0)
        self.assertEqual(real_time_factor(generation_ms=1000, audio_seconds=0), 0.0)


if __name__ == "__main__":
    unittest.main()
