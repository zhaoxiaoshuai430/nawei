"""ASR protocol helpers and SenseVoice tag parser tests."""

from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from asr_protocol import decode_base64_audio, streaming_capabilities  # noqa: E402
from tags import parse_sensevoice_raw  # noqa: E402


class ASRProtocolTest(unittest.TestCase):
    def test_base64_decoder_accepts_whitespace_and_missing_padding(self) -> None:
        encoded = base64.b64encode(b"RIFF-test-audio").decode("ascii").rstrip("=")
        spaced = f"  {encoded[:5]}\n{encoded[5:]}  "
        self.assertEqual(decode_base64_audio(spaced), b"RIFF-test-audio")

    def test_base64_decoder_rejects_invalid_or_empty_audio(self) -> None:
        for value in ("", "%%%"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_base64_audio(value)

    def test_base64_decoder_rejects_audio_over_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "upload limit"):
            decode_base64_audio("QUJD", max_bytes=2)

    def test_streaming_capability_does_not_claim_realtime_decoder_state(self) -> None:
        capabilities = streaming_capabilities()
        self.assertFalse(capabilities["realtime_streaming"])
        self.assertFalse(capabilities["keeps_decoder_state"])
        self.assertEqual(capabilities["streaming_mode"], "independent-chunk")

    def test_sensevoice_tags_include_language_emotion_and_events(self) -> None:
        parsed = parse_sensevoice_raw("<|zh|><|HAPPY|><|Laughter|>你好")
        self.assertEqual(parsed["language"], "中文")
        self.assertEqual(parsed["emotion"], "HAPPY")
        self.assertEqual(parsed["events"][0]["tag"], "Laughter")
        self.assertIn("开心", parsed["emotion_hint"])


if __name__ == "__main__":
    unittest.main()
