"""Preset asset download safety regression tests."""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TTS_ROOT = Path(__file__).resolve().parents[1]
if str(TTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TTS_ROOT))

presets = importlib.import_module("app.services.tts.presets")


class _Response:
    def __init__(self, content: bytes, *, content_length: int | None = None) -> None:
        self._content = content
        self._offset = 0
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._content):
            return b""
        end = len(self._content) if size < 0 else min(self._offset + size, len(self._content))
        chunk = self._content[self._offset:end]
        self._offset = end
        return chunk


class TTSPresetDownloadTests(unittest.TestCase):
    def test_https_download_is_validated_and_atomically_installed(self) -> None:
        wav = b"RIFF\x04\x00\x00\x00WAVEdata"
        observed_timeout: list[float] = []

        def open_response(request, *, timeout: float):
            self.assertEqual(request.full_url, "https://example.test/ref.wav")
            observed_timeout.append(timeout)
            return _Response(wav, content_length=len(wav))

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ref.wav"
            with patch.object(presets, "urlopen", open_response):
                presets._download_asset("https://example.test/ref.wav", destination, max_bytes=1024)

            self.assertEqual(destination.read_bytes(), wav)
            self.assertTrue(presets._valid_wav(destination))
            self.assertEqual(observed_timeout, [presets.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS])
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_oversized_download_keeps_existing_asset(self) -> None:
        existing = b"RIFF\x04\x00\x00\x00WAVEold"
        oversized = b"RIFF\x04\x00\x00\x00WAVE" + b"x" * 128

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ref.wav"
            destination.write_bytes(existing)
            with patch.object(presets, "urlopen", return_value=_Response(oversized)):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    presets._download_asset("https://example.test/ref.wav", destination, max_bytes=32)

            self.assertEqual(destination.read_bytes(), existing)
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_plain_http_download_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ref.wav"
            with patch.object(presets, "urlopen") as mocked_open:
                with self.assertRaisesRegex(ValueError, "HTTPS"):
                    presets._download_asset("http://example.test/ref.wav", destination, max_bytes=1024)
            mocked_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
