"""TTS service authentication boundary tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

try:
    from services.tts.app.service_security import require_tts_internal_api_key, tts_cors_origins
except ModuleNotFoundError:
    from app.service_security import require_tts_internal_api_key, tts_cors_origins


class TTSServiceSecurityTests(unittest.TestCase):
    def test_development_allows_missing_internal_key(self) -> None:
        with patch.dict(os.environ, {"NAWEI_ENV": "development"}, clear=True):
            require_tts_internal_api_key(None)

    def test_production_requires_internal_key(self) -> None:
        with patch.dict(os.environ, {"NAWEI_ENV": "production"}, clear=True):
            with self.assertRaises(HTTPException) as captured:
                require_tts_internal_api_key(None)

        self.assertEqual(captured.exception.status_code, 503)

    def test_configured_key_must_match(self) -> None:
        environment = {"NAWEI_ENV": "production", "TTS_INTERNAL_API_KEY": "t" * 32}
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(HTTPException) as captured:
                require_tts_internal_api_key("wrong")
            require_tts_internal_api_key("t" * 32)

        self.assertEqual(captured.exception.status_code, 401)

    def test_production_rejects_wildcard_cors(self) -> None:
        environment = {"NAWEI_ENV": "production", "TTS_CORS_ORIGINS": "*"}
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "cannot contain"):
                tts_cors_origins()


if __name__ == "__main__":
    unittest.main()
