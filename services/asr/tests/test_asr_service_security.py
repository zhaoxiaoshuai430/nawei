"""ASR service authentication boundary tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

try:
    from services.asr.src.service_security import asr_cors_origins, verify_asr_internal_api_key
except ModuleNotFoundError:
    from src.service_security import asr_cors_origins, verify_asr_internal_api_key


class ASRServiceSecurityTests(unittest.TestCase):
    def test_development_allows_missing_internal_key(self) -> None:
        with patch.dict(os.environ, {"NAWEI_ENV": "development"}, clear=True):
            verify_asr_internal_api_key(None)

    def test_production_requires_internal_key(self) -> None:
        with patch.dict(os.environ, {"NAWEI_ENV": "production"}, clear=True):
            with self.assertRaises(HTTPException) as captured:
                verify_asr_internal_api_key(None)

        self.assertEqual(captured.exception.status_code, 503)

    def test_configured_key_must_match(self) -> None:
        environment = {"NAWEI_ENV": "production", "ASR_INTERNAL_API_KEY": "a" * 32}
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(HTTPException) as captured:
                verify_asr_internal_api_key("wrong")
            verify_asr_internal_api_key("a" * 32)

        self.assertEqual(captured.exception.status_code, 401)

    def test_production_rejects_wildcard_cors(self) -> None:
        environment = {"NAWEI_ENV": "production", "ASR_CORS_ORIGINS": "*"}
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "cannot contain"):
                asr_cors_origins()


if __name__ == "__main__":
    unittest.main()
