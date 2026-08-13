"""Standalone ASR security configuration tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from service_security import asr_cors_origins, require_asr_internal_api_key, verify_asr_internal_api_key


class ASRSecurityTest(unittest.TestCase):
    def test_production_requires_strong_internal_key(self) -> None:
        with patch.dict(os.environ, {"NAWEI_ENV": "production"}, clear=True), self.assertRaises(HTTPException) as raised:
            require_asr_internal_api_key(None)
        self.assertEqual(raised.exception.status_code, 503)

        env = {"NAWEI_ENV": "production", "ASR_INTERNAL_API_KEY": "short"}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(HTTPException) as raised:
            require_asr_internal_api_key("short")
        self.assertEqual(raised.exception.status_code, 503)

    def test_configured_internal_key_is_verified(self) -> None:
        env = {"ASR_INTERNAL_API_KEY": "a" * 32}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(HTTPException) as raised:
            require_asr_internal_api_key("wrong")
        self.assertEqual(raised.exception.status_code, 401)

    def test_websocket_key_uses_same_validation(self) -> None:
        env = {"ASR_INTERNAL_API_KEY": "b" * 32}
        with patch.dict(os.environ, env, clear=True):
            verify_asr_internal_api_key("b" * 32)
            with self.assertRaises(HTTPException) as raised:
                verify_asr_internal_api_key("wrong")
        self.assertEqual(raised.exception.status_code, 401)

    def test_production_cors_rejects_wildcard(self) -> None:
        env = {"NAWEI_ENV": "production", "ASR_CORS_ORIGINS": "*"}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(RuntimeError):
            asr_cors_origins()
