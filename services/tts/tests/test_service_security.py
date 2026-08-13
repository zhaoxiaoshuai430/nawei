"""Standalone TTS security configuration tests."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.service_security import require_tts_internal_api_key, tts_cors_origins


def test_production_requires_strong_internal_key() -> None:
    with patch.dict(os.environ, {"NAWEI_ENV": "production"}, clear=True), pytest.raises(HTTPException) as raised:
        require_tts_internal_api_key(None)
    assert raised.value.status_code == 503

    env = {"NAWEI_ENV": "production", "TTS_INTERNAL_API_KEY": "short"}
    with patch.dict(os.environ, env, clear=True), pytest.raises(HTTPException) as raised:
        require_tts_internal_api_key("short")
    assert raised.value.status_code == 503


def test_configured_internal_key_is_verified() -> None:
    env = {"TTS_INTERNAL_API_KEY": "a" * 32}
    with patch.dict(os.environ, env, clear=True), pytest.raises(HTTPException) as raised:
        require_tts_internal_api_key("wrong")
    assert raised.value.status_code == 401


def test_production_cors_rejects_wildcard() -> None:
    env = {"NAWEI_ENV": "production", "TTS_CORS_ORIGINS": "*"}
    with patch.dict(os.environ, env, clear=True), pytest.raises(RuntimeError):
        tts_cors_origins()
