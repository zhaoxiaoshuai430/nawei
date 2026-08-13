"""Security configuration for the standalone TTS service."""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException


def require_tts_internal_api_key(
    x_internal_api_key: str | None = Header(None, alias="X-Internal-API-Key"),
) -> None:
    configured_key = os.getenv("TTS_INTERNAL_API_KEY", "").strip()
    environment = os.getenv("NAWEI_ENV", "development").strip().lower()
    if not configured_key:
        if environment in {"production", "prod"}:
            raise HTTPException(status_code=503, detail="TTS_INTERNAL_API_KEY is required in production.")
        return
    if environment in {"production", "prod"} and len(configured_key.encode("utf-8")) < 32:
        raise HTTPException(status_code=503, detail="TTS_INTERNAL_API_KEY must contain at least 32 bytes.")
    if not x_internal_api_key or not secrets.compare_digest(x_internal_api_key, configured_key):
        raise HTTPException(status_code=401, detail="Invalid internal API key.")


def tts_cors_origins() -> list[str]:
    origins = [origin.strip() for origin in os.getenv("TTS_CORS_ORIGINS", "").split(",") if origin.strip()]
    environment = os.getenv("NAWEI_ENV", "development").strip().lower()
    if environment in {"production", "prod"} and "*" in origins:
        raise RuntimeError("TTS_CORS_ORIGINS cannot contain '*' in production.")
    return origins
