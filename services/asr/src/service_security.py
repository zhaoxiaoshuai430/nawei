"""Security configuration for the standalone ASR service."""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException


def require_asr_internal_api_key(
    x_internal_api_key: str | None = Header(None, alias="X-Internal-API-Key"),
) -> None:
    verify_asr_internal_api_key(x_internal_api_key)


def verify_asr_internal_api_key(provided_key: str | None) -> None:
    """Validate an internal key supplied outside normal HTTP dependencies."""

    configured_key = os.getenv("ASR_INTERNAL_API_KEY", "").strip()
    environment = os.getenv("NAWEI_ENV", "development").strip().lower()
    if not configured_key:
        if environment in {"production", "prod"}:
            raise HTTPException(status_code=503, detail="ASR_INTERNAL_API_KEY is required in production.")
        return
    if environment in {"production", "prod"} and len(configured_key.encode("utf-8")) < 32:
        raise HTTPException(status_code=503, detail="ASR_INTERNAL_API_KEY must contain at least 32 bytes.")
    if not provided_key or not secrets.compare_digest(provided_key, configured_key):
        raise HTTPException(status_code=401, detail="Invalid internal API key.")


def asr_cors_origins() -> list[str]:
    origins = [origin.strip() for origin in os.getenv("ASR_CORS_ORIGINS", "").split(",") if origin.strip()]
    environment = os.getenv("NAWEI_ENV", "development").strip().lower()
    if environment in {"production", "prod"} and "*" in origins:
        raise RuntimeError("ASR_CORS_ORIGINS cannot contain '*' in production.")
    return origins
