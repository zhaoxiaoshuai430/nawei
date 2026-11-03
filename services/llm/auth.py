"""Internal authentication boundary for the LLM service."""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException


def authorize_llm_internal_request(
    x_internal_api_key: str | None = Header(None, alias="X-Internal-API-Key"),
) -> None:
    configured_key = os.getenv("LLM_INTERNAL_API_KEY", "").strip()
    environment = os.getenv("NAWEI_ENV", "development").strip().lower()
    if not configured_key:
        if environment in {"production", "prod"}:
            raise HTTPException(status_code=503, detail="LLM_INTERNAL_API_KEY is required in production.")
        return
    if environment in {"production", "prod"} and len(configured_key.encode("utf-8")) < 32:
        raise HTTPException(status_code=503, detail="LLM_INTERNAL_API_KEY must contain at least 32 bytes.")
    if not x_internal_api_key or not secrets.compare_digest(x_internal_api_key.encode("utf-8"), configured_key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid internal API key.")
