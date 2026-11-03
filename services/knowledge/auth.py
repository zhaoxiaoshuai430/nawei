"""Authentication boundaries for knowledge retrieval and administration."""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException


def authorize_knowledge_internal_request(
    x_internal_api_key: str | None = Header(None, alias="X-Internal-API-Key"),
) -> None:
    _authorize("KNOWLEDGE_INTERNAL_API_KEY", x_internal_api_key)


def authorize_knowledge_admin_request(
    x_admin_api_key: str | None = Header(None, alias="X-Admin-API-Key"),
) -> None:
    _authorize("KNOWLEDGE_ADMIN_API_KEY", x_admin_api_key)


def _authorize(name: str, provided: str | None) -> None:
    configured = os.getenv(name, "").strip()
    production = os.getenv("NAWEI_ENV", "development").strip().lower() in {"production", "prod"}
    if not configured:
        if production:
            raise HTTPException(status_code=503, detail=f"{name} is required in production.")
        return
    if production and len(configured.encode("utf-8")) < 32:
        raise HTTPException(status_code=503, detail=f"{name} must contain at least 32 bytes.")
    if not provided or not secrets.compare_digest(provided.encode("utf-8"), configured.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid knowledge API key.")
