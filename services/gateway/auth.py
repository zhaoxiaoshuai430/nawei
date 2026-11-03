"""Configurable local or HS256 JWT authentication for Gateway management APIs."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import time
from typing import Any

from fastapi import Header, HTTPException


def authenticate_user(
    authorization: str | None = Header(None, alias="Authorization"),
    x_user_id: str | None = Header(None, alias="X-User-ID"),
) -> str:
    """Return a verified user id using the configured Gateway auth mode."""

    mode = os.getenv("GATEWAY_AUTH_MODE", "local").strip().lower()
    if mode == "local":
        _reject_local_auth_in_production()
        return _validate_local_user(x_user_id)
    if mode == "jwt":
        return _authenticate_jwt(authorization)
    raise HTTPException(status_code=503, detail=f"Unsupported GATEWAY_AUTH_MODE: {mode}")


def authorize_request_user(
    request_user_id: str,
    authorization: str | None,
    x_user_id: str | None,
) -> str:
    """Verify body user ownership while preserving header-optional local development."""

    mode = os.getenv("GATEWAY_AUTH_MODE", "local").strip().lower()
    if mode == "local":
        _reject_local_auth_in_production()
        if not x_user_id:
            if _env_flag("GATEWAY_ALLOW_BODY_USER_ID", False):
                return _validate_local_user(request_user_id)
            raise HTTPException(status_code=400, detail="X-User-ID is required in local authentication mode.")
    authenticated_user_id = authenticate_user(authorization, x_user_id)
    if request_user_id != authenticated_user_id:
        raise HTTPException(status_code=403, detail="Request user does not match the authenticated user.")
    return authenticated_user_id


def _validate_local_user(user_id: str | None) -> str:
    value = (user_id or "").strip()
    if not value or value == "anonymous":
        raise HTTPException(status_code=400, detail="A persistent user identity is required.")
    return value


def _reject_local_auth_in_production() -> None:
    environment = os.getenv("NAWEI_ENV", "development").strip().lower()
    if environment in {"production", "prod"}:
        raise HTTPException(status_code=503, detail="GATEWAY_AUTH_MODE=local is not allowed in production.")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _authenticate_jwt(authorization: str | None) -> str:
    secret = os.getenv("GATEWAY_JWT_SECRET", "")
    if len(secret.encode("utf-8")) < 32:
        raise HTTPException(status_code=503, detail="GATEWAY_JWT_SECRET must contain at least 32 bytes.")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="A Bearer token is required.")

    token = authorization[7:].strip()
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        header = _decode_json_segment(encoded_header)
        payload = _decode_json_segment(encoded_payload)
        signature = _decode_segment(encoded_signature)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid Bearer token.") from exc

    if header.get("alg") != "HS256" or header.get("typ", "JWT") != "JWT":
        raise HTTPException(status_code=401, detail="Only HS256 JWT tokens are accepted.")

    signed = f"{encoded_header}.{encoded_payload}".encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid Bearer token signature.")

    _validate_registered_claims(payload)
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject.strip() or subject.strip() == "anonymous":
        raise HTTPException(status_code=401, detail="JWT subject must be a persistent user id.")
    return subject.strip()


def _validate_registered_claims(payload: dict[str, Any]) -> None:
    now = time.time()
    try:
        expires_at = float(payload["exp"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=401, detail="JWT exp claim is required.") from exc
    if not math.isfinite(expires_at) or expires_at <= now:
        raise HTTPException(status_code=401, detail="Bearer token has expired.")

    not_before = payload.get("nbf")
    if not_before is not None:
        try:
            not_before_value = float(not_before)
            if not math.isfinite(not_before_value):
                raise HTTPException(status_code=401, detail="Invalid JWT nbf claim.")
            if not_before_value > now:
                raise HTTPException(status_code=401, detail="Bearer token is not active yet.")
        except (TypeError, ValueError, OverflowError) as exc:
            raise HTTPException(status_code=401, detail="Invalid JWT nbf claim.") from exc

    configured_issuer = os.getenv("GATEWAY_JWT_ISSUER", "").strip()
    if configured_issuer and payload.get("iss") != configured_issuer:
        raise HTTPException(status_code=401, detail="Invalid JWT issuer.")

    configured_audience = os.getenv("GATEWAY_JWT_AUDIENCE", "").strip()
    if configured_audience and not _audience_contains(payload.get("aud"), configured_audience):
        raise HTTPException(status_code=401, detail="Invalid JWT audience.")


def _audience_contains(claim: Any, expected: str) -> bool:
    # Audience identifiers are public Unicode strings, not secret MAC values.
    # compare_digest(str, str) rejects non-ASCII claims with TypeError.
    if isinstance(claim, str):
        return claim == expected
    if isinstance(claim, list):
        return any(isinstance(item, str) and item == expected for item in claim)
    return False


def _decode_json_segment(value: str) -> dict[str, Any]:
    body = json.loads(_decode_segment(value))
    if not isinstance(body, dict):
        raise ValueError("JWT segment must be an object")
    return body


def _decode_segment(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


_PUBLIC_AGE_KEYS = frozenset({"age", "user_age", "birth_date", "birthday", "date_of_birth", "grade", "age_years", "age_stage"})


def public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep client metadata from overriding server-owned audience identity."""
    return {
        key: public_metadata(value) if isinstance(value, dict) else value
        for key, value in metadata.items()
        if key not in _PUBLIC_AGE_KEYS
    }
