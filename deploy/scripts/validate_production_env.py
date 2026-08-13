"""Fail fast when the four-service production chain is misconfigured."""

from __future__ import annotations

import os
from urllib.parse import urlparse


SECRET_KEYS = (
    "GATEWAY_JWT_SECRET",
    "ASR_INTERNAL_API_KEY",
    "LLM_INTERNAL_API_KEY",
    "TTS_INTERNAL_API_KEY",
    "LLM_API_KEY",
    "POSTGRES_PASSWORD",
)
INTERNAL_KEYS = (
    "ASR_INTERNAL_API_KEY",
    "LLM_INTERNAL_API_KEY",
    "TTS_INTERNAL_API_KEY",
)
SERVICE_URL_KEYS = (
    "GATEWAY_ASR_URL",
    "GATEWAY_ASR_REALTIME_URL",
    "GATEWAY_LLM_URL",
    "GATEWAY_LLM_STREAM_URL",
    "GATEWAY_TTS_URL",
    "GATEWAY_TTS_STREAM_URL",
    "DATABASE_URL",
    "REDIS_URL",
)


def validate_environment() -> list[str]:
    errors: list[str] = []
    _require_exact("NAWEI_ENV", "production", errors)
    _require_exact("GATEWAY_AUTH_MODE", "jwt", errors)
    _require_exact("LLM_PROVIDER", "deepseek", errors)
    _require_exact("LLM_AGENT_ENABLED", "true", errors)
    _require_exact("MEMORY_ENABLED", "true", errors)
    _require_exact("LLM_MEMORY_REQUIRED", "true", errors)

    for key in SECRET_KEYS:
        value = os.getenv(key, "")
        if not value or value.startswith("REPLACE_WITH_"):
            errors.append(f"{key} must be configured with a real value.")
        elif key != "LLM_API_KEY" and len(value.encode("utf-8")) < 32:
            errors.append(f"{key} must contain at least 32 bytes.")

    internal_values = [os.getenv(key, "") for key in INTERNAL_KEYS]
    if len(set(internal_values)) != len(internal_values):
        errors.append("ASR, LLM, and TTS internal API keys must be distinct.")

    for key in SERVICE_URL_KEYS:
        value = os.getenv(key, "")
        if not value:
            errors.append(f"{key} is required.")
            continue
        hostname = urlparse(value).hostname
        if hostname in {"127.0.0.1", "localhost", "0.0.0.0"}:
            errors.append(f"{key} must use its Docker service name, not {hostname}.")

    return errors


def _require_exact(name: str, expected: str, errors: list[str]) -> None:
    if os.getenv(name, "").strip().lower() != expected:
        errors.append(f"{name} must be {expected}.")


def main() -> None:
    errors = validate_environment()
    if errors:
        raise SystemExit("Invalid production environment:\n- " + "\n- ".join(errors))
    print("Production environment is valid for Gateway -> ASR -> LLM -> TTS.")


if __name__ == "__main__":
    main()
