"""Fail fast when the five-service production chain is misconfigured."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from urllib.parse import urlparse


SECRET_KEYS = (
    "GATEWAY_JWT_SECRET",
    "ASR_INTERNAL_API_KEY",
    "LLM_INTERNAL_API_KEY",
    "TTS_INTERNAL_API_KEY",
    "TTS_ADMIN_API_KEY",
    "LLM_API_KEY",
    "POSTGRES_PASSWORD",
    "QDRANT_API_KEY",
    "NEO4J_PASSWORD",
    "KNOWLEDGE_INTERNAL_API_KEY",
    "KNOWLEDGE_ADMIN_API_KEY",
)
INTERNAL_KEYS = (
    "ASR_INTERNAL_API_KEY",
    "LLM_INTERNAL_API_KEY",
    "TTS_INTERNAL_API_KEY",
    "TTS_ADMIN_API_KEY",
    "KNOWLEDGE_INTERNAL_API_KEY",
    "KNOWLEDGE_ADMIN_API_KEY",
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
    "QDRANT_URL",
    "NEO4J_URI",
)


def validate_environment(environ: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    errors: list[str] = []
    required = {
        "NAWEI_ENV": "production", "GATEWAY_AUTH_MODE": "jwt", "GATEWAY_ENABLE_DIAGNOSTICS": "false",
        "LLM_PROVIDER": "deepseek", "LLM_AGENT_ENABLED": "true", "MEMORY_ENABLED": "true",
        "LLM_MEMORY_REQUIRED": "true", "KNOWLEDGE_ENABLED": "true", "KNOWLEDGE_REQUIRED": "true",
        "XIAOZHI_ENABLED": "true", "LLM_HOST": "0.0.0.0",
    }
    for name, expected in required.items():
        if env.get(name, "").strip().lower() != expected:
            errors.append(f"{name} must be {expected}.")

    provider = env.get("EMBEDDING_PROVIDER", "").strip().lower()
    secret_keys = (*SECRET_KEYS, "EMBEDDING_API_KEY") if provider in {"openai", "openai_compatible", "http"} else SECRET_KEYS
    for key in secret_keys:
        value = env.get(key, "")
        if not value or value.startswith("REPLACE_WITH_"):
            errors.append(f"{key} must be configured with a real value.")
        elif key not in {"LLM_API_KEY", "EMBEDDING_API_KEY"} and len(value.encode("utf-8")) < 32:
            errors.append(f"{key} must contain at least 32 bytes.")

    internal_values = [env.get(key, "") for key in (*INTERNAL_KEYS, "GATEWAY_JWT_SECRET")]
    if len(set(internal_values)) != len(internal_values):
        errors.append("Internal and administrative API keys must all be distinct.")

    for key in ("POSTGRES_PASSWORD", "NEO4J_PASSWORD"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,}", env.get(key, "")):
            errors.append(f"{key} must use at least 32 URL-safe letters, digits, underscores or hyphens.")
    if provider == "fastembed":
        if env.get("EMBEDDING_MODEL") != "BAAI/bge-small-zh-v1.5" or env.get("EMBEDDING_DIMENSION") != "512":
            errors.append("The bundled FastEmbed deployment requires BAAI/bge-small-zh-v1.5 with dimension 512.")
    elif provider in {"openai", "openai_compatible", "http"}:
        endpoint = urlparse(env.get("EMBEDDING_API_BASE_URL", ""))
        if endpoint.scheme != "https" or not endpoint.hostname or "REPLACE_WITH" in endpoint.netloc:
            errors.append("EMBEDDING_API_BASE_URL must be a configured HTTPS endpoint.")
        if not env.get("EMBEDDING_MODEL") or env.get("EMBEDDING_MODEL", "").startswith("REPLACE_WITH"):
            errors.append("EMBEDDING_MODEL must be configured.")
        if not env.get("EMBEDDING_DIMENSION", "").isdigit() or int(env.get("EMBEDDING_DIMENSION", "0")) <= 0:
            errors.append("EMBEDDING_DIMENSION must be a positive integer.")
    else:
        errors.append("EMBEDDING_PROVIDER must be fastembed or openai_compatible, never hash.")

    for key in SERVICE_URL_KEYS:
        value = env.get(key, "")
        if not value:
            errors.append(f"{key} is required.")
            continue
        try:
            parsed = urlparse(value)
            hostname = parsed.hostname
            _ = parsed.port
        except ValueError:
            errors.append(f"{key} must be a valid service URL.")
            continue
        if not hostname or not parsed.scheme:
            errors.append(f"{key} must be a valid service URL.")
        if hostname in {"127.0.0.1", "localhost", "0.0.0.0"}:
            errors.append(f"{key} must use its Docker service name, not {hostname}.")

    domain = env.get("NAWEI_DOMAIN", "").strip()
    if not valid_domain(domain):
        errors.append("NAWEI_DOMAIN must be a real DNS hostname without a scheme, port or path.")
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", env.get("TLS_EMAIL", "")):
        errors.append("TLS_EMAIL must be a valid certificate contact email.")
    if env.get("NAWEI_PUBLIC_ROBOT_URL", "").rstrip("/") != f"https://{domain}":
        errors.append("NAWEI_PUBLIC_ROBOT_URL must be https://NAWEI_DOMAIN.")
    if env.get("XIAOZHI_PUBLIC_WEBSOCKET_URL") != f"wss://{domain}/xiaozhi/v1/":
        errors.append("XIAOZHI_PUBLIC_WEBSOCKET_URL must be wss://NAWEI_DOMAIN/xiaozhi/v1/.")
    if env.get("ASR_REALTIME_BACKEND", "sensevoice") not in {"sensevoice", "paraformer"}:
        errors.append("ASR_REALTIME_BACKEND must be sensevoice or paraformer.")
    if env.get("ASR_DEVICE", "cpu") != "cpu":
        errors.append("The single-GPU production profile uses ASR_DEVICE=cpu; the GPU is reserved for TTS.")
    if env.get("GATEWAY_VOICE_PREVIEW_ENABLED", "false").lower() == "true" and env.get("LLM_WORKERS", "1") != "1":
        errors.append("Voice preview requires LLM_WORKERS=1 until its cache is shared across workers.")
    return errors


def valid_domain(domain: str) -> bool:
    return bool(
        3 <= len(domain) <= 253 and "." in domain and not domain.endswith("example.com")
        and not domain.replace(".", "").isdigit()
        and all(re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label) for label in domain.split("."))
    )


def main() -> None:
    errors = validate_environment()
    if errors:
        raise SystemExit("Invalid production environment:\n- " + "\n- ".join(errors))
    print("Production environment is valid for the five-service robot backend.")


if __name__ == "__main__":
    main()
