"""Configurable external keyword-filter client for ASR transcripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import httpx


class KeywordFilterError(RuntimeError):
    """Raised when the keyword-filter API cannot return a valid decision."""


@dataclass(frozen=True)
class KeywordFilterSettings:
    enabled: bool
    url: str
    timeout_seconds: float
    failure_mode: str
    fallback_text: str
    text_field: str
    blocked_field: str
    allowed_field: str
    matched_keywords_field: str
    reason_field: str
    api_key: str
    api_key_header: str
    api_key_prefix: str

    @classmethod
    def load(cls) -> "KeywordFilterSettings":
        failure_mode = os.getenv("GATEWAY_KEYWORD_FILTER_FAILURE_MODE", "closed").strip().lower()
        if failure_mode not in {"open", "closed"}:
            raise ValueError("GATEWAY_KEYWORD_FILTER_FAILURE_MODE must be open or closed")
        timeout_seconds = float(os.getenv("GATEWAY_KEYWORD_FILTER_TIMEOUT_SECONDS", "3").strip() or "3")
        if timeout_seconds <= 0:
            raise ValueError("GATEWAY_KEYWORD_FILTER_TIMEOUT_SECONDS must be greater than 0")
        return cls(
            enabled=_read_bool_env("GATEWAY_KEYWORD_FILTER_ENABLED", False),
            url=os.getenv("GATEWAY_KEYWORD_FILTER_URL", "").strip(),
            timeout_seconds=timeout_seconds,
            failure_mode=failure_mode,
            fallback_text=os.getenv(
                "GATEWAY_KEYWORD_FILTER_FALLBACK_TEXT",
                "抱歉，这个话题我暂时不能回答。我们换一个安全、合适的话题吧。",
            ).strip(),
            text_field=os.getenv("GATEWAY_KEYWORD_FILTER_TEXT_FIELD", "text").strip() or "text",
            blocked_field=os.getenv("GATEWAY_KEYWORD_FILTER_BLOCKED_FIELD", "blocked").strip(),
            allowed_field=os.getenv("GATEWAY_KEYWORD_FILTER_ALLOWED_FIELD", "").strip(),
            matched_keywords_field=os.getenv(
                "GATEWAY_KEYWORD_FILTER_MATCHED_KEYWORDS_FIELD", "matched_keywords"
            ).strip(),
            reason_field=os.getenv("GATEWAY_KEYWORD_FILTER_REASON_FIELD", "reason").strip(),
            api_key=os.getenv("GATEWAY_KEYWORD_FILTER_API_KEY", "").strip(),
            api_key_header=os.getenv("GATEWAY_KEYWORD_FILTER_API_KEY_HEADER", "Authorization").strip()
            or "Authorization",
            api_key_prefix=os.getenv("GATEWAY_KEYWORD_FILTER_API_KEY_PREFIX", "Bearer").strip(),
        )


@dataclass(frozen=True)
class KeywordFilterDecision:
    blocked: bool
    matched_keywords: tuple[str, ...] = ()
    reason: str = ""
    failed: bool = False


async def check_keyword_filter(
    client: httpx.AsyncClient,
    text: str,
    settings: KeywordFilterSettings,
) -> KeywordFilterDecision:
    if not settings.url:
        raise KeywordFilterError("GATEWAY_KEYWORD_FILTER_URL is required when keyword filtering is enabled")
    if not settings.fallback_text:
        raise KeywordFilterError("GATEWAY_KEYWORD_FILTER_FALLBACK_TEXT must not be empty")

    headers: dict[str, str] = {"Accept": "application/json"}
    if settings.api_key:
        headers[settings.api_key_header] = _credential_value(settings.api_key, settings.api_key_prefix)
    payload: dict[str, Any] = {}
    _set_path(payload, settings.text_field, text)

    try:
        response = await client.post(settings.url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPError as exc:
        raise KeywordFilterError(f"Keyword-filter request failed: {type(exc).__name__}") from exc
    except ValueError as exc:
        raise KeywordFilterError("Keyword-filter response is not valid JSON") from exc

    if not isinstance(body, Mapping):
        raise KeywordFilterError("Keyword-filter response must be a JSON object")
    return parse_keyword_filter_response(body, settings)


def parse_keyword_filter_response(
    body: Mapping[str, Any],
    settings: KeywordFilterSettings,
) -> KeywordFilterDecision:
    blocked_value = _get_path(body, settings.blocked_field) if settings.blocked_field else _MISSING
    if blocked_value is not _MISSING:
        blocked = _as_bool(blocked_value, settings.blocked_field)
    elif settings.allowed_field:
        allowed_value = _get_path(body, settings.allowed_field)
        if allowed_value is _MISSING:
            raise KeywordFilterError(
                f"Keyword-filter response is missing fields {settings.blocked_field!r} and {settings.allowed_field!r}"
            )
        blocked = not _as_bool(allowed_value, settings.allowed_field)
    else:
        raise KeywordFilterError(f"Keyword-filter response is missing field {settings.blocked_field!r}")

    matched_keywords = _as_keywords(_get_path(body, settings.matched_keywords_field))
    reason_value = _get_path(body, settings.reason_field) if settings.reason_field else _MISSING
    reason = "" if reason_value is _MISSING or reason_value is None else str(reason_value).strip()
    return KeywordFilterDecision(blocked=blocked, matched_keywords=matched_keywords, reason=reason)


_MISSING = object()


def _get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise KeywordFilterError("GATEWAY_KEYWORD_FILTER_TEXT_FIELD must not be empty")
    current = target
    for part in parts[:-1]:
        child: dict[str, Any] = {}
        current[part] = child
        current = child
    current[parts[-1]] = value


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "blocked", "reject", "rejected", "hit", "unsafe"}:
            return True
        if normalized in {"0", "false", "no", "allowed", "allow", "passed", "pass", "safe"}:
            return False
    raise KeywordFilterError(f"Keyword-filter field {field!r} must contain a boolean decision")


def _as_keywords(value: Any) -> tuple[str, ...]:
    if value is _MISSING or value is None:
        return ()
    if isinstance(value, str):
        keyword = value.strip()
        return (keyword,) if keyword else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _credential_value(api_key: str, prefix: str) -> str:
    return f"{prefix} {api_key}" if prefix else api_key


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
