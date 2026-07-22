"""Unified protocol errors."""

from __future__ import annotations

from typing import Optional

from .base import ProtocolModel, RequestId
from .enums import ErrorCode


ERROR_STATUS_MAP = {
    ErrorCode.INVALID_ARGUMENT: 400,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.SESSION_NOT_FOUND: 404,
    ErrorCode.AUDIO_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_AUDIO_FORMAT: 415,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.ASR_TIMEOUT: 504,
    ErrorCode.LLM_TIMEOUT: 504,
    ErrorCode.TTS_TIMEOUT: 504,
}

RETRYABLE_ERROR_CODES = {
    ErrorCode.ASR_TIMEOUT,
    ErrorCode.LLM_TIMEOUT,
    ErrorCode.TTS_TIMEOUT,
    ErrorCode.MODEL_UNAVAILABLE,
    ErrorCode.RATE_LIMITED,
    ErrorCode.INTERNAL_ERROR,
}


class ProtocolErrorDetail(ProtocolModel):
    """Serializable error response body."""

    code: ErrorCode
    message: str
    request_id: str = RequestId
    trace_id: Optional[str] = None
    retryable: bool = False


class ErrorResponse(ProtocolModel):
    """Frontend-facing error response wrapper."""

    error: ProtocolErrorDetail


class ProtocolException(Exception):
    """Exception carrying a protocol error detail."""

    def __init__(self, detail: ProtocolErrorDetail, status_code: Optional[int] = None) -> None:
        super().__init__(detail.message)
        self.detail = detail
        self.status_code = status_code or status_for_error(detail.code)


def status_for_error(code: ErrorCode) -> int:
    """Return the HTTP status code mapped to a platform error code."""

    return ERROR_STATUS_MAP.get(code, 500)


def is_retryable_error(code: ErrorCode) -> bool:
    """Return whether the error code is generally safe to retry."""

    return code in RETRYABLE_ERROR_CODES


def build_error_response(
    code: ErrorCode,
    message: str,
    request_id: str,
    trace_id: Optional[str] = None,
) -> ErrorResponse:
    """Build a unified frontend error response."""

    return ErrorResponse(
        error=ProtocolErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
            trace_id=trace_id,
            retryable=is_retryable_error(code),
        )
    )