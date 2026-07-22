"""LLM service using app_protocols DTOs and the DeepSeek API backend."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app_protocols import ErrorCode, EventType, LLMRequest, LLMResponse, StreamEvent, Usage, sse_encode

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - fallback keeps local tests runnable without optional dependency
    _load_dotenv = None


MODEL_ID = "stub-llm"
PROVIDER_STUB = "stub"
PROVIDER_DEEPSEEK = "deepseek"
DEFAULT_TIMEOUT_SECONDS = 60.0
PLACEHOLDER_API_KEY_PREFIX = "REPLACE_WITH_"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


def _load_dotenv_file(path: Path) -> None:
    """Load local .env values without overriding existing environment variables."""

    if not path.exists():
        return
    if _load_dotenv is not None:
        _load_dotenv(dotenv_path=path, override=False)
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _strip_env_quotes(value.strip())


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


_load_dotenv_file(DOTENV_PATH)

app = FastAPI(title="Navi LLM Service", version="0.1.0")


@dataclass(frozen=True)
class LLMSettings:
    """Runtime configuration for the LLM backend."""

    provider: str = PROVIDER_STUB
    base_url: str = ""
    api_key: str = ""
    model: str = MODEL_ID
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @property
    def is_stub(self) -> bool:
        """Return whether the service should use the local placeholder."""

        return self.provider == PROVIDER_STUB


@app.get("/health")
async def health() -> dict[str, str]:
    """Return LLM service health."""

    settings = load_settings()
    return {"status": "ok", "model": settings.model, "provider": settings.provider}


@app.post("/api/v1/llm", response_model=LLMResponse)
async def complete_llm(request: LLMRequest) -> LLMResponse:
    """Run an LLM completion through the configured backend."""

    settings = load_settings()
    if settings.is_stub:
        return complete_stub(request)

    try:
        return await complete_deepseek(request, settings)
    except httpx.HTTPStatusError as exc:
        detail = _extract_upstream_error(exc.response)
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"LLM upstream request failed: {exc}") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Invalid LLM upstream response: {exc}") from exc


@app.post("/api/v1/llm/stream")
async def stream_llm(request: LLMRequest) -> StreamingResponse:
    """Stream an LLM completion as canonical Server-Sent Events."""

    settings = load_settings()

    async def event_source() -> AsyncIterator[str]:
        try:
            if settings.is_stub:
                async for event in stream_stub(request):
                    yield sse_encode(event)
            else:
                async for event in stream_deepseek(request, settings):
                    yield sse_encode(event)
        except httpx.HTTPStatusError as exc:
            yield sse_encode(_error_event(request, _extract_upstream_error(exc.response)))
        except httpx.HTTPError as exc:
            yield sse_encode(_error_event(request, f"LLM upstream request failed: {exc}"))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            yield sse_encode(_error_event(request, f"Invalid LLM upstream response: {exc}"))

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def complete_stub(request: LLMRequest) -> LLMResponse:
    """Return the local placeholder response."""

    user_text = next((message.content for message in reversed(request.messages) if message.role == "user"), "")
    reply = f"我听到了：{user_text}"
    return LLMResponse(
        request_id=request.request_id,
        text=reply,
        finish_reason="stop",
        usage=Usage(input_tokens=len(user_text), output_tokens=len(reply)),
        metadata={"model": MODEL_ID},
    )


async def stream_stub(request: LLMRequest) -> AsyncIterator[StreamEvent]:
    """Stream the local placeholder one character at a time."""

    response = complete_stub(request)
    for text in response.text:
        yield _stream_event(request, EventType.MESSAGE_DELTA, {"text": text})
        await asyncio.sleep(0)
    yield _stream_event(
        request,
        EventType.MESSAGE_DONE,
        {
            "message_id": request.request_id,
            "finish_reason": response.finish_reason,
            "usage": _model_to_dict(response.usage),
        },
    )


async def complete_deepseek(request: LLMRequest, settings: LLMSettings) -> LLMResponse:
    """Call the DeepSeek chat completions API."""

    payload = _build_deepseek_payload(request, settings)
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
        response = await client.post(_chat_completions_url(settings.base_url), json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()

    choice = body["choices"][0]
    message = choice.get("message") or {}
    text = message.get("content") or choice.get("text") or ""
    usage = body.get("usage") or {}
    model = body.get("model") or settings.model
    return LLMResponse(
        request_id=request.request_id,
        text=text,
        finish_reason=choice.get("finish_reason") or "stop",
        usage=Usage(
            input_tokens=usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            output_tokens=usage.get("completion_tokens", usage.get("output_tokens", 0)),
        ),
        metadata={
            "provider": settings.provider,
            "model": model,
            "upstream_id": body.get("id"),
        },
    )


async def stream_deepseek(request: LLMRequest, settings: LLMSettings) -> AsyncIterator[StreamEvent]:
    """Translate an OpenAI-compatible DeepSeek stream into canonical events."""

    payload = _build_deepseek_payload(request, settings, stream=True)
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    finish_reason = "stop"
    usage: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
        async with client.stream(
            "POST",
            _chat_completions_url(settings.base_url),
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                if not isinstance(chunk, dict):
                    raise ValueError("DeepSeek SSE data must be a JSON object.")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    yield _stream_event(request, EventType.MESSAGE_DELTA, {"text": str(text)})
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])

    yield _stream_event(
        request,
        EventType.MESSAGE_DONE,
        {
            "message_id": request.request_id,
            "finish_reason": finish_reason,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            },
        },
    )


def load_settings() -> LLMSettings:
    """Load LLM settings from environment variables."""

    provider = os.getenv("LLM_PROVIDER", PROVIDER_STUB).strip().lower()
    model = os.getenv("LLM_MODEL", MODEL_ID).strip() or MODEL_ID
    timeout_seconds = _read_float_env("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    settings = LLMSettings(
        provider=provider,
        base_url=os.getenv("LLM_API_BASE_URL", "").strip(),
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        model=model,
        timeout_seconds=timeout_seconds,
    )
    if settings.provider not in {PROVIDER_STUB, PROVIDER_DEEPSEEK}:
        raise HTTPException(status_code=500, detail=f"Unsupported LLM_PROVIDER: {settings.provider}")
    if not settings.is_stub and (not settings.base_url or not settings.model):
        raise HTTPException(
            status_code=500,
            detail="LLM_API_BASE_URL and LLM_MODEL are required when LLM_PROVIDER=deepseek.",
        )
    if not settings.is_stub and settings.api_key.startswith(PLACEHOLDER_API_KEY_PREFIX):
        raise HTTPException(
            status_code=500,
            detail="Replace the LLM_API_KEY placeholder in .env before calling the real LLM API.",
        )
    return settings


def _build_deepseek_payload(
    request: LLMRequest,
    settings: LLMSettings,
    *,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": [_message_to_deepseek_payload(message) for message in request.messages],
        "stream": stream,
    }
    for key in ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty"):
        if key in request.generation_config:
            payload[key] = request.generation_config[key]
    if request.tools:
        payload["tools"] = request.tools
    return payload


def _message_to_deepseek_payload(message: Any) -> dict[str, Any]:
    payload = {"role": str(message.role), "content": message.content}
    if message.name:
        payload["name"] = message.name
    return payload


def _chat_completions_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def _extract_upstream_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            return str(error.get("message") or error)
        if error:
            return str(error)
    return str(body)


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _stream_event(request: LLMRequest, event: EventType, data: dict[str, Any]) -> StreamEvent:
    return StreamEvent(
        event=event,
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        data=data,
    )


def _error_event(request: LLMRequest, message: str) -> StreamEvent:
    return _stream_event(
        request,
        EventType.ERROR,
        {"code": ErrorCode.INTERNAL_ERROR, "message": message, "retryable": True},
    )


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()
