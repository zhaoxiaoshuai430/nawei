"""LLM service using app_protocols DTOs and a DeepSeek-compatible backend."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import Field

from app_protocols import EventType, LLMRequest, LLMResponse, Message, MessageRole, StreamEvent, Usage
from app_protocols.base import ProtocolModel
from app_protocols.events import sse_encode
from services.llm.agent_runtime import AgentTurn, LLMAgentRuntime
from services.llm.auth import authorize_llm_internal_request

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - fallback keeps local tests runnable without optional dependency
    _load_dotenv = None


MODEL_ID = "stub-llm"
PROVIDER_STUB = "stub"
PROVIDER_DEEPSEEK = "deepseek"
DEFAULT_TIMEOUT_SECONDS = 25.0
PLACEHOLDER_API_KEY_PREFIX = "REPLACE_WITH_"
DEFAULT_FALLBACK_TEXT = "现在模型响应有点慢，我先给你一个简短版：先把问题拆成“概念、例子、练习”三步。你可以把具体题目或卡住的地方发给我，我会继续一步一步带你推。"
DEFAULT_SYSTEM_PROMPT = (
    "你是小维，学生的 AI 教育语音助手。"
    "用简体中文回答，适合朗读；不要说自己是 DeepSeek。"
    "默认回答要更完整，不要只给一句短答。"
    "学习问题先给结论和思路，再给必要步骤、例子、易错点；信息不足时先说明可判断的部分，再提出1个追问。"
)
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

_deepseek_clients: dict[float, httpx.AsyncClient] = {}
agent_runtime = LLMAgentRuntime()


@asynccontextmanager
async def _llm_lifespan(application: FastAPI) -> AsyncIterator[None]:
    await agent_runtime.initialize()
    try:
        yield
    finally:
        await agent_runtime.close()
        for client in list(_deepseek_clients.values()):
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()
        _deepseek_clients.clear()


app = FastAPI(title="Xiaowei Agentic LLM Service", version="0.3.0", lifespan=_llm_lifespan)


@dataclass(frozen=True)
class LLMSettings:
    """Runtime configuration for the LLM backend."""

    provider: str = PROVIDER_STUB
    base_url: str = ""
    api_key: str = ""
    model: str = MODEL_ID
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    fallback_enabled: bool = True

    @property
    def is_stub(self) -> bool:
        """Return whether the service should use the local placeholder."""

        return self.provider == PROVIDER_STUB


class MemoryRememberRequest(ProtocolModel):
    session_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1, max_length=2000)
    memory_type: str = Field("episodic", min_length=1, max_length=50)
    importance: int = Field(5, ge=1, le=10)
    confidence: float = Field(0.75, ge=0.0, le=1.0)
    sensitivity: str = Field("normal", max_length=30)
    tags: list[str] = Field(default_factory=list)


class MemoryProfileRequest(ProtocolModel):
    session_id: str = Field(..., min_length=1)
    changes: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field("manual", min_length=1, max_length=100)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return inference and internal Agent runtime health."""

    settings = load_settings()
    runtime = await agent_runtime.health()
    memory = runtime["memory"]
    required = _read_bool_env("LLM_MEMORY_REQUIRED", False)
    ready = not required or all(memory.get(name) == "ok" for name in ("postgres", "redis"))
    return {
        "status": "ok" if ready else "degraded",
        "model": settings.model,
        "provider": settings.provider,
        "runtime": runtime,
    }


@app.get(
    "/api/v1/llm/runtime",
    dependencies=[Depends(authorize_llm_internal_request)],
)
async def runtime_status() -> dict[str, Any]:
    """Return the LLM-owned Agent, Skills, Tools, and Memory status."""

    return await agent_runtime.health()


@app.get(
    "/api/v1/llm/memory/{user_id}",
    dependencies=[Depends(authorize_llm_internal_request)],
)
async def inspect_memory(
    user_id: str,
    session_id: str = Query(..., min_length=1),
    query: str = Query("", max_length=2000),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Inspect one user's LLM-owned profile, history, memories, and concepts."""

    return await agent_runtime.memory_store.inspect(
        user_id=user_id,
        session_id=session_id,
        query=query,
        limit=limit,
    )


@app.post(
    "/api/v1/llm/memory/{user_id}/remember",
    dependencies=[Depends(authorize_llm_internal_request)],
)
async def remember_memory(user_id: str, request: MemoryRememberRequest) -> dict[str, Any]:
    """Write or merge one governed long-term memory."""

    memory_id = await agent_runtime.memory_store.remember(
        user_id=user_id,
        session_id=request.session_id,
        content=request.content,
        memory_type=request.memory_type,
        importance=request.importance,
        confidence=request.confidence,
        sensitivity=request.sensitivity,
        tags=request.tags,
        source_agent="llm_agent",
    )
    return {"stored": memory_id is not None, "memory_id": memory_id}


@app.post(
    "/api/v1/llm/memory/{user_id}/profile",
    dependencies=[Depends(authorize_llm_internal_request)],
)
async def update_memory_profile(user_id: str, request: MemoryProfileRequest) -> dict[str, Any]:
    """Merge trusted profile fields into persistent LLM Memory."""

    await agent_runtime.memory_store.update_profile(
        user_id=user_id,
        changes=request.changes,
        reason=request.reason,
        session_id=request.session_id,
    )
    return {"updated": True, "profile": await agent_runtime.memory_store.get_profile(user_id)}


@app.delete(
    "/api/v1/llm/memory/{user_id}/items/{memory_id}",
    dependencies=[Depends(authorize_llm_internal_request)],
)
async def forget_memory(user_id: str, memory_id: str) -> dict[str, bool]:
    """Soft-delete one long-term memory owned by the user."""

    return {"deleted": await agent_runtime.memory_store.forget_memory(user_id=user_id, memory_id=memory_id)}


@app.delete(
    "/api/v1/llm/memory/{user_id}",
    dependencies=[Depends(authorize_llm_internal_request)],
)
async def clear_memory(user_id: str, session_id: str | None = None) -> dict[str, bool]:
    """Clear one user's persistent LLM Memory and optional session history."""

    await agent_runtime.memory_store.clear_user(user_id=user_id, session_id=session_id)
    return {"cleared": True}


@app.post(
    "/api/v1/llm",
    response_model=LLMResponse,
    dependencies=[Depends(authorize_llm_internal_request)],
)
async def complete_llm(request: LLMRequest) -> LLMResponse:
    """Run Agent, Skills, Tools, Memory, and configured inference as one turn."""

    settings = load_settings()
    turn: AgentTurn | None = None
    try:
        turn = await agent_runtime.prepare(request)
        response = complete_stub(turn.request) if settings.is_stub else await complete_deepseek(turn.request, settings)
        return await _finalize_agent_response(turn, response)
    except httpx.HTTPStatusError as exc:
        if settings.fallback_enabled:
            active_request = turn.request if turn is not None else request
            response = fallback_response(active_request, settings, reason=_extract_upstream_error(exc.response), status_code=exc.response.status_code)
            return await _finalize_agent_response(turn, response)
        raise HTTPException(status_code=502, detail=_extract_upstream_error(exc.response)) from exc
    except httpx.HTTPError as exc:
        if settings.fallback_enabled:
            active_request = turn.request if turn is not None else request
            response = fallback_response(active_request, settings, reason=f"LLM upstream request failed: {exc}")
            return await _finalize_agent_response(turn, response)
        raise HTTPException(status_code=502, detail=f"LLM upstream request failed: {exc}") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        if settings.fallback_enabled:
            active_request = turn.request if turn is not None else request
            response = fallback_response(active_request, settings, reason=f"Invalid LLM upstream response: {exc}")
            return await _finalize_agent_response(turn, response)
        raise HTTPException(status_code=502, detail=f"Invalid LLM upstream response: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"LLM Agent runtime unavailable: {exc}") from exc


@app.post(
    "/api/v1/llm/stream",
    dependencies=[Depends(authorize_llm_internal_request)],
)
async def stream_llm(request: LLMRequest) -> StreamingResponse:
    """Stream one complete Agent and inference turn as canonical SSE."""

    settings = load_settings()

    async def event_source() -> AsyncIterator[str]:
        try:
            turn = await agent_runtime.prepare(request)
            parts: list[str] = []
            done_event: StreamEvent | None = None
            failed = False
            async for event in _stream_backend_with_fallback(turn.request, settings):
                if event.event == EventType.MESSAGE_DELTA:
                    parts.append(str(event.data.get("text", "")))
                if event.event == EventType.MESSAGE_DONE:
                    done_event = event
                    continue
                if event.event == EventType.ERROR:
                    failed = True
                yield sse_encode(event)
            if failed:
                return
            persistence = await agent_runtime.after_turn(turn, "".join(parts))
            final_event = done_event or StreamEvent(
                event=EventType.MESSAGE_DONE,
                request_id=request.request_id,
                trace_id=request.trace_id,
                session_id=request.session_id,
                data={"message_id": request.request_id, "finish_reason": "stop"},
            )
            final_data = dict(final_event.data)
            final_data["runtime"] = turn.metadata
            final_data["persistence"] = persistence
            yield sse_encode(_copy_stream_event(final_event, data=final_data))
        except RuntimeError as exc:
            yield sse_encode(_error_event(request, f"LLM Agent runtime unavailable: {exc}"))

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def complete_stub(request: LLMRequest) -> LLMResponse:
    """Return the local placeholder response."""

    user_text = _last_user_text(request)
    reply = _stub_teaching_reply(user_text, _system_text(request))
    return LLMResponse(
        request_id=request.request_id,
        text=reply,
        finish_reason="stop",
        usage=Usage(input_tokens=len(user_text), output_tokens=len(reply)),
        metadata={"model": MODEL_ID},
    )


async def stream_stub(request: LLMRequest) -> AsyncIterator[StreamEvent]:
    """Yield a deterministic local stream for development."""

    response = complete_stub(request)
    for chunk in _text_chunks(response.text, size=8):
        yield StreamEvent(
            event=EventType.MESSAGE_DELTA,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"text": chunk},
        )
    yield StreamEvent(
        event=EventType.MESSAGE_DONE,
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        data={
            "message_id": request.request_id,
            "finish_reason": response.finish_reason,
            "usage": _model_to_dict(response.usage),
        },
    )


async def _stream_backend_with_fallback(
    request: LLMRequest,
    settings: LLMSettings,
) -> AsyncIterator[StreamEvent]:
    try:
        source = stream_stub(request) if settings.is_stub else stream_deepseek(request, settings)
        async for event in source:
            yield event
    except httpx.HTTPStatusError as exc:
        if not settings.fallback_enabled:
            yield _error_event(request, _extract_upstream_error(exc.response))
            return
        async for event in stream_fallback(
            request,
            settings,
            reason=_extract_upstream_error(exc.response),
            status_code=exc.response.status_code,
        ):
            yield event
    except httpx.HTTPError as exc:
        if not settings.fallback_enabled:
            yield _error_event(request, f"LLM upstream stream failed: {exc}")
            return
        async for event in stream_fallback(request, settings, reason=f"LLM upstream stream failed: {exc}"):
            yield event
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        if not settings.fallback_enabled:
            yield _error_event(request, f"Invalid LLM upstream stream response: {exc}")
            return
        async for event in stream_fallback(
            request,
            settings,
            reason=f"Invalid LLM upstream stream response: {exc}",
        ):
            yield event


def fallback_response(
    request: LLMRequest,
    settings: LLMSettings,
    *,
    reason: str,
    status_code: int | None = None,
) -> LLMResponse:
    """Return a short deterministic response when the upstream LLM is slow or unavailable."""

    configured_text = os.getenv("LLM_FALLBACK_TEXT", "").strip()
    text = configured_text or _contextual_fallback_text(_last_user_text(request))
    return LLMResponse(
        request_id=request.request_id,
        text=text,
        finish_reason="fallback",
        usage=Usage(input_tokens=len(_last_user_text(request)), output_tokens=len(text)),
        metadata={
            "provider": settings.provider,
            "model": settings.model,
            "fallback": True,
            "fallback_reason": reason[:300],
            "upstream_status_code": status_code,
        },
    )


def _contextual_fallback_text(user_text: str) -> str:
    normalized = user_text.strip()
    if any(marker in normalized for marker in ("故事", "童话", "寓言")):
        return "我先讲一个短故事：小星星掉进草丛里，急得直闪光。小兔子捧来一片露珠，映出天空的路。星星顺着光回家前，悄悄把草地照成了银色。"
    if any(marker in normalized for marker in ("笑话", "搞笑", "逗我")):
        return "给你一个轻一点的笑话：小明问老师，为什么书包这么重？老师说，因为里面装着知识。小明想了想，那我先少装一点，慢慢进步。"
    if any(marker in normalized for marker in ("自我介绍", "你是谁", "介绍一下你")):
        return "我是小维，你的 AI 教育语音助手。你可以问我知识点、作业思路、复习计划，也可以让我用更简单的例子重新讲一遍。"
    if any(marker in normalized for marker in ("不会", "不懂", "卡住", "怎么做", "怎么解")):
        return "我先给你一个临时思路：别急着算答案，先圈出题目给了什么、要求什么，再找对应概念。你把题目发完整，我会按步骤带你拆。"
    return DEFAULT_FALLBACK_TEXT


async def stream_fallback(
    request: LLMRequest,
    settings: LLMSettings,
    *,
    reason: str,
    status_code: int | None = None,
) -> AsyncIterator[StreamEvent]:
    """Yield fallback text as normal message events so clients do not hang."""

    response = fallback_response(request, settings, reason=reason, status_code=status_code)
    for chunk in _text_chunks(response.text, size=12):
        yield StreamEvent(
            event=EventType.MESSAGE_DELTA,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"text": chunk, "fallback": True, "reason": reason[:120]},
        )
    yield StreamEvent(
        event=EventType.MESSAGE_DONE,
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        data={
            "message_id": request.request_id,
            "finish_reason": response.finish_reason,
            "usage": _model_to_dict(response.usage),
            "fallback": True,
            "reason": reason[:120],
        },
    )


async def complete_deepseek(request: LLMRequest, settings: LLMSettings) -> LLMResponse:
    """Call the DeepSeek chat completions API."""

    payload = _build_deepseek_payload(request, settings)
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    client = _deepseek_client(settings.timeout_seconds)
    response = await client.post(_chat_completions_url(settings.base_url), json=payload, headers=headers)
    response.raise_for_status()
    body = response.json()

    choice = body["choices"][0]
    message = choice.get("message") or {}
    text = message.get("content") or choice.get("text") or ""
    if not str(text).strip():
        raise ValueError("LLM upstream returned empty content")
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
    """Call the DeepSeek chat completions streaming API."""

    payload = _build_deepseek_payload(request, settings)
    payload["stream"] = True
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    emitted_text = False
    client = _deepseek_client(settings.timeout_seconds)
    async with client.stream("POST", _chat_completions_url(settings.base_url), json=payload, headers=headers) as response:
        response.raise_for_status()
        async for body in _iter_openai_sse(response):
            choice = (body.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            text = delta.get("content") or choice.get("text") or ""
            if text:
                emitted_text = True
                yield StreamEvent(
                    event=EventType.MESSAGE_DELTA,
                    request_id=request.request_id,
                    trace_id=request.trace_id,
                    session_id=request.session_id,
                    data={"text": text},
                )
    if not emitted_text:
        fallback_response = await complete_deepseek(request, settings)
        for chunk in _text_chunks(fallback_response.text, size=12):
            yield StreamEvent(
                event=EventType.MESSAGE_DELTA,
                request_id=request.request_id,
                trace_id=request.trace_id,
                session_id=request.session_id,
                data={"text": chunk, "fallback": "non_stream_after_empty_stream"},
            )
    yield StreamEvent(
        event=EventType.MESSAGE_DONE,
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        data={"message_id": request.request_id, "finish_reason": "stop", "usage": {}},
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
        fallback_enabled=_read_bool_env("LLM_FALLBACK_ENABLED", True),
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


def _deepseek_client(timeout_seconds: float) -> httpx.AsyncClient:
    client_class = httpx.AsyncClient
    client = _deepseek_clients.get(timeout_seconds)
    if client is not None and client.__class__ is client_class and not getattr(client, "is_closed", False):
        return client

    try:
        client = client_class(
            timeout=timeout_seconds,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=60.0),
        )
    except TypeError:
        client = client_class(timeout=timeout_seconds)
    _deepseek_clients[timeout_seconds] = client
    return client


def _build_deepseek_payload(request: LLMRequest, settings: LLMSettings) -> dict[str, Any]:
    messages = list(request.messages)
    if not any(message.role == MessageRole.SYSTEM for message in messages):
        messages.insert(0, Message(role=MessageRole.SYSTEM, content=_default_system_prompt()))
    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": [_message_to_deepseek_payload(message) for message in messages],
        "stream": False,
    }
    for key in ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty"):
        if key in request.generation_config:
            payload[key] = request.generation_config[key]
    if request.tools:
        payload["tools"] = request.tools
    if not _read_bool_env("LLM_THINKING_ENABLED", False):
        payload["thinking"] = {"type": "disabled"}
    return payload


def _message_to_deepseek_payload(message: Any) -> dict[str, Any]:
    payload = {"role": str(message.role), "content": message.content}
    if message.name:
        payload["name"] = message.name
    return payload


async def _iter_openai_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    async for line in response.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Expected JSON object")
        yield value


def _last_user_text(request: LLMRequest) -> str:
    return next((message.content for message in reversed(request.messages) if message.role == MessageRole.USER), "")


def _system_text(request: LLMRequest) -> str:
    return "\n\n".join(message.content for message in request.messages if message.role == MessageRole.SYSTEM)


def _current_date_context() -> str:
    now = datetime.now(timezone(timedelta(hours=8)))
    return f"{now:%Y-%m-%d} {now:%A} Asia/Shanghai"


def _default_system_prompt() -> str:
    return (
        DEFAULT_SYSTEM_PROMPT
        + f"当前日期：{_current_date_context()}。涉及今年、最新、赛事、新闻或可能变化的信息时，必须先说明时效性；没有实时工具结果时不要编造最新结论。"
    )


def _stub_teaching_reply(user_text: str, system_text: str = "") -> str:
    if any(marker in user_text for marker in ("你是谁", "自我介绍", "介绍一下你")):
        return (
            "我是小维，你的 AI 教育语音助手。"
            "我主要陪你学习、讲题、复习知识点，也可以把复杂内容拆成更容易听懂的小步骤。"
            "如果你愿意，可以直接告诉我现在卡在哪一题或哪一个知识点。"
        )

    if any(marker in user_text for marker in ("笑话", "搞笑", "逗我")):
        return "给你一个短笑话：老师问小明，为什么这道题空着？小明说，我给答案留了成长空间。好了，笑完我们继续学习也行。"

    if any(marker in user_text for marker in ("故事", "童话", "寓言")):
        return (
            "从前有一颗小星星掉进草地里，怎么也找不到回家的路。"
            "一只小兔子捧来露珠，让露珠映出天空。"
            "小星星顺着亮光飞回去，临走前把草地照得像铺了一层银霜。"
        )

    if "二次函数" in user_text or "顶点式" in user_text:
        return (
            "二次函数可以先抓住三个信息：开口方向、顶点位置、和图像宽窄。"
            "如果是顶点式 y=a(x-h)^2+k，顶点就是 (h,k)，a 决定开口方向和宽窄。"
            "比如 y=2(x-3)^2+1，顶点是 (3,1)，开口向上，而且比 y=x^2 更窄。"
            "做题时先找 h、k，再看 a 的正负和大小，最后再画大致图像。"
            "你可以把具体题目发来，我会按这四步带你做一遍。"
        )

    if any(marker in user_text for marker in ("出题", "练习", "测验", "考我")):
        return (
            "可以，我先给你一道短练习。"
            "题目：函数 y=2(x-1)^2+3 的顶点坐标是什么？开口向上还是向下？"
            "你先试着回答，我会根据你的答案继续提示。"
        )

    return (
        f"我听到了：{user_text}。"
        "我会按学习辅导的方式来处理。"
        "第一步先确认你卡住的是概念、公式、步骤还是题目条件。"
        "第二步把问题拆成两三个小问题，一个个解决。"
        "第三步用一个短例子检查你是不是真的理解了。"
        "你可以继续补充具体题目，或者告诉我是哪一步不懂。"
    )


def _text_chunks(text: str, size: int = 12) -> list[str]:
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


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


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[no-any-return, attr-defined]
    return model.dict()  # type: ignore[no-any-return, attr-defined]


def _response_with_runtime_metadata(
    response: LLMResponse,
    turn: AgentTurn,
    persistence: dict[str, Any],
) -> LLMResponse:
    metadata = dict(response.metadata)
    metadata["runtime"] = turn.metadata
    metadata["persistence"] = persistence
    update = {"metadata": metadata}
    if hasattr(response, "model_copy"):
        return response.model_copy(update=update)
    return response.copy(update=update)


async def _finalize_agent_response(turn: AgentTurn | None, response: LLMResponse) -> LLMResponse:
    if turn is None:
        return response
    persistence = await agent_runtime.after_turn(turn, response.text)
    return _response_with_runtime_metadata(response, turn, persistence)


def _copy_stream_event(event: StreamEvent, *, data: dict[str, Any]) -> StreamEvent:
    update = {"data": data}
    if hasattr(event, "model_copy"):
        return event.model_copy(update=update)
    return event.copy(update=update)


def _error_event(request: LLMRequest, message: str) -> StreamEvent:
    return StreamEvent(
        event=EventType.ERROR,
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        data={"message": message, "retryable": True},
    )
