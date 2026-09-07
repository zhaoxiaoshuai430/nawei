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
from services.llm.safety import LocalSafetyGuard, SafetyDecision, SafetyStreamBuffer

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - fallback keeps local tests runnable without optional dependency
    _load_dotenv = None  # type: ignore[assignment]


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
safety_guard: LocalSafetyGuard = agent_runtime.safety_guard


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

    request, input_decision = _audit_request_input(request)
    if input_decision.blocked:
        return _blocked_input_response(request, input_decision)
    settings = load_settings()
    turn: AgentTurn | None = None
    try:
        turn = await agent_runtime.prepare(request)
        if settings.is_stub:
            response = complete_stub(turn.request)
        elif _native_tool_calling_requested(turn.request):
            response = await _complete_deepseek_with_native_tools(turn, settings)
        else:
            response = await complete_deepseek(turn.request, settings)
        return await _finalize_agent_response(turn, response, input_decision=input_decision)
    except httpx.HTTPStatusError as exc:
        if settings.fallback_enabled:
            active_request = turn.request if turn is not None else request
            response = fallback_response(active_request, settings, reason=_extract_upstream_error(exc.response), status_code=exc.response.status_code)
            return await _finalize_agent_response(turn, response, input_decision=input_decision)
        raise HTTPException(status_code=502, detail=_extract_upstream_error(exc.response)) from exc
    except httpx.HTTPError as exc:
        if settings.fallback_enabled:
            active_request = turn.request if turn is not None else request
            response = fallback_response(active_request, settings, reason=f"LLM upstream request failed: {exc}")
            return await _finalize_agent_response(turn, response, input_decision=input_decision)
        raise HTTPException(status_code=502, detail=f"LLM upstream request failed: {exc}") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        if settings.fallback_enabled:
            active_request = turn.request if turn is not None else request
            response = fallback_response(active_request, settings, reason=f"Invalid LLM upstream response: {exc}")
            return await _finalize_agent_response(turn, response, input_decision=input_decision)
        raise HTTPException(status_code=502, detail=f"Invalid LLM upstream response: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"LLM Agent runtime unavailable: {exc}") from exc


@app.post(
    "/api/v1/llm/stream",
    dependencies=[Depends(authorize_llm_internal_request)],
)
async def stream_llm(request: LLMRequest) -> StreamingResponse:
    """Stream one complete Agent and inference turn as canonical SSE."""

    request, input_decision = _audit_request_input(request)
    settings = load_settings()

    async def event_source() -> AsyncIterator[str]:
        if input_decision.blocked:
            yield sse_encode(_safety_delta_event(request, input_decision.content))
            yield sse_encode(_safety_done_event(request, input_decision=input_decision))
            return
        try:
            turn = await agent_runtime.prepare(request)
            safe_parts: list[str] = []
            raw_parts: list[str] = []
            done_event: StreamEvent | None = None
            delta_template: StreamEvent | None = None
            failed = False
            stream_safety = SafetyStreamBuffer(safety_guard)
            stream_chunk_chars = _stream_chunk_chars()
            async for event in _stream_backend_with_fallback(turn.request, settings):
                tool_agent_metadata = event.data.get("tool_agent")
                if isinstance(tool_agent_metadata, dict):
                    turn.metadata["tools"] = {
                        **dict(turn.metadata.get("tools") or {}),
                        **tool_agent_metadata,
                        "results": event.data.get("tool_results") or {},
                        "errors": event.data.get("tool_errors") or {},
                    }
                if event.event == EventType.MESSAGE_DELTA:
                    delta_template = event
                    raw_text = str(event.data.get("text", ""))
                    raw_parts.append(raw_text)
                    for safe_text in stream_safety.feed(raw_text):
                        safe_parts.append(safe_text)
                        chunks = [safe_text] if stream_safety.blocked else _text_chunks(
                            safe_text,
                            size=stream_chunk_chars,
                        )
                        for text in chunks:
                            yield sse_encode(_stream_event_with_text(event, text))
                    if stream_safety.blocked:
                        break
                    continue
                if event.event == EventType.MESSAGE_DONE:
                    done_event = event
                    continue
                if event.event == EventType.ERROR:
                    failed = True
                yield sse_encode(event)
            if failed:
                return
            for safe_text in stream_safety.finish():
                safe_parts.append(safe_text)
                chunks = [safe_text] if stream_safety.blocked else _text_chunks(
                    safe_text,
                    size=stream_chunk_chars,
                )
                for text in chunks:
                    yield sse_encode(_stream_text_event(request, text, template=delta_template))
            output_decision = stream_safety.decision or safety_guard.audit_output("".join(raw_parts))
            if output_decision.changed:
                safety_guard.log(stage="output", decision=output_decision, request_id=request.request_id)
            persistence = await agent_runtime.after_turn(turn, "".join(safe_parts))
            final_event = done_event or StreamEvent(
                event=EventType.MESSAGE_DONE,
                request_id=request.request_id,
                trace_id=request.trace_id,
                session_id=request.session_id,
                data={"message_id": request.request_id, "finish_reason": "stop"},
            )
            final_data = dict(final_event.data)
            if output_decision.blocked:
                final_data["finish_reason"] = "safety"
            final_data["runtime"] = turn.metadata
            final_data["persistence"] = persistence
            final_data["safety"] = _safety_metadata(input_decision, output_decision)
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
    emitted_content = False
    try:
        if settings.is_stub:
            source = stream_stub(request)
        elif _native_tool_calling_requested(request):
            source = stream_deepseek_with_native_tools(request, settings)
        else:
            source = stream_deepseek(request, settings)
        async for event in source:
            if event.event == EventType.MESSAGE_DELTA and str(event.data.get("text") or ""):
                emitted_content = True
            yield event
    except httpx.HTTPStatusError as exc:
        if emitted_content or not settings.fallback_enabled:
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
        if emitted_content or not settings.fallback_enabled:
            yield _error_event(request, f"LLM upstream stream failed: {exc}")
            return
        async for event in stream_fallback(request, settings, reason=f"LLM upstream stream failed: {exc}"):
            yield event
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        if emitted_content or not settings.fallback_enabled:
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


async def _complete_deepseek_with_native_tools(
    turn: AgentTurn,
    settings: LLMSettings,
) -> LLMResponse:
    first_response = await complete_deepseek(turn.request, settings)
    calls = _parse_native_tool_calls(first_response.metadata.get("tool_calls"))
    if not calls:
        return first_response

    followup_request, tool_metadata = await agent_runtime.execute_native_tool_calls(
        turn.request,
        calls=calls,
        first_text=first_response.text,
    )
    final_response = await complete_deepseek(followup_request, settings)
    final_response.usage = Usage(
        input_tokens=first_response.usage.input_tokens + final_response.usage.input_tokens,
        output_tokens=first_response.usage.output_tokens + final_response.usage.output_tokens,
    )
    final_response.metadata = {**dict(final_response.metadata), **tool_metadata}
    turn.metadata["tools"] = {
        **dict(turn.metadata.get("tools") or {}),
        **tool_metadata["tool_agent"],
        "results": tool_metadata["tool_results"],
        "errors": tool_metadata["tool_errors"],
    }
    return final_response


async def complete_deepseek(request: LLMRequest, settings: LLMSettings) -> LLMResponse:
    """Call the DeepSeek chat completions API."""

    payload = _build_deepseek_payload(request, settings)
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    response = await _post_deepseek_with_transport_retry(
        _chat_completions_url(settings.base_url),
        payload=payload,
        headers=headers,
        timeout_seconds=settings.timeout_seconds,
    )
    response.raise_for_status()
    body = response.json()

    choice = body["choices"][0]
    message = choice.get("message") or {}
    text = message.get("content") or choice.get("text") or ""
    tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    if not str(text).strip() and not tool_calls:
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
            **({"tool_calls": tool_calls} if tool_calls else {}),
        },
    )


async def stream_deepseek(request: LLMRequest, settings: LLMSettings) -> AsyncIterator[StreamEvent]:
    """Call the DeepSeek chat completions streaming API."""

    payload = _build_deepseek_payload(request, settings)
    payload["stream"] = True
    emitted_text = False
    async for body in _iter_deepseek_stream_bodies(payload, settings):
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


async def stream_deepseek_with_native_tools(
    request: LLMRequest,
    settings: LLMSettings,
) -> AsyncIterator[StreamEvent]:
    """Stream model-selected tools followed by the final model answer."""

    payload = _build_deepseek_payload(request, settings)
    payload["stream"] = True
    tool_call_parts: dict[int, dict[str, str]] = {}
    first_text_parts: list[str] = []
    emitted_text = False
    async for body in _iter_deepseek_stream_bodies(payload, settings):
        choice = (body.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        _merge_stream_tool_call_parts(tool_call_parts, delta.get("tool_calls"))
        text = str(delta.get("content") or choice.get("text") or "")
        if not text:
            continue
        emitted_text = True
        first_text_parts.append(text)
        yield StreamEvent(
            event=EventType.MESSAGE_DELTA,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"text": text, "phase": "tool_selection"},
        )

    calls = _parse_native_tool_calls(_stream_tool_calls(tool_call_parts))
    if not calls:
        if not emitted_text:
            fallback_response = await _complete_deepseek_with_native_request_tools(request, settings)
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
        return

    followup_request, tool_metadata = await agent_runtime.execute_native_tool_calls(
        request,
        calls=calls,
        first_text="".join(first_text_parts),
    )
    async for event in stream_deepseek(followup_request, settings):
        if event.event == EventType.MESSAGE_DELTA:
            event = _copy_stream_event(
                event,
                data={
                    **dict(event.data),
                    "phase": "final_answer",
                    "tool_agent": tool_metadata["tool_agent"],
                    "tool_results": tool_metadata["tool_results"],
                    "tool_errors": tool_metadata["tool_errors"],
                },
            )
        elif event.event == EventType.MESSAGE_DONE:
            event = _copy_stream_event(
                event,
                data={
                    **dict(event.data),
                    "tool_agent": tool_metadata["tool_agent"],
                    "tool_results": tool_metadata["tool_results"],
                    "tool_errors": tool_metadata["tool_errors"],
                },
            )
        yield event


async def _complete_deepseek_with_native_request_tools(
    request: LLMRequest,
    settings: LLMSettings,
) -> LLMResponse:
    first_response = await complete_deepseek(request, settings)
    calls = _parse_native_tool_calls(first_response.metadata.get("tool_calls"))
    if not calls:
        return first_response
    followup_request, tool_metadata = await agent_runtime.execute_native_tool_calls(
        request,
        calls=calls,
        first_text=first_response.text,
    )
    final_response = await complete_deepseek(followup_request, settings)
    final_response.metadata = {**dict(final_response.metadata), **tool_metadata}
    return final_response


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


async def _post_deepseek_with_transport_retry(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> httpx.Response:
    """Retry once when a pooled HTTP connection became stale while idle."""

    for attempt in range(2):
        try:
            return await _deepseek_client(timeout_seconds).post(url, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise
        except httpx.TransportError:
            if attempt > 0:
                raise
    raise RuntimeError("unreachable")


async def _iter_deepseek_stream_bodies(
    payload: dict[str, Any],
    settings: LLMSettings,
) -> AsyncIterator[dict[str, Any]]:
    """Retry a stream once only when no upstream data has arrived yet."""

    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    for attempt in range(2):
        received_data = False
        try:
            client = _deepseek_client(settings.timeout_seconds)
            async with client.stream(
                "POST",
                _chat_completions_url(settings.base_url),
                json=payload,
                headers=headers,
            ) as response:
                await _raise_for_status_with_body(response)
                async for body in _iter_openai_sse(response):
                    received_data = True
                    yield body
            return
        except httpx.TimeoutException:
            raise
        except httpx.TransportError:
            if attempt > 0 or received_data:
                raise


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
        payload["tool_choice"] = "auto"
    if not _read_bool_env("LLM_THINKING_ENABLED", False):
        payload["thinking"] = {"type": "disabled"}
    return payload


def _message_to_deepseek_payload(message: Any) -> dict[str, Any]:
    payload = {"role": str(message.role), "content": message.content}
    if message.name:
        payload["name"] = message.name
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [_model_to_dict(item) for item in tool_calls]
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = str(tool_call_id)
    return payload


def _native_tool_calling_requested(request: LLMRequest) -> bool:
    mode = str(
        request.generation_config.get("tool_selection_mode")
        or os.getenv("LLM_TOOL_SELECTION_MODE", "native")
    ).strip().lower()
    return mode in {"native", "function", "tool_calls"} and bool(request.tools)


def _merge_stream_tool_call_parts(
    parts: dict[int, dict[str, str]],
    raw_calls: Any,
) -> None:
    if not isinstance(raw_calls, list):
        return
    for fallback_index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index", fallback_index))
        except (TypeError, ValueError):
            index = fallback_index
        item = parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if raw.get("id"):
            item["id"] = str(raw["id"])
        function = raw.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name"):
            item["name"] += str(function["name"])
        if function.get("arguments"):
            item["arguments"] += str(function["arguments"])


def _stream_tool_calls(parts: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id") or f"call_{index + 1}",
            "type": "function",
            "function": {
                "name": item.get("name") or "",
                "arguments": item.get("arguments") or "{}",
            },
        }
        for index, item in sorted(parts.items())
    ]


def _parse_native_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if not isinstance(raw_calls, list):
        return calls
    allowed = set(agent_runtime.tool_agent.registered_tools)
    max_calls = max(1, _read_int_env("LLM_TOOL_PLANNER_MAX_CALLS", 3))
    for index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip().lower().replace("-", "_")
        if name not in allowed:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                parsed_arguments = {}
        elif isinstance(arguments, dict):
            parsed_arguments = arguments
        else:
            parsed_arguments = {}
        if not isinstance(parsed_arguments, dict):
            parsed_arguments = {}
        call_id = str(raw.get("id") or f"call_{index + 1}")
        calls.append(
            {
                "id": call_id,
                "name": name,
                "args": parsed_arguments,
            }
        )
        if len(calls) >= max_calls:
            break
    return calls


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


def _stream_chunk_chars() -> int:
    return min(32, max(1, _read_int_env("LLM_STREAM_CHUNK_CHARS", 1)))


def _chat_completions_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def _extract_upstream_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except httpx.ResponseNotRead:
        return f"Upstream returned HTTP {response.status_code}"
    except ValueError:
        try:
            return response.text
        except httpx.ResponseNotRead:
            return f"Upstream returned HTTP {response.status_code}"
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            return str(error.get("message") or error)
        if error:
            return str(error)
    return str(body)


async def _raise_for_status_with_body(response: httpx.Response) -> None:
    if response.is_error:
        await response.aread()
    response.raise_for_status()


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


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


def _audit_request_input(request: LLMRequest) -> tuple[LLMRequest, SafetyDecision]:
    user_index = next(
        (
            index
            for index in range(len(request.messages) - 1, -1, -1)
            if request.messages[index].role == MessageRole.USER
        ),
        None,
    )
    if user_index is None:
        return request, safety_guard.audit_input("")
    decision = safety_guard.audit_input(request.messages[user_index].content)
    if decision.changed:
        safety_guard.log(stage="input", decision=decision, request_id=request.request_id)
    if decision.action != "mask":
        return request, decision

    messages = list(request.messages)
    original = messages[user_index]
    messages[user_index] = _copy_message_with_content(original, decision.content)
    metadata = dict(request.metadata)
    safety_metadata = metadata.get("safety")
    if not isinstance(safety_metadata, dict):
        safety_metadata = {}
    safety_metadata["input"] = decision.metadata()
    metadata["safety"] = safety_metadata
    return _copy_llm_request(request, messages=messages, metadata=metadata), decision


def _copy_message_with_content(message: Message, content: str) -> Message:
    update = {"content": content}
    if hasattr(message, "model_copy"):
        return message.model_copy(update=update)
    return message.copy(update=update)


def _copy_llm_request(
    request: LLMRequest,
    *,
    messages: list[Message],
    metadata: dict[str, Any],
) -> LLMRequest:
    update = {"messages": messages, "metadata": metadata}
    if hasattr(request, "model_copy"):
        return request.model_copy(update=update)
    return request.copy(update=update)


def _blocked_input_response(request: LLMRequest, decision: SafetyDecision) -> LLMResponse:
    return LLMResponse(
        request_id=request.request_id,
        text=decision.content,
        finish_reason="safety",
        usage=Usage(input_tokens=0, output_tokens=len(decision.content)),
        metadata={
            "provider": "local-safety",
            "model": "deterministic-rules",
            "safety": _safety_metadata(decision, safety_guard.audit_output(decision.content)),
        },
    )


def _audit_final_output(text: str, *, request_id: str) -> tuple[str, SafetyDecision]:
    decision = safety_guard.audit_output(text)
    if decision.changed:
        safety_guard.log(stage="output", decision=decision, request_id=request_id)
    return decision.content, decision


def _safety_metadata(input_decision: SafetyDecision, output_decision: SafetyDecision) -> dict[str, Any]:
    return {
        "engine": "local_rules",
        "uses_review_model": False,
        "input": input_decision.metadata(),
        "output": output_decision.metadata(),
    }


def _response_with_safety_metadata(
    response: LLMResponse,
    *,
    safe_text: str,
    input_decision: SafetyDecision,
    output_decision: SafetyDecision,
) -> LLMResponse:
    metadata = dict(response.metadata)
    metadata["safety"] = _safety_metadata(input_decision, output_decision)
    update = {
        "text": safe_text,
        "finish_reason": "safety" if output_decision.blocked else response.finish_reason,
        "metadata": metadata,
    }
    if hasattr(response, "model_copy"):
        return response.model_copy(update=update)
    return response.copy(update=update)


def _stream_event_with_text(event: StreamEvent, text: str) -> StreamEvent:
    data = dict(event.data)
    data["text"] = text
    return _copy_stream_event(event, data=data)


def _stream_text_event(
    request: LLMRequest,
    text: str,
    *,
    template: StreamEvent | None,
) -> StreamEvent:
    if template is not None:
        return _stream_event_with_text(template, text)
    return StreamEvent(
        event=EventType.MESSAGE_DELTA,
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        data={"text": text, "provider": "local-safety"},
    )


def _safety_delta_event(request: LLMRequest, text: str) -> StreamEvent:
    return _stream_text_event(request, text, template=None)


def _safety_done_event(request: LLMRequest, *, input_decision: SafetyDecision) -> StreamEvent:
    output_decision = safety_guard.audit_output(input_decision.content)
    return StreamEvent(
        event=EventType.MESSAGE_DONE,
        request_id=request.request_id,
        trace_id=request.trace_id,
        session_id=request.session_id,
        data={
            "message_id": request.request_id,
            "finish_reason": "safety",
            "safety": _safety_metadata(input_decision, output_decision),
        },
    )


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


async def _finalize_agent_response(
    turn: AgentTurn | None,
    response: LLMResponse,
    *,
    input_decision: SafetyDecision,
) -> LLMResponse:
    safe_text, output_decision = _audit_final_output(response.text, request_id=response.request_id)
    safe_response = _response_with_safety_metadata(
        response,
        safe_text=safe_text,
        input_decision=input_decision,
        output_decision=output_decision,
    )
    if turn is None:
        return safe_response
    persistence = await agent_runtime.after_turn(turn, safe_text)
    return _response_with_runtime_metadata(safe_response, turn, persistence)


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
