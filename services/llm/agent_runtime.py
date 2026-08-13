"""Agent, Skill, Tool, and Memory orchestration inside the LLM service."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from app_protocols import LLMRequest, Message, MessageRole
from services.llm.agents import LangChainRuntimeToolAgent
from services.llm.memory import MemoryStore
from services.llm.skills import AgentSkillContext, AgentSkillManager
from services.llm.tools import (
    BochaWebSearchTool,
    CalculatorTool,
    DictionaryTool,
    EncyclopediaTool,
    LocalTimeTool,
    OpenMeteoWeatherTool,
    UnitConverterTool,
    format_calculator_system_prompt,
    format_dictionary_system_prompt,
    format_encyclopedia_system_prompt,
    format_time_system_prompt,
    format_unit_system_prompt,
    format_weather_system_prompt,
    format_web_search_system_prompt,
)


@dataclass
class AgentTurn:
    """Prepared LLM turn and the context needed for post-turn persistence."""

    request: LLMRequest
    original_request: LLMRequest
    user_id: str
    user_text: str
    context: AgentSkillContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)


class LLMAgentRuntime:
    """Own the complete Agent -> Skills/Tools/Memory -> inference pipeline."""

    def __init__(self, memory_store: MemoryStore | None = None) -> None:
        self.memory_store = memory_store or MemoryStore()
        self.skill_manager = AgentSkillManager(self.memory_store)
        self.tool_agent = _build_tool_agent()

    async def initialize(self) -> None:
        """Initialize persistent dependencies before serving requests."""

        await self.memory_store.initialize()
        if _read_bool_env("LLM_MEMORY_REQUIRED", False):
            health = await self.memory_store.health()
            unavailable = [name for name in ("postgres", "redis") if health.get(name) != "ok"]
            if unavailable:
                raise RuntimeError(f"Required LLM memory dependencies are unavailable: {', '.join(unavailable)}")

    async def close(self) -> None:
        await self.memory_store.close()

    async def health(self) -> dict[str, Any]:
        memory = await self.memory_store.health()
        return {
            "agent": {
                "enabled": _read_bool_env("LLM_AGENT_ENABLED", True),
                "mode": self.skill_manager.agent_mode,
                "langgraph": self.skill_manager.tutor_graph is not None,
            },
            "skills": ["memory", "concept", "tutor"],
            "tools": self.tool_agent.registered_tools,
            "memory": memory,
        }

    async def prepare(self, request: LLMRequest) -> AgentTurn:
        user_text = _last_user_text(request)
        user_id = request.user_id.strip() or "anonymous"
        turn = AgentTurn(
            request=request,
            original_request=request,
            user_id=user_id,
            user_text=user_text,
        )
        if not user_text or not _agent_enabled_for_request(request):
            turn.metadata = {"agent": {"enabled": False}}
            return turn

        context_started = time.perf_counter()
        context = await self.skill_manager.prepare_context(
            user_id=user_id,
            session_id=request.session_id,
            query=user_text,
        )
        context_ms = round((time.perf_counter() - context_started) * 1000)

        tool_started = time.perf_counter()
        tool_result = await self.tool_agent.collect(
            query=user_text,
            default_timeout_seconds=_read_float_env("LLM_TOOL_TIMEOUT_SECONDS", 6.0),
        )
        tool_ms = round((time.perf_counter() - tool_started) * 1000)
        tool_prompts = [item.prompt for item in tool_result.outputs if item.prompt]
        system_prompt = self.skill_manager.render_system_prompt(context)
        if tool_prompts:
            system_prompt += "\n\n本轮可靠工具结果：\n" + "\n\n".join(tool_prompts)

        messages = [message for message in request.messages if message.role != MessageRole.SYSTEM]
        messages.insert(0, Message(role=MessageRole.SYSTEM, content=system_prompt))
        turn.request = _copy_request(request, messages=messages)
        turn.context = context
        turn.metadata = {
            "agent": context.skill_metadata,
            "skills": {
                "active": ["memory", "concept", "tutor"],
                "tutor_strategy": context.tutor_strategy,
            },
            "tools": {
                **tool_result.metadata,
                "results": {
                    item.name: item.result for item in tool_result.outputs if item.result is not None
                },
                "errors": {
                    item.name: item.error for item in tool_result.outputs if item.error
                },
            },
            "memory": {
                "user_id": user_id,
                "profile_fields": len(context.memory.profile),
                "history_count": len(context.memory.history),
                "memory_count": len(context.memory.memories),
                "concept_count": len(context.concepts),
            },
            "timings": {"context_ms": context_ms, "tools_ms": tool_ms},
        }
        return turn

    async def after_turn(self, turn: AgentTurn, assistant_text: str) -> dict[str, Any]:
        if not assistant_text.strip() or turn.context is None or not _agent_enabled_for_request(turn.original_request):
            return {}

        post_started = time.perf_counter()
        result = await self.skill_manager.after_turn(
            request_id=turn.original_request.request_id,
            trace_id=turn.original_request.trace_id,
            session_id=turn.original_request.session_id,
            user_id=turn.user_id,
            user_text=turn.user_text,
            assistant_text=assistant_text,
            context=turn.context,
            latency_ms=round((time.perf_counter() - turn.started_at) * 1000),
            timings=turn.metadata.get("timings"),
        )
        turn.metadata.setdefault("timings", {})["memory_write_ms"] = round(
            (time.perf_counter() - post_started) * 1000
        )
        return result


def _build_tool_agent() -> LangChainRuntimeToolAgent:
    time_tool = LocalTimeTool()
    calculator_tool = CalculatorTool()
    unit_tool = UnitConverterTool()
    dictionary_tool = DictionaryTool()
    encyclopedia_tool = EncyclopediaTool()
    weather_tool = OpenMeteoWeatherTool()
    web_search_tool = BochaWebSearchTool()
    return LangChainRuntimeToolAgent(
        time_tool=time_tool,
        calculator_tool=calculator_tool,
        unit_converter_tool=unit_tool,
        dictionary_tool=dictionary_tool,
        encyclopedia_tool=encyclopedia_tool,
        weather_tool=weather_tool,
        web_search_tool=web_search_tool,
        format_time=format_time_system_prompt,
        format_calculator=format_calculator_system_prompt,
        format_unit=format_unit_system_prompt,
        format_dictionary=format_dictionary_system_prompt,
        format_encyclopedia=format_encyclopedia_system_prompt,
        format_weather=format_weather_system_prompt,
        format_web_search=format_web_search_system_prompt,
    )


def _agent_enabled_for_request(request: LLMRequest) -> bool:
    if not _read_bool_env("LLM_AGENT_ENABLED", True):
        return False
    value = request.generation_config.get("agent_enabled", True)
    return bool(value)


def _last_user_text(request: LLMRequest) -> str:
    return next(
        (message.content for message in reversed(request.messages) if message.role == MessageRole.USER),
        "",
    )


def _copy_request(request: LLMRequest, *, messages: list[Message]) -> LLMRequest:
    update = {"messages": messages}
    if hasattr(request, "model_copy"):
        return request.model_copy(update=update)
    return request.copy(update=update)


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default
