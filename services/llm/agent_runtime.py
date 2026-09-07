"""Agent, Skill, Tool, and Memory orchestration inside the LLM service."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from app_protocols import LLMRequest, Message, MessageRole, ToolCall, ToolFunctionCall
from app_protocols.persona import XIAOWEI_PERSONA_VERSION
from services.knowledge.client import KnowledgeClient
from services.llm.agents import LangChainRuntimeToolAgent, RuntimeToolAgentResult
from services.llm.agents.runtime_tool_agent import RuntimeToolOutput
from services.llm.agents.tutor_graph import is_contextual_followup
from services.llm.memory import MemoryStore
from services.llm.safety import LocalSafetyGuard
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

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        knowledge_client: KnowledgeClient | None = None,
        safety_guard: LocalSafetyGuard | None = None,
    ) -> None:
        self.memory_store = memory_store or MemoryStore()
        self.knowledge_client = knowledge_client or KnowledgeClient()
        self.safety_guard = safety_guard or LocalSafetyGuard()
        self.skill_manager = AgentSkillManager(self.memory_store, self.knowledge_client)
        self.tool_agent = _build_tool_agent()

    async def initialize(self) -> None:
        """Initialize persistent dependencies before serving requests."""

        await self.memory_store.initialize()
        if _read_bool_env("LLM_MEMORY_REQUIRED", False):
            health = await self.memory_store.health()
            unavailable = [name for name in ("postgres", "redis") if health.get(name) != "ok"]
            if unavailable:
                raise RuntimeError(f"Required LLM memory dependencies are unavailable: {', '.join(unavailable)}")
        knowledge = await self.knowledge_client.health()
        if _read_bool_env("KNOWLEDGE_REQUIRED", False):
            for _ in range(20):
                if knowledge.get("status") == "ok":
                    break
                await asyncio.sleep(0.5)
                knowledge = await self.knowledge_client.health()
            else:
                raise RuntimeError("Required knowledge service is unavailable.")

    async def close(self) -> None:
        await self.memory_store.close()
        await self.knowledge_client.close()

    async def health(self) -> dict[str, Any]:
        memory = await self.memory_store.health()
        knowledge = await self.knowledge_client.health()
        return {
            "agent": {
                "enabled": _read_bool_env("LLM_AGENT_ENABLED", True),
                "mode": self.skill_manager.agent_mode,
                "langgraph": self.skill_manager.tutor_graph is not None,
                "persona_version": XIAOWEI_PERSONA_VERSION,
            },
            "skills": ["memory", "concept", "knowledge", "tutor"],
            "tools": self.tool_agent.registered_tools,
            "tool_selection_mode": os.getenv("LLM_TOOL_SELECTION_MODE", "native").strip().lower(),
            "safety": {
                "enabled": self.safety_guard.enabled(),
                "input": self.safety_guard.enabled("input"),
                "tool_result": self.safety_guard.enabled("tool_result"),
                "output": self.safety_guard.enabled("output"),
                "audit_log": _read_bool_env("SAFETY_AUDIT_LOG_ENABLED", True),
            },
            "memory": memory,
            "knowledge": knowledge,
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
        knowledge_base_id = str(
            request.metadata.get("knowledge_base_id")
            or request.generation_config.get("knowledge_base_id")
            or os.getenv("KNOWLEDGE_DEFAULT_BASE_ID", "default")
        ).strip()
        prepare_context = (
            self.skill_manager.prepare_short_term_context
            if is_contextual_followup(user_text)
            else self.skill_manager.prepare_context
        )
        context = await prepare_context(
            user_id=user_id,
            session_id=request.session_id,
            query=user_text,
            knowledge_base_id=knowledge_base_id,
        )
        context_ms = round((time.perf_counter() - context_started) * 1000)

        tool_started = time.perf_counter()
        native_tool_mode = _native_tool_calling_enabled(request)
        runtime_tools, tool_schema_reason = _runtime_tool_definitions(
            self.tool_agent,
            query=user_text,
            intent=context.tutor_strategy.get("intent"),
            native_tool_mode=native_tool_mode,
        )
        if native_tool_mode:
            tool_result = RuntimeToolAgentResult(
                outputs=[],
                metadata={
                    "agent": "LangChainRuntimeToolAgent",
                    "selector": "native_tool_calling",
                    "schema_selection_reason": tool_schema_reason,
                    "available_tools": list(self.tool_agent.registered_tools),
                    "registered_tools": [
                        str(item.get("function", {}).get("name") or "") for item in runtime_tools
                    ],
                    "selected_tools": [],
                    "result_tools": [],
                    "safety": {"engine": "local_rules", "changed_count": 0, "events": []},
                },
            )
        else:
            tool_result = await self.tool_agent.collect(
                query=user_text,
                default_timeout_seconds=_read_float_env("LLM_TOOL_TIMEOUT_SECONDS", 6.0),
            )
            tool_result = self._audit_tool_result(tool_result, request_id=request.request_id)
        tool_ms = round((time.perf_counter() - tool_started) * 1000)
        tool_prompts = [item.prompt for item in tool_result.outputs if item.prompt]
        system_prompt = self.skill_manager.render_system_prompt(context)
        system_prompt += _reference_delivery_prompt(request)
        if tool_prompts:
            system_prompt += "\n\n本轮可靠工具结果：\n" + "\n\n".join(tool_prompts)
        if runtime_tools:
            system_prompt += "\n\n" + _native_tool_policy_prompt()

        messages = [message for message in request.messages if message.role != MessageRole.SYSTEM]
        messages.insert(0, Message(role=MessageRole.SYSTEM, content=system_prompt))
        turn.request = _copy_request(request, messages=messages, tools=runtime_tools)
        turn.context = context
        turn.metadata = {
            "agent": context.skill_metadata,
            "skills": {
                "active": ["memory", "concept", "knowledge", "tutor"],
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
            "knowledge": {
                "knowledge_base_id": context.knowledge.knowledge_base_id,
                "chunk_count": len(context.knowledge.chunks),
                "relation_count": len(context.knowledge.relations),
                "chunk_ids": context.knowledge_chunk_ids,
                "anchor_terms": list(context.knowledge.metadata.get("anchor_terms") or []),
                "grounding_status": context.knowledge.metadata.get("grounding_status"),
                "retrieved_chunk_count": context.knowledge.metadata.get("retrieved_chunk_count"),
                "graph_match_level": context.knowledge.metadata.get("graph_match_level"),
                "resolved_entity_names": list(
                    context.knowledge.metadata.get("resolved_entity_names") or []
                ),
                "relation_group_counts": dict(
                    context.knowledge.metadata.get("relation_group_counts") or {}
                ),
                "qdrant_filters": dict(context.knowledge.metadata.get("qdrant_filters") or {}),
                "retrieval_skipped": bool(
                    context.knowledge.metadata.get("retrieval_skipped")
                ),
                "skip_reason": context.knowledge.metadata.get("skip_reason"),
            },
            "timings": {"context_ms": context_ms, "tools_ms": tool_ms},
        }
        return turn

    async def execute_native_tool_calls(
        self,
        request: LLMRequest,
        *,
        calls: list[dict[str, Any]],
        first_text: str,
    ) -> tuple[LLMRequest, dict[str, Any]]:
        """Execute model-selected tools and build the second model-round request."""

        tool_result = await self.tool_agent.collect(
            query=_last_user_text(request),
            default_timeout_seconds=_read_float_env("LLM_TOOL_TIMEOUT_SECONDS", 6.0),
            planned_calls=[{"name": item["name"], "args": item["args"]} for item in calls],
        )
        tool_result = self._audit_tool_result(tool_result, request_id=request.request_id)
        raw_tool_calls = [
            ToolCall(
                id=item["id"],
                function=ToolFunctionCall(
                    name=item["name"],
                    arguments=json.dumps(item["args"], ensure_ascii=False, separators=(",", ":")),
                ),
            )
            for item in calls
        ]
        followup_messages = list(request.messages)
        followup_messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=first_text,
                tool_calls=raw_tool_calls,
            )
        )
        tool_results: dict[str, Any] = {}
        tool_errors: dict[str, str] = {}
        for index, item in enumerate(calls):
            output = tool_result.outputs[index] if index < len(tool_result.outputs) else None
            result = output.result if output is not None else None
            error = output.error if output is not None else f"Tool {item['name']} is unavailable or invalid."
            metadata_key = item["name"]
            if metadata_key in tool_results or metadata_key in tool_errors:
                metadata_key = f"{item['name']}:{item['id']}"
            if result is not None:
                tool_results[metadata_key] = result
            if error:
                tool_errors[metadata_key] = error
            followup_messages.append(
                Message(
                    role=MessageRole.TOOL,
                    name=item["name"],
                    tool_call_id=item["id"],
                    content=json.dumps(
                        {"result": result, "error": error},
                        ensure_ascii=False,
                        default=str,
                    ),
                )
            )

        followup_request = _copy_request(request, messages=followup_messages, tools=[])
        metadata = {
            "tool_agent": {
                **tool_result.metadata,
                "selector": "native_tool_calling",
                "selected_tools": [item["name"] for item in calls],
                "model_rounds": 2,
            },
            "tool_results": tool_results,
            "tool_errors": tool_errors,
        }
        return followup_request, metadata

    def _audit_tool_result(
        self,
        tool_result: RuntimeToolAgentResult,
        *,
        request_id: str,
    ) -> RuntimeToolAgentResult:
        outputs: list[RuntimeToolOutput] = []
        events: list[dict[str, Any]] = []
        for item in tool_result.outputs:
            decision = self.safety_guard.audit_tool_result(
                item.name,
                prompt=item.prompt,
                result=item.result,
                error=item.error,
            )
            if decision.action != "allow":
                self.safety_guard.log(
                    stage="tool_result",
                    decision=decision,
                    request_id=request_id,
                    tool_name=item.name,
                )
                events.append({"tool": item.name, **decision.metadata()})
            outputs.append(
                RuntimeToolOutput(
                    name=item.name,
                    prompt=decision.prompt,
                    result=decision.result,
                    error=decision.error,
                    elapsed_ms=item.elapsed_ms,
                )
            )
        metadata = dict(tool_result.metadata)
        metadata["safety"] = {
            "engine": "local_rules",
            "changed_count": len(events),
            "events": events,
        }
        return RuntimeToolAgentResult(outputs=outputs, metadata=metadata)

    async def after_turn(self, turn: AgentTurn, assistant_text: str) -> dict[str, Any]:
        if not assistant_text.strip() or turn.context is None or not _agent_enabled_for_request(turn.original_request):
            return {}

        post_started = time.perf_counter()
        user_content = self.memory_store.cache_message(
            user_id=turn.user_id,
            session_id=turn.original_request.session_id,
            role="user",
            content=turn.user_text,
        )
        assistant_content = self.memory_store.cache_message(
            user_id=turn.user_id,
            session_id=turn.original_request.session_id,
            role="assistant",
            content=assistant_text,
        )
        persistence_tasks = [
            asyncio.create_task(
                self.memory_store.persist_message(
                    user_id=turn.user_id,
                    session_id=turn.original_request.session_id,
                    role=role,
                    content=content,
                )
            )
            for role, content in (("user", user_content), ("assistant", assistant_content))
            if content
        ]
        try:
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
                history_already_saved=True,
            )
        except Exception:
            if persistence_tasks:
                await asyncio.gather(*persistence_tasks, return_exceptions=True)
            raise
        if persistence_tasks:
            await asyncio.gather(*persistence_tasks)
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


def _copy_request(
    request: LLMRequest,
    *,
    messages: list[Message],
    tools: list[dict[str, Any]] | None = None,
) -> LLMRequest:
    update: dict[str, Any] = {"messages": messages}
    if tools is not None:
        update["tools"] = tools
    if hasattr(request, "model_copy"):
        return request.model_copy(update=update)
    return request.copy(update=update)


def _native_tool_calling_enabled(request: LLMRequest) -> bool:
    configured = str(
        request.generation_config.get("tool_selection_mode")
        or os.getenv("LLM_TOOL_SELECTION_MODE", "native")
    ).strip().lower()
    provider = os.getenv("LLM_PROVIDER", "stub").strip().lower()
    return configured in {"native", "function", "tool_calls"} and provider != "stub"


_NO_TOOL_SCHEMA_CATEGORIES = {
    "crisis_support",
    "academic_integrity",
    "account_security",
    "identity",
    "preference",
    "casual",
    "daily_life",
    "support",
    "realtime",
}


def _runtime_tool_definitions(
    tool_agent: LangChainRuntimeToolAgent,
    *,
    query: str,
    intent: Any,
    native_tool_mode: bool,
) -> tuple[list[dict[str, Any]], str]:
    if not native_tool_mode:
        return [], "native_tool_calling_disabled"

    intent_data = intent if isinstance(intent, dict) else {}
    category = str(intent_data.get("category") or "explain")
    if category not in _NO_TOOL_SCHEMA_CATEGORIES:
        return tool_agent.openai_tool_definitions(), f"intent:{category}:all"

    suggested_names = tool_agent.suggested_tool_names(
        query,
        default_timeout_seconds=_read_float_env("LLM_TOOL_TIMEOUT_SECONDS", 6.0),
    )
    if suggested_names:
        return (
            tool_agent.openai_tool_definitions(suggested_names),
            f"intent:{category}:deterministic_candidates",
        )
    return [], f"intent:{category}:skipped"


def _native_tool_policy_prompt() -> str:
    return (
        "Tool selection has the highest priority for factual reliability. You MUST call weather for weather forecasts, "
        "calculator for explicit arithmetic, time for the current date or time, unit_converter for conversions, "
        "dictionary for English word definitions, encyclopedia for stable reference facts, and web_search for current facts. "
        "Never invent facts available from a tool. Answer directly without tools when none is needed."
    )


def _reference_delivery_prompt(request: LLMRequest) -> str:
    if not bool(request.metadata.get("include_references", False)):
        return ""
    return (
        "\n\n语音回答的可视来源要求：如果本轮知识上下文提供了[资料N]，"
        "可视文本必须在结尾单独列出“参考来源”，正文不要出现[资料N]或【资料N】等行内编号。"
        "参考来源仅供屏幕显示，必须与正文分行，不要把资料编号、机构、标题、路径或链接"
        "混入朗读正文；不得编造未提供的来源。音频层会自动跳过该来源区块。"
    )


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default
