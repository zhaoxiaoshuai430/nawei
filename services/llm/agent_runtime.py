"""Agent, Skill, Tool, and Memory orchestration inside the LLM service."""

# 单轮智能编排的核心：prepare 准备上下文和工具，main.py 调用模型，after_turn 保存结果。
# MemoryStore 管用户数据，KnowledgeClient 通过 HTTP 访问独立知识服务，SkillManager 组织提示词。
# 不在此处理音频或 WebSocket；返回 AgentTurn，让不同输出模式共享同一套决策上下文。

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app_protocols import LLMRequest, Message, MessageRole, ToolCall, ToolFunctionCall
from app_protocols.persona import XIAOWEI_PERSONA_VERSION
from services.knowledge.client import KnowledgeClient
from services.llm.agents import LangChainRuntimeToolAgent, RuntimeToolAgentResult
from services.llm.agents.runtime_tool_agent import RuntimeToolOutput
from services.llm.agents.tutor_graph import is_contextual_followup
from services.llm.memory import MemoryStore
from services.llm.conversation import (
    MEMORY_TOOL_NAMES,
    _normalize_answer_mode,
    _answer_mode_generation_config,
    _apply_answer_mode_prompt,
    _native_tool_policy_prompt,
    _requested_memory_mode,
    _should_skip_memory_prepare,
    _should_use_lightweight_memory,
    _memory_prepare_timeout_seconds,
    _direct_temporal_memory_answer,
    direct_tool_answer,
    _runtime_tool_planner_prompt,
    _parse_runtime_tool_plan,
)
from services.llm.safety import LocalSafetyGuard
from services.llm.skills import AgentSkillContext, AgentSkillManager
from services.llm.tools import (
    BochaWebSearchTool,
    CalculatorTool,
    DictionaryTool,
    LocalTimeTool,
    OpenMeteoWeatherTool,
    UnitConverterTool,
    format_calculator_system_prompt,
    format_dictionary_system_prompt,
    format_time_system_prompt,
    format_unit_system_prompt,
    format_weather_system_prompt,
    format_web_search_system_prompt,
)


@dataclass
class AgentTurn:
    """Prepared LLM turn and the context needed for post-turn persistence."""

    # request 可能已加入系统提示、历史和工具定义；original_request 保留原始参数，
    # 便于落库时使用真实用户输入，避免把检索材料或系统提示误存为用户消息。
    request: LLMRequest
    original_request: LLMRequest
    user_id: str
    user_text: str
    context: AgentSkillContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)
    direct_answer: str | None = None


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
        self.tool_agent = _build_tool_agent(self.memory_store)
        self._pending_weather_locations: dict[tuple[str, str], tuple[float, str]] = {}
        self._persistence_tasks: set[asyncio.Task[Any]] = set()
        self._persistence_locks: list[asyncio.Lock] = []
        self._persistence_loop: asyncio.AbstractEventLoop | None = None
        self._persistence_started: dict[asyncio.Task[Any], float] = {}
        self._persistence_failures = 0
        self._closing = False

    async def initialize(self) -> None:
        """Initialize persistent dependencies before serving requests."""

        self._closing = False
        await self.memory_store.initialize()
        if _read_bool_env("LLM_MEMORY_REQUIRED", False):
            deadline = time.monotonic() + max(
                0.0,
                _read_float_env("LLM_MEMORY_STARTUP_TIMEOUT_SECONDS", 30.0),
            )
            while True:
                health = await self.memory_store.health()
                unavailable = [name for name in ("postgres", "redis") if health.get(name) != "ok"]
                if not unavailable:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"Required LLM memory dependencies are unavailable: {', '.join(unavailable)}"
                    )
                await asyncio.sleep(min(0.5, remaining))
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
        # 先等待已登记的轮后写入，再关闭存储；此处的优雅退出不等于强制终止时也能保全数据。
        self._closing = True
        try:
            if self._persistence_tasks:
                async with asyncio.timeout(_read_float_env("MEMORY_SHUTDOWN_TIMEOUT_SECONDS", 130.0)):
                    await asyncio.gather(*list(self._persistence_tasks), return_exceptions=True)
        except TimeoutError:
            logging.getLogger(__name__).error("Memory shutdown deadline exceeded; unfinished writes were cancelled")
            raise
        finally:
            await self.tool_agent.close()
            await self.memory_store.close()
            await self.knowledge_client.close()

    def persistence_status(self) -> dict[str, Any]:
        return {
            "pending": len(self._persistence_tasks),
            "limit": max(1, int(_read_float_env("MEMORY_POST_TURN_MAX_PENDING", 64))),
            "failed": self._persistence_failures,
            "oldest_seconds": round(max((time.monotonic() - started for started in self._persistence_started.values()), default=0), 3),
        }

    async def health(self) -> dict[str, Any]:
        memory, knowledge = await asyncio.gather(self.memory_store.health(), self.knowledge_client.health())
        return {
            "agent": {
                "enabled": _read_bool_env("LLM_AGENT_ENABLED", True),
                "mode": self.skill_manager.agent_mode,
                "langgraph": self.skill_manager.tutor_graph is not None,
                "persona_version": XIAOWEI_PERSONA_VERSION,
                "conversation_version": "agent02-12-age-persona-fusion",
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
            "persistence": self.persistence_status(),
        }

    async def prepare(self, request: LLMRequest) -> AgentTurn:
        # 返回可直接交给推理层的请求及轮后上下文；本函数不生成最终模型回答。
        # Agent 被关闭时保留原请求，用于独立推理、提取任务和测试，避免递归套用 Agent。
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
        # 通用选项以 generation_config 覆盖 metadata；记忆读取模式和回答深度分别决定。
        # memory_mode=off 关闭本轮记忆读取/工具，不等价于删除历史或禁止轮后写入。
        options = {**request.metadata, **request.generation_config}
        answer_mode = _normalize_answer_mode(options.get("answer_mode"))
        memory_mode = "full" if _requested_memory_mode(options) == "full" else "lightweight"
        if _should_skip_memory_prepare(user_text=user_text, answer_mode=answer_mode, metadata=options):
            memory_mode = "off"
        elif (is_contextual_followup(user_text) and _requested_memory_mode(options) != "full") or _should_use_lightweight_memory(
            user_text=user_text, answer_mode=answer_mode, metadata=options,
        ):
            memory_mode = "lightweight"
        knowledge_base_id = str(
            request.metadata.get("knowledge_base_id")
            or request.generation_config.get("knowledge_base_id")
            or os.getenv("KNOWLEDGE_DEFAULT_BASE_ID", "default")
        ).strip()
        # 先准备画像、历史与知识上下文；读取预算按回答模式限制，降低简单问答的额外延迟。
        context = await self.skill_manager.prepare_conversation_context(
            user_id=user_id,
            session_id=request.session_id,
            query=user_text,
            knowledge_base_id=knowledge_base_id,
            memory_mode=memory_mode,
            memory_timeout=_memory_prepare_timeout_seconds(answer_mode, user_text=user_text, metadata=options),
        )
        context_ms = round((time.perf_counter() - context_started) * 1000)

        tool_started = time.perf_counter()
        native_tool_mode = _native_tool_calling_enabled(request)
        weather_resolver = self._preview_weather_query if request.defer_persistence else self._resolve_weather_query
        tool_query = await weather_resolver(user_id=user_id, session_id=request.session_id, user_text=user_text)
        weather_query = self.tool_agent.weather_tool.extract_query(tool_query)
        if weather_query is not None and not weather_query[0]:
            turn.direct_answer = "请告诉我你要查询的城市或地区，例如“北京今天天气怎么样”或“哈尔滨明天会下雨吗”。"
        if _requested_memory_mode(options) != "off" and not turn.direct_answer:
            turn.direct_answer = await _direct_temporal_memory_answer(
                self.memory_store, user_id=user_id, user_text=user_text,
            )
        runtime_tools, tool_schema_reason = _runtime_tool_definitions(
            self.tool_agent,
            query=user_text,
            intent=context.tutor_strategy.get("intent"),
            native_tool_mode=native_tool_mode,
        )
        if _requested_memory_mode(options) == "off":
            runtime_tools = [item for item in runtime_tools if item["function"]["name"] not in MEMORY_TOOL_NAMES]
        if turn.direct_answer:
            runtime_tools = []
        # 原生模式先暴露工具定义，待模型选择后执行；规则/规划模式在本阶段收集工具结果。
        # 缺少天气地点或命中确定性历史查询时，可直接回答，不必额外请求模型。
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
            planned_calls: list[dict[str, Any]] | None = [] if turn.direct_answer else None
            planner_error = None
            selection_mode = str(options.get("tool_selection_mode") or os.getenv("LLM_TOOL_SELECTION_MODE", "native")).lower()
            if planned_calls is None and selection_mode in {"llm", "model", "planner", "agent"}:
                try:
                    planned_calls = await self._plan_runtime_tools(request, tool_query)
                    if _requested_memory_mode(options) == "off":
                        planned_calls = [item for item in planned_calls if item["name"] not in MEMORY_TOOL_NAMES]
                except Exception as exc:
                    planner_error = type(exc).__name__
            tool_result = await self.tool_agent.collect(
                query=tool_query,
                default_timeout_seconds=_read_float_env("LLM_TOOL_TIMEOUT_SECONDS", 6.0),
                user_id=user_id if _requested_memory_mode(options) != "off" else "",
                session_id=request.session_id,
                planned_calls=planned_calls,
            )
            if planner_error:
                tool_result.metadata["planner_error"] = planner_error
                tool_result.metadata["planner_fallback"] = "rules"
            tool_result = self._audit_tool_result(tool_result, request_id=request.request_id)
            if not turn.direct_answer and not _read_bool_env("LLM_SEND_TOOL_RESULTS_TO_LLM", False):
                turn.direct_answer = direct_tool_answer(
                    {item.name: item.result for item in tool_result.outputs if item.result is not None},
                    {item.name: item.error for item in tool_result.outputs if item.error},
                )
        tool_ms = round((time.perf_counter() - tool_started) * 1000)
        tool_prompts = [item.prompt for item in tool_result.outputs if item.prompt]
        system_prompt = self.skill_manager.render_system_prompt(context)
        system_prompt = _apply_answer_mode_prompt(system_prompt, answer_mode)
        system_prompt += _reference_delivery_prompt(request)
        device_role = request.metadata.get("device_role")
        if device_role == "tutor":
            system_prompt += "\n机器人对话偏好：采用学习伙伴风格；仅在本轮长度和格式要求允许时解释关键概念并举例。"
        elif device_role == "companion":
            system_prompt += "\n机器人对话偏好：采用自然的陪伴助手风格，回答简洁，适合语音交流。"
        if tool_prompts:
            system_prompt += "\n\n本轮可靠工具结果：\n" + "\n\n".join(tool_prompts)
        if runtime_tools:
            system_prompt += "\n\n" + _native_tool_policy_prompt()
        if tool_query != user_text:
            system_prompt += "\n本轮用户补充了天气查询地点，完整查询为：" + tool_query

        messages = [message for message in request.messages if message.role != MessageRole.SYSTEM]
        messages.insert(0, Message(role=MessageRole.SYSTEM, content=system_prompt))
        turn.request = _copy_request(request, messages=messages, tools=runtime_tools)
        generation_config = {**request.generation_config, **_answer_mode_generation_config(answer_mode, options)}
        generation_config.pop("max_tokens", None)
        generation_config["answer_mode"] = answer_mode
        turn.request = turn.request.model_copy(update={"generation_config": generation_config})
        turn.context = context
        turn.metadata = {
            "answer_mode": answer_mode,
            "conversation_version": "agent02-12-age-persona-fusion",
            "direct_answer": turn.direct_answer is not None,
            "agent": context.skill_metadata,
            "skills": {
                "active": list(context.skill_metadata.get("skills", [])),
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
                **context.skill_metadata.get("memory_preparation", {}),
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

    def _weather_tool_query(self, *, user_id: str, session_id: str, user_text: str) -> str:
        now = time.monotonic()
        self._pending_weather_locations = {
            key: value for key, value in self._pending_weather_locations.items() if now - value[0] < 300
        }
        key = (user_id, session_id)
        weather_query = self.tool_agent.weather_tool.extract_query(user_text)
        if weather_query is not None:
            city, day = weather_query
            if city:
                self._pending_weather_locations.pop(key, None)
            elif session_id:
                if len(self._pending_weather_locations) >= 1024:
                    self._pending_weather_locations.pop(next(iter(self._pending_weather_locations)))
                self._pending_weather_locations[key] = (now, day)
            return user_text
        pending = self._pending_weather_locations.get(key)
        if pending is None:
            return user_text
        city = self.tool_agent.weather_tool.extract_location_reply(user_text)
        if city:
            self._pending_weather_locations.pop(key, None)
            day_text = {"tomorrow": "明天", "day_after_tomorrow": "后天"}.get(pending[1], "今天")
            return f"{city}{day_text}天气"
        if any(marker in user_text for marker in ("不用", "算了", "取消", "不查")):
            self._pending_weather_locations.pop(key, None)
        return user_text

    async def _preview_weather_query(self, *, user_id: str, session_id: str, user_text: str) -> str:
        if self.tool_agent.weather_tool.extract_query(user_text) is not None:
            return user_text
        pending = self._pending_weather_locations.get((user_id, session_id))
        day = pending[1] if pending and time.monotonic() - pending[0] < 300 else None
        if day is None and session_id:
            day = await self.memory_store.get_pending_weather_day(user_id=user_id, session_id=session_id)
        city = self.tool_agent.weather_tool.extract_location_reply(user_text) if day else None
        if city:
            day_text = {"tomorrow": "明天", "day_after_tomorrow": "后天"}.get(day or "", "今天")
            return f"{city}{day_text}天气"
        return user_text

    async def _resolve_weather_query(self, *, user_id: str, session_id: str, user_text: str) -> str:
        key = (user_id, session_id)
        query = self.tool_agent.weather_tool.extract_query(user_text)
        if session_id and query is None and key not in self._pending_weather_locations:
            day = await self.memory_store.get_pending_weather_day(user_id=user_id, session_id=session_id)
            if day:
                self._pending_weather_locations[key] = (time.monotonic(), day)
        resolved = self._weather_tool_query(user_id=user_id, session_id=session_id, user_text=user_text)
        if session_id and query is not None:
            await self.memory_store.set_pending_weather_day(
                user_id=user_id, session_id=session_id, day=query[1] if not query[0] else None,
            )
        elif session_id and (resolved != user_text or any(marker in user_text for marker in ("不用", "算了", "取消", "不查"))):
            await self.memory_store.set_pending_weather_day(user_id=user_id, session_id=session_id, day=None)
        return resolved

    async def _plan_runtime_tools(self, request: LLMRequest, query: str) -> list[dict[str, Any]]:
        from services.llm.main import complete_deepseek, load_settings

        planner_request = request.model_copy(update={
            "messages": [Message(role=MessageRole.SYSTEM, content=_runtime_tool_planner_prompt()),
                         Message(role=MessageRole.USER, content=query)],
            "tools": [], "stream": False, "generation_config": {"temperature": 0},
        })
        response = await asyncio.wait_for(
            complete_deepseek(planner_request, load_settings()),
            timeout=_read_float_env("LLM_TOOL_PLANNER_TIMEOUT_SECONDS", 6.0),
        )
        return _parse_runtime_tool_plan(response.text)

    async def execute_native_tool_calls(
        self,
        request: LLMRequest,
        *,
        calls: list[dict[str, Any]],
        first_text: str,
    ) -> tuple[LLMRequest, dict[str, Any]]:
        """Execute model-selected tools and build the second model-round request."""

        # 只执行本轮允许的工具；用户/会话身份取自可信请求，不能由模型参数指定。
        # 工具返回先经过安全过滤，再按 tool_call_id 配对，组成第二轮模型输入。
        options = {**request.metadata, **request.generation_config}
        allowed = {item["function"]["name"] for item in request.tools}

        async def execute_one(item: dict[str, Any]) -> RuntimeToolOutput:
            name = item["name"]
            if name not in allowed or (name in MEMORY_TOOL_NAMES and _requested_memory_mode(options) == "off"):
                return RuntimeToolOutput(name=name, prompt=None, result=None, error="Tool not enabled for this request.", elapsed_ms=0)
            result = await self.tool_agent.collect(
                query=_last_user_text(request),
                default_timeout_seconds=_read_float_env("LLM_TOOL_TIMEOUT_SECONDS", 6.0),
                planned_calls=[{"name": name, "args": item["args"]}],
                user_id=request.user_id.strip() or "anonymous", session_id=request.session_id,
            )
            return result.outputs[0] if result.outputs else RuntimeToolOutput(
                name=name, prompt=None, result=None, error="Invalid tool arguments.", elapsed_ms=0,
            )

        outputs = await asyncio.gather(*(execute_one(item) for item in calls))
        tool_result = RuntimeToolAgentResult(outputs=list(outputs), metadata={
            "selected_tools": [item["name"] for item in calls],
            "result_tools": [item.name for item in outputs if item.result is not None],
        })
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

        # 第二轮清空工具定义，让模型根据已取得的结果作答，避免无界递归调用工具。
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
        if turn.original_request.defer_persistence:
            return {"saved_turn": False, "reason": "voice_preview"}
        # 先写进程内近期历史，使同进程的快速追问能看到刚完成的对话。
        # history_only 只保存消息；full 继续更新记忆/学习状态；fast 模式可配置跳过写入。
        if not assistant_text.strip() or turn.context is None or not _agent_enabled_for_request(turn.original_request):
            return {}

        mode = "history_only" if turn.direct_answer is not None else "full"
        if turn.metadata.get("answer_mode") == "fast":
            mode = os.getenv("LLM_FAST_POST_TURN_MODE", "history_only").strip().lower()
        if mode in {"off", "none", "disabled"}:
            return {"mode": "off", "saved_turn": False}
        if self._closing:
            raise RuntimeError("Memory persistence is shutting down")
        post_started = time.perf_counter()
        user_message_id, assistant_message_id = uuid.uuid4().hex, uuid.uuid4().hex
        user_content = self.memory_store.cache_message(
            user_id=turn.user_id,
            session_id=turn.original_request.session_id,
            role="user",
            content=turn.user_text,
            message_id=user_message_id,
        )
        assistant_content = self.memory_store.cache_message(
            user_id=turn.user_id,
            session_id=turn.original_request.session_id,
            role="assistant",
            content=assistant_text,
            message_id=assistant_message_id,
        )
        loop = asyncio.get_running_loop()
        if self._persistence_loop is not loop:
            self._persistence_loop = loop
            self._persistence_locks = [asyncio.Lock() for _ in range(32)]
        # 固定数量的分片锁使同一用户会话的后台写入按序执行，避免用户/助手消息交错。
        # 这是进程内协调，不是跨多个 LLM worker 的分布式锁。
        lock = self._persistence_locks[hash((turn.user_id, turn.original_request.session_id)) % 32]

        async def persist_turn() -> dict[str, Any]:
            async with lock:
                for role, content, message_id in (("user", user_content, user_message_id),
                                                 ("assistant", assistant_content, assistant_message_id)):
                    if content:
                        await self.memory_store.persist_message(
                            user_id=turn.user_id, session_id=turn.original_request.session_id,
                            role=role, content=content, message_id=message_id,
                        )
                if mode in {"history", "history_only", "messages"}:
                    return {"mode": "history_only", "saved_turn": True}
                return await self.skill_manager.after_turn(
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

        async def bounded_persist_turn() -> dict[str, Any]:
            async with asyncio.timeout(_read_float_env("MEMORY_POST_TURN_TIMEOUT_SECONDS", 120.0)):
                return await persist_turn()

        if _read_bool_env("MEMORY_POST_TURN_ASYNC", True):
            # scheduled 仅确认任务已创建；同步模式才会在返回前等待 persist_turn 完成。
            # 后台异常由完成回调记录，健康/一致性测试应等待存储最终状态，而非立即比对。
            try:
                async with asyncio.timeout(_read_float_env("MEMORY_POST_TURN_TIMEOUT_SECONDS", 120.0)):
                    while len(self._persistence_tasks) >= self.persistence_status()["limit"]:
                        await asyncio.wait(self._persistence_tasks, return_when=asyncio.FIRST_COMPLETED)
                if self._closing:
                    raise RuntimeError("Memory persistence is shutting down")
            except (Exception, asyncio.CancelledError):
                self._persistence_failures += 1
                logging.getLogger(__name__).exception("Memory persistence admission failed")
                raise
            task = asyncio.create_task(bounded_persist_turn())
            self._persistence_tasks.add(task)
            self._persistence_started[task] = time.monotonic()
            task.add_done_callback(self._persistence_completed)
            result = {"mode": mode, "scheduled": True, "history_locally_cached": True}
        else:
            try:
                result = await bounded_persist_turn()
            except (Exception, asyncio.CancelledError):
                self._persistence_failures += 1
                logging.getLogger(__name__).exception("Synchronous memory persistence failed")
                raise
        turn.metadata.setdefault("timings", {})["memory_write_ms"] = round(
            (time.perf_counter() - post_started) * 1000
        )
        return result

    def _persistence_completed(self, task: asyncio.Task[Any]) -> None:
        self._persistence_tasks.discard(task)
        self._persistence_started.pop(task, None)
        if task.cancelled() or task.exception() is not None:
            self._persistence_failures += 1
            error_type = "CancelledError" if task.cancelled() else type(task.exception()).__name__
            logging.getLogger(__name__).error("Background memory persistence failed: %s", error_type)


def _build_tool_agent(memory_store: MemoryStore | None = None) -> LangChainRuntimeToolAgent:
    time_tool = LocalTimeTool()
    calculator_tool = CalculatorTool()
    unit_tool = UnitConverterTool()
    dictionary_tool = DictionaryTool()
    weather_tool = OpenMeteoWeatherTool()
    web_search_tool = BochaWebSearchTool()
    return LangChainRuntimeToolAgent(
        time_tool=time_tool,
        calculator_tool=calculator_tool,
        unit_converter_tool=unit_tool,
        dictionary_tool=dictionary_tool,
        memory_store=memory_store,
        weather_tool=weather_tool,
        web_search_tool=web_search_tool,
        format_time=format_time_system_prompt,
        format_calculator=format_calculator_system_prompt,
        format_unit=format_unit_system_prompt,
        format_dictionary=format_dictionary_system_prompt,
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


def _runtime_tool_definitions(
    tool_agent: LangChainRuntimeToolAgent,
    *,
    query: str,
    intent: Any,
    native_tool_mode: bool,
) -> tuple[list[dict[str, Any]], str]:
    if not native_tool_mode:
        return [], "native_tool_calling_disabled"

    return tool_agent.openai_tool_definitions(), "target_branch:all_enabled"


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
