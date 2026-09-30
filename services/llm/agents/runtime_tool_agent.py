"""LangChain/LangGraph runtime tool agent for LLM deterministic tools."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, NotRequired, TypedDict

try:  # pragma: no cover - optional dependency in lightweight installs
    from langchain_core.runnables import RunnableLambda
except ImportError:  # pragma: no cover
    RunnableLambda = None  # type: ignore[misc,assignment]

try:  # pragma: no cover - optional dependency in lightweight installs
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover
    StructuredTool = None  # type: ignore[misc,assignment]

try:  # pragma: no cover - optional dependency in lightweight installs
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = "__end__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[misc,assignment]


Formatter = Callable[[Any], str]


@dataclass(frozen=True)
class RuntimeToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 1.0


@dataclass(frozen=True)
class RuntimeToolOutput:
    name: str
    prompt: str | None
    result: dict[str, Any] | None
    error: str | None
    elapsed_ms: int


@dataclass(frozen=True)
class RuntimeToolAgentResult:
    outputs: list[RuntimeToolOutput]
    metadata: dict[str, Any]

    def as_prompt_tuples(self) -> list[tuple[str, str | None, dict[str, Any] | None, str | None]]:
        return [(item.name, item.prompt, item.result, item.error) for item in self.outputs]


class RuntimeToolAgentState(TypedDict):
    query: str
    default_timeout_seconds: float
    user_id: NotRequired[str]
    session_id: NotRequired[str]
    selected_tools: NotRequired[list[RuntimeToolCall]]
    outputs: NotRequired[list[RuntimeToolOutput]]
    metadata: NotRequired[dict[str, Any]]


class LangChainRuntimeToolAgent:
    """Select and run LLM tools through LangChain tools and LangGraph nodes."""

    def __init__(
        self,
        *,
        time_tool: Any,
        calculator_tool: Any,
        unit_converter_tool: Any,
        dictionary_tool: Any,
        weather_tool: Any,
        format_time: Formatter,
        format_calculator: Formatter,
        format_unit: Formatter,
        format_dictionary: Formatter,
        format_weather: Formatter,
        format_web_search: Formatter | None = None,
        web_search_tool: Any | None = None,
        memory_store: Any | None = None,
    ) -> None:
        self.time_tool = time_tool
        self.calculator_tool = calculator_tool
        self.unit_converter_tool = unit_converter_tool
        self.dictionary_tool = dictionary_tool
        self.weather_tool = weather_tool
        self.web_search_tool = web_search_tool
        self.memory_store = memory_store
        self.format_time = format_time
        self.format_calculator = format_calculator
        self.format_unit = format_unit
        self.format_dictionary = format_dictionary
        self.format_weather = format_weather
        self.format_web_search = format_web_search
        self.agent_mode = os.getenv("LLM_TOOL_AGENT_MODE", "langgraph").strip().lower() or "langgraph"
        self.uses_langchain = self.agent_mode not in {"off", "legacy", "false", "0"} and StructuredTool is not None and RunnableLambda is not None
        self.uses_langgraph = self.uses_langchain and self.agent_mode in {"langgraph", "agent", "tools"} and StateGraph is not None
        self._tools = self._build_tools()
        self._select_chain = RunnableLambda(self._select_tool_calls) if self.uses_langchain and RunnableLambda is not None else None
        self._graph = self._compile_graph() if self.uses_langgraph and StateGraph is not None else None

    @property
    def registered_tools(self) -> list[str]:
        return [name for name in _TOOL_ORDER if self._tool_enabled(name)]

    async def close(self) -> None:
        for tool in (self.dictionary_tool, self.weather_tool, self.web_search_tool):
            close = getattr(tool, "aclose", None)
            if close is not None:
                await close()

    def suggested_tool_names(self, query: str, *, default_timeout_seconds: float = 6.0) -> list[str]:
        return [call.name for call in self._select_tool_calls(
            {"query": query, "default_timeout_seconds": default_timeout_seconds}
        )]

    def openai_tool_definitions(self, names: list[str] | set[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Return enabled runtime tools in OpenAI-compatible function format."""

        definitions = _openai_tool_definitions()
        return [item for item in definitions if self._tool_enabled(str(item["function"]["name"])) and (names is None or item["function"]["name"] in names)]

    async def collect(
        self,
        *,
        query: str,
        default_timeout_seconds: float,
        planned_calls: Sequence[dict[str, Any] | RuntimeToolCall] | None = None,
        user_id: str = "",
        session_id: str = "",
    ) -> RuntimeToolAgentResult:
        state: RuntimeToolAgentState = {
            "query": query,
            "default_timeout_seconds": default_timeout_seconds,
            "user_id": user_id,
            "session_id": session_id,
        }
        if planned_calls is not None:
            selected_tools = self._normalize_planned_tool_calls(
                planned_calls,
                query=query,
                default_timeout=default_timeout_seconds,
                user_id=user_id,
                session_id=session_id,
            )
            state = {
                **state,
                "selected_tools": selected_tools,
                "metadata": self._selection_metadata(selected_tools, selector="llm_planner"),
            }
            state = await self._run_tools_node(state)
            state = await self._assemble_node(state)
            return RuntimeToolAgentResult(outputs=state.get("outputs", []), metadata=state.get("metadata", {}))

        if self._graph is not None:
            state = await self._graph.ainvoke(state)
        else:
            state = await self._select_tools_node(state)
            state = await self._run_tools_node(state)
            state = await self._assemble_node(state)
        return RuntimeToolAgentResult(outputs=state.get("outputs", []), metadata=state.get("metadata", {}))

    def _compile_graph(self) -> Any:
        graph = StateGraph(RuntimeToolAgentState)
        graph.add_node("select_tools", self._select_tools_node)
        graph.add_node("run_tools", self._run_tools_node)
        graph.add_node("assemble_results", self._assemble_node)
        graph.set_entry_point("select_tools")
        graph.add_edge("select_tools", "run_tools")
        graph.add_edge("run_tools", "assemble_results")
        graph.add_edge("assemble_results", END)
        return graph.compile()

    async def _select_tools_node(self, state: RuntimeToolAgentState) -> RuntimeToolAgentState:
        if self._select_chain is not None:
            selected_tools = await self._select_chain.ainvoke(state)
        else:
            selected_tools = self._select_tool_calls(state)
        metadata = self._selection_metadata(selected_tools, selector="rules")
        return {**state, "selected_tools": selected_tools, "metadata": metadata}

    async def _run_tools_node(self, state: RuntimeToolAgentState) -> RuntimeToolAgentState:
        selected_tools = state.get("selected_tools", [])
        if not selected_tools:
            return {**state, "outputs": []}
        outputs = await asyncio.gather(*(self._run_one_tool(call) for call in selected_tools))
        return {**state, "outputs": list(outputs)}

    async def _assemble_node(self, state: RuntimeToolAgentState) -> RuntimeToolAgentState:
        outputs = state.get("outputs", [])
        metadata = dict(state.get("metadata", {}))
        metadata["result_tools"] = [item.name for item in outputs if item.result is not None]
        metadata["error_tools"] = [item.name for item in outputs if item.error]
        metadata["tool_timings_ms"] = {item.name: item.elapsed_ms for item in outputs}
        metadata["tool_wall_ms"] = max((item.elapsed_ms for item in outputs), default=0)
        return {**state, "metadata": metadata}

    def _select_tool_calls(self, state: RuntimeToolAgentState) -> list[RuntimeToolCall]:
        query = state["query"]
        default_timeout = state["default_timeout_seconds"]
        calls: list[RuntimeToolCall] = []

        if self.memory_store is not None and state.get("user_id") and _read_bool_env("MEMORY_TOOLS_ENABLED", True):
            memory_call = self._select_memory_tool_call(query, state)
            if memory_call is not None:
                calls.append(memory_call)

        if _read_bool_env("TIME_TOOL_ENABLED", True) and self.time_tool.should_run(query):
            calls.append(RuntimeToolCall("time", {}, self._timeout_for("TIME", default_timeout)))

        if _read_bool_env("CALCULATOR_TOOL_ENABLED", True):
            expression = self.calculator_tool.extract_expression(query)
            if expression:
                calls.append(RuntimeToolCall("calculator", {"text": query}, self._timeout_for("CALCULATOR", default_timeout)))

        if _read_bool_env("UNIT_TOOL_ENABLED", True) and self._has_unit_conversion(query):
            calls.append(RuntimeToolCall("unit_converter", {"text": query}, self._timeout_for("UNIT", default_timeout)))

        if _read_bool_env("DICTIONARY_TOOL_ENABLED", True):
            word = self.dictionary_tool.extract_word(query)
            if word:
                calls.append(RuntimeToolCall("dictionary", {"word": word}, self._timeout_for("DICTIONARY", default_timeout)))

        web_query = self.web_search_tool.extract_query(query) if self.web_search_tool is not None and _read_bool_env("BOCHA_WEB_SEARCH_ENABLED", True) else None
        if web_query:
            calls.append(
                RuntimeToolCall(
                    "web_search",
                    {"query": web_query},
                    self._timeout_for("BOCHA_SEARCH", default_timeout),
                )
            )

        if _read_bool_env("WEATHER_TOOL_ENABLED", True):
            weather_query = self.weather_tool.extract_query(query)
            if weather_query:
                city, day = weather_query
                if city:
                    weather_default = max(default_timeout, float(getattr(self.weather_tool, "timeout_seconds", default_timeout)) + 1.0)
                    calls.append(RuntimeToolCall("weather", {"city": city, "day": day}, self._timeout_for("WEATHER", weather_default)))

        return _dedupe_tool_calls(calls)

    def _select_memory_tool_call(self, query: str, state: RuntimeToolAgentState) -> RuntimeToolCall | None:
        """Rules fallback for memory tools when the LLM planner is unavailable."""

        if _looks_like_concept_memory_query(query):
            return RuntimeToolCall(
                "concept_lookup",
                self._memory_context_args({"query": query}, state),
                self._timeout_for("MEMORY", state["default_timeout_seconds"]),
            )
        if _looks_like_history_memory_query(query):
            return RuntimeToolCall(
                "history_search",
                self._memory_context_args({"query": query}, state),
                self._timeout_for("MEMORY", state["default_timeout_seconds"]),
            )
        if _looks_like_profile_memory_query(query):
            return RuntimeToolCall(
                "profile_lookup",
                self._memory_context_args({}, state),
                self._timeout_for("MEMORY", state["default_timeout_seconds"]),
            )
        if _looks_like_long_term_memory_query(query):
            return RuntimeToolCall(
                "memory_search",
                self._memory_context_args({"query": query}, state),
                self._timeout_for("MEMORY", state["default_timeout_seconds"]),
            )
        return None

    def _normalize_planned_tool_calls(
        self,
        planned_calls: Sequence[dict[str, Any] | RuntimeToolCall],
        *,
        query: str,
        default_timeout: float,
        user_id: str = "",
        session_id: str = "",
    ) -> list[RuntimeToolCall]:
        calls: list[RuntimeToolCall] = []
        for item in planned_calls:
            if isinstance(item, RuntimeToolCall):
                name = item.name
                args = dict(item.args)
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("tool") or "").strip()
                raw_args = item.get("args") or item.get("arguments") or {}
                args = dict(raw_args) if isinstance(raw_args, dict) else {}
            else:
                continue

            name = name.lower().replace("-", "_")
            normalized = self._normalize_tool_args(
                name,
                args,
                query=query,
                user_id=user_id,
                session_id=session_id,
            )
            if normalized is None:
                continue
            calls.append(RuntimeToolCall(name, normalized, self._timeout_for(_TIMEOUT_PREFIX_BY_TOOL[name], default_timeout)))
        return calls

    def _normalize_tool_args(
        self,
        name: str,
        args: dict[str, Any],
        *,
        query: str,
        user_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any] | None:
        if name not in _TOOL_ORDER or not self._tool_enabled(name):
            return None
        if name == "time":
            return {}
        if name == "calculator":
            text = str(args.get("text") or args.get("expression") or query).strip()
            return {"text": text} if text else None
        if name == "unit_converter":
            text = str(args.get("text") or query).strip()
            return {"text": text} if text else None
        if name == "dictionary":
            word = str(args.get("word") or "").strip()
            if not word:
                word = str(self.dictionary_tool.extract_word(query) or "").strip()
            return {"word": word} if word else None
        if name == "web_search":
            tool_query = str(args.get("query") or "").strip()
            if not tool_query and self.web_search_tool is not None:
                tool_query = str(self.web_search_tool.extract_query(query) or "").strip()
            return {"query": tool_query} if tool_query else None
        if name == "weather":
            city = str(args.get("city") or args.get("location") or "").strip()
            day = str(args.get("day") or "today").strip() or "today"
            weather_query = self.weather_tool.extract_query(query)
            if weather_query and not city:
                parsed_city, parsed_day = weather_query
                city, day = parsed_city, parsed_day
            return {"city": city, "day": day} if city else None
        if name in {"memory_search", "history_search", "concept_lookup"}:
            tool_query = str(args.get("query") or args.get("text") or query).strip()
            return self._memory_context_args({"query": tool_query}, {"user_id": user_id, "session_id": session_id}) if tool_query else None
        if name == "profile_lookup":
            return self._memory_context_args({}, {"user_id": user_id, "session_id": session_id})
        return None

    def _memory_context_args(self, args: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **args,
            "user_id": str(state.get("user_id") or ""),
            "session_id": str(state.get("session_id") or ""),
        }

    def _tool_enabled(self, name: str) -> bool:
        if name == "time":
            return _read_bool_env("TIME_TOOL_ENABLED", True)
        if name == "calculator":
            return _read_bool_env("CALCULATOR_TOOL_ENABLED", True)
        if name == "unit_converter":
            return _read_bool_env("UNIT_TOOL_ENABLED", True)
        if name == "dictionary":
            return _read_bool_env("DICTIONARY_TOOL_ENABLED", True)
        if name == "web_search":
            return self.web_search_tool is not None and _read_bool_env("BOCHA_WEB_SEARCH_ENABLED", True)
        if name == "weather":
            return _read_bool_env("WEATHER_TOOL_ENABLED", True)
        if name in {"memory_search", "history_search", "profile_lookup", "concept_lookup"}:
            return self.memory_store is not None and _read_bool_env("MEMORY_TOOLS_ENABLED", True)
        return False

    def _selection_metadata(self, selected_tools: list[RuntimeToolCall], *, selector: str) -> dict[str, Any]:
        return {
            "agent": "LangChainRuntimeToolAgent",
            "agent_mode": "langgraph_tools" if self.uses_langgraph else self.agent_mode,
            "selector": selector,
            "langchain_enabled": self.uses_langchain,
            "langgraph_enabled": self.uses_langgraph,
            "registered_tools": sorted(self._tools.keys()) if self._tools else _TOOL_ORDER,
            "selected_tools": [call.name for call in selected_tools],
            "graph_nodes": ["select_tools", "run_tools", "assemble_results"],
        }

    async def _run_one_tool(self, call: RuntimeToolCall) -> RuntimeToolOutput:
        started_at = time.perf_counter()
        try:
            tool = self._tools.get(call.name)
            if tool is not None:
                raw = await asyncio.wait_for(tool.ainvoke(call.args), timeout=max(call.timeout_seconds, 0.1))
            else:
                raw = await asyncio.wait_for(self._fallback_ainvoke(call), timeout=max(call.timeout_seconds, 0.1))
            prompt = raw.get("prompt") if isinstance(raw, dict) else None
            result = raw.get("result") if isinstance(raw, dict) else None
            return RuntimeToolOutput(
                name=call.name,
                prompt=str(prompt) if prompt else None,
                result=result if isinstance(result, dict) else None,
                error=None,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000),
            )
        except asyncio.TimeoutError:
            return RuntimeToolOutput(
                name=call.name,
                prompt=None,
                result=None,
                error=f"{call.name} tool timed out after {call.timeout_seconds:.1f}s",
                elapsed_ms=round((time.perf_counter() - started_at) * 1000),
            )
        except Exception as exc:
            return RuntimeToolOutput(
                name=call.name,
                prompt=None,
                result=None,
                error=str(exc)[:300],
                elapsed_ms=round((time.perf_counter() - started_at) * 1000),
            )

    async def _fallback_ainvoke(self, call: RuntimeToolCall) -> dict[str, Any]:
        if call.name == "time":
            return await self._time_lookup()
        if call.name == "calculator":
            return await self._calculator_from_text(str(call.args.get("text") or ""))
        if call.name == "unit_converter":
            return await self._unit_from_text(str(call.args.get("text") or ""))
        if call.name == "dictionary":
            return await self._dictionary_lookup(str(call.args.get("word") or ""))
        if call.name == "web_search":
            return await self._web_search_lookup(str(call.args.get("query") or ""))
        if call.name == "weather":
            return await self._weather_lookup(str(call.args.get("city") or ""), str(call.args.get("day") or "today"))
        if call.name == "memory_search":
            return await self._memory_search(**call.args)
        if call.name == "history_search":
            return await self._history_search(**call.args)
        if call.name == "profile_lookup":
            return await self._profile_lookup(**call.args)
        if call.name == "concept_lookup":
            return await self._concept_lookup(**call.args)
        raise ValueError(f"Unknown runtime tool: {call.name}")

    def _build_tools(self) -> dict[str, Any]:
        if not self.uses_langchain or StructuredTool is None:
            return {}
        return {
            "time": StructuredTool.from_function(
                coroutine=self._time_lookup,
                name="time",
                description="Return the current local date and time.",
            ),
            "calculator": StructuredTool.from_function(
                coroutine=self._calculator_from_text,
                name="calculator",
                description="Evaluate a simple numeric expression contained in user text.",
            ),
            "unit_converter": StructuredTool.from_function(
                coroutine=self._unit_from_text,
                name="unit_converter",
                description="Convert explicit unit expressions such as 3 kilometers to meters.",
            ),
            "dictionary": StructuredTool.from_function(
                coroutine=self._dictionary_lookup,
                name="dictionary",
                description="Look up an English word definition.",
            ),
            **(
                {
                    "web_search": StructuredTool.from_function(
                        coroutine=self._web_search_lookup,
                        name="web_search",
                        description="Search the live web through Bocha Web Search.",
                    )
                }
                if self.web_search_tool is not None and self.format_web_search is not None
                else {}
            ),
            "weather": StructuredTool.from_function(
                coroutine=self._weather_lookup,
                name="weather",
                description="Look up weather for an explicit city or region. A location is required.",
            ),
            **(
                {
                    "memory_search": StructuredTool.from_function(
                        coroutine=self._memory_search,
                        name="memory_search",
                        description="Search relevant long-term student memories when the current question needs personalization or prior facts.",
                    ),
                    "history_search": StructuredTool.from_function(
                        coroutine=self._history_search,
                        name="history_search",
                        description="Search older conversation history across sessions for a prior discussion or answer.",
                    ),
                    "profile_lookup": StructuredTool.from_function(
                        coroutine=self._profile_lookup,
                        name="profile_lookup",
                        description="Retrieve the full student profile when the question asks about preferences, identity, grade, or learning background.",
                    ),
                    "concept_lookup": StructuredTool.from_function(
                        coroutine=self._concept_lookup,
                        name="concept_lookup",
                        description="Retrieve tracked learning concepts, mastery, weaknesses, and effective teaching strategy.",
                    ),
                }
                if self.memory_store is not None
                else {}
            ),
        }

    async def _time_lookup(self) -> dict[str, Any]:
        result = self.time_tool.lookup()
        return {"prompt": self.format_time(result), "result": result.to_dict()}

    async def _calculator_from_text(self, text: str) -> dict[str, Any]:
        result = self.calculator_tool.calculate_from_text(text)
        if result is None:
            return {"prompt": None, "result": None}
        return {"prompt": self.format_calculator(result), "result": result.to_dict()}

    async def _unit_from_text(self, text: str) -> dict[str, Any]:
        result = self.unit_converter_tool.convert_from_text(text)
        if result is None:
            return {"prompt": None, "result": None}
        return {"prompt": self.format_unit(result), "result": result.to_dict()}

    async def _dictionary_lookup(self, word: str) -> dict[str, Any]:
        result = await self.dictionary_tool.lookup(word)
        return {"prompt": self.format_dictionary(result), "result": result.to_dict()}

    async def _web_search_lookup(self, query: str) -> dict[str, Any]:
        if self.web_search_tool is None or self.format_web_search is None:
            return {"prompt": None, "result": None}
        result = await self.web_search_tool.lookup(query=query)
        return {"prompt": self.format_web_search(result), "result": result.to_dict()}

    async def _weather_lookup(self, city: str, day: str = "today") -> dict[str, Any]:
        result = await self.weather_tool.lookup(city=city, day=day)
        return {"prompt": self.format_weather(result), "result": result.to_dict()}

    async def _memory_search(self, query: str = "", user_id: str = "", session_id: str = "") -> dict[str, Any]:
        if self.memory_store is None or not user_id or not query.strip():
            return {"prompt": "长期记忆查询无结果。", "result": {"memories": [], "query": query}}
        hits = await self.memory_store.recall_layered(user_id=user_id, query=query, top_k=8)
        items = [_memory_hit_to_dict(item) for item in hits]
        prompt = _format_memory_tool_prompt("相关长期记忆", items, query=query)
        return {"prompt": prompt, "result": {"query": query, "memories": items}}

    async def _history_search(self, query: str = "", user_id: str = "", session_id: str = "") -> dict[str, Any]:
        if self.memory_store is None or not user_id:
            return {"prompt": "历史对话查询无结果。", "result": {"history": [], "query": query}}
        items = await self.memory_store.search_history(user_id=user_id, session_id=session_id, query=query, limit=8)
        prompt = _format_memory_tool_prompt("历史对话", items, query=query)
        return {"prompt": prompt, "result": {"query": query, "history": items}}

    async def _profile_lookup(self, user_id: str = "", session_id: str = "") -> dict[str, Any]:
        if self.memory_store is None or not user_id:
            return {"prompt": "用户画像无结果。", "result": {"profile": {}}}
        profile = await self.memory_store.get_profile(user_id)
        return {"prompt": _format_memory_tool_prompt("完整用户画像", [profile]), "result": {"profile": profile}}

    async def _concept_lookup(self, query: str = "", user_id: str = "", session_id: str = "") -> dict[str, Any]:
        if self.memory_store is None or not user_id:
            return {"prompt": "知识点记录无结果。", "result": {"concepts": [], "query": query}}
        concepts = await self.memory_store.list_concepts(user_id=user_id, query=query, limit=8)
        items = [_learning_concept_to_dict(item) for item in concepts]
        return {"prompt": _format_memory_tool_prompt("知识点掌握状态", items, query=query), "result": {"query": query, "concepts": items}}

    def _has_unit_conversion(self, query: str) -> bool:
        try:
            return self.unit_converter_tool.convert_from_text(query) is not None
        except Exception:
            return False

    def _timeout_for(self, prefix: str, default: float) -> float:
        return _read_float_env(f"LLM_{prefix}_TOOL_TIMEOUT_SECONDS", _read_float_env(f"{prefix}_TOOL_TIMEOUT_SECONDS", default))


_TOOL_ORDER = [
    "time",
    "calculator",
    "unit_converter",
    "dictionary",
    "web_search",
    "weather",
    "memory_search",
    "history_search",
    "profile_lookup",
    "concept_lookup",
]
_TIMEOUT_PREFIX_BY_TOOL = {
    "time": "TIME",
    "calculator": "CALCULATOR",
    "unit_converter": "UNIT",
    "dictionary": "DICTIONARY",
    "web_search": "BOCHA_SEARCH",
    "weather": "WEATHER",
    "memory_search": "MEMORY",
    "history_search": "MEMORY",
    "profile_lookup": "MEMORY",
    "concept_lookup": "MEMORY",
}


def _openai_tool_definitions() -> list[dict[str, Any]]:
    def function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    query_property = {"type": "string", "description": "The user's current query."}
    return [
        function("time", "Get the current local date and time. Must be used for current date or time questions.", {}, []),
        function(
            "calculator",
            "算术计算工具。用户询问明确计算时必须调用，例如：13乘以27、144的平方根、带括号四则运算；不要心算代替。",
            {"text": {"type": "string", "description": "The expression or original arithmetic question."}},
            ["text"],
        ),
        function(
            "unit_converter",
            "单位换算工具。用户询问明确的单位换算时必须调用，例如：3公里等于多少米、500克换算成千克。",
            {"text": {"type": "string", "description": "The complete unit-conversion request."}},
            ["text"],
        ),
        function(
            "dictionary",
            "Look up the definition and usage of an English word.",
            {"word": {"type": "string", "description": "A single English word."}},
            ["word"],
        ),
        function(
            "web_search",
            "Search the live web. Must be used for current, latest, recent, or year-specific sports results, awards, news, policies, and other time-sensitive facts.",
            {"query": query_property},
            ["query"],
        ),
        function(
            "weather",
            "Get weather for an explicit city or region. Ask the user for a location when none is provided.",
            {
                "city": {"type": "string", "description": "Explicit city or region name."},
                "day": {
                    "type": "string",
                    "enum": ["today", "tomorrow", "day_after_tomorrow"],
                    "description": "Forecast day.",
                },
            },
            ["city", "day"],
        ),
        function(
            "memory_search",
            "Search relevant long-term student memories when prior facts or personalization are needed.",
            {"query": query_property},
            ["query"],
        ),
        function(
            "history_search",
            "Search older conversation history when the user refers to previous discussions or activities.",
            {"query": query_property},
            ["query"],
        ),
        function("profile_lookup", "Load the student's full profile when current profile context is insufficient.", {}, []),
        function(
            "concept_lookup",
            "Look up recorded mastery and weakness information for learning concepts.",
            {"query": query_property},
            ["query"],
        ),
    ]


def _dedupe_tool_calls(calls: list[RuntimeToolCall]) -> list[RuntimeToolCall]:
    deduped: list[RuntimeToolCall] = []
    seen: set[str] = set()
    for call in calls:
        if call.name in seen:
            continue
        seen.add(call.name)
        deduped.append(call)
    return deduped


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _memory_hit_to_dict(item: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(item, "id", "")),
        "memory_type": str(getattr(item, "memory_type", "")),
        "content": str(getattr(item, "content", "")),
        "summary": getattr(item, "summary", None),
        "importance": int(getattr(item, "importance", 0) or 0),
        "score": float(getattr(item, "score", 0.0) or 0.0),
        "confidence": float(getattr(item, "confidence", 0.0) or 0.0),
        "tags": list(getattr(item, "tags", []) or []),
    }


def _learning_concept_to_dict(item: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(item, "id", "")),
        "subject": str(getattr(item, "subject", "")),
        "concept": str(getattr(item, "concept", "")),
        "mastery_score": float(getattr(item, "mastery_score", 0.0) or 0.0),
        "confidence": float(getattr(item, "confidence", 0.0) or 0.0),
        "exposure_count": int(getattr(item, "exposure_count", 0) or 0),
        "weakness_count": int(getattr(item, "weakness_count", 0) or 0),
        "correct_count": int(getattr(item, "correct_count", 0) or 0),
        "incorrect_count": int(getattr(item, "incorrect_count", 0) or 0),
        "effective_strategy": getattr(item, "effective_strategy", None),
    }


def _format_memory_tool_prompt(title: str, items: list[Any], *, query: str = "") -> str:
    if not items:
        return f"记忆工具结果：{title}\n查询：{query or '无'}\n结果：无"
    rendered = "\n".join(
        f"- {item if isinstance(item, str) else item.get('content') or item.get('concept') or item}"
        for item in items[:8]
    )
    return f"记忆工具结果：{title}\n查询：{query or '无'}\n{rendered}"


def _looks_like_long_term_memory_query(query: str) -> bool:
    markers = ("记得", "记住", "以前", "过去", "长期", "上次", "之前", "我的情况", "我喜欢", "我不喜欢")
    return any(marker in query for marker in markers)


def _looks_like_history_memory_query(query: str) -> bool:
    markers = ("上次聊", "之前聊", "以前问", "历史对话", "刚才以外", "哪次", "曾经")
    return any(marker in query for marker in markers)


def _looks_like_profile_memory_query(query: str) -> bool:
    markers = ("我叫什么", "我的名字", "我的年级", "我的偏好", "我喜欢什么", "我的画像", "你了解我")
    return any(marker in query for marker in markers)


def _looks_like_concept_memory_query(query: str) -> bool:
    markers = ("薄弱", "掌握", "学到哪", "知识点", "学习情况", "不会什么", "错题", "学习记录")
    return any(marker in query for marker in markers)
