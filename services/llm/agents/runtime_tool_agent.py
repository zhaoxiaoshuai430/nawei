"""LangChain/LangGraph runtime tool agent owned by the LLM service."""

from __future__ import annotations

import asyncio
import os
import time
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
    END = "__end__"  # type: ignore[misc,assignment]
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
        encyclopedia_tool: Any,
        weather_tool: Any,
        format_time: Formatter,
        format_calculator: Formatter,
        format_unit: Formatter,
        format_dictionary: Formatter,
        format_encyclopedia: Formatter,
        format_weather: Formatter,
        format_web_search: Formatter | None = None,
        web_search_tool: Any | None = None,
    ) -> None:
        self.time_tool = time_tool
        self.calculator_tool = calculator_tool
        self.unit_converter_tool = unit_converter_tool
        self.dictionary_tool = dictionary_tool
        self.encyclopedia_tool = encyclopedia_tool
        self.weather_tool = weather_tool
        self.web_search_tool = web_search_tool
        self.format_time = format_time
        self.format_calculator = format_calculator
        self.format_unit = format_unit
        self.format_dictionary = format_dictionary
        self.format_encyclopedia = format_encyclopedia
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
        """Return the stable names exposed by this runtime."""

        return sorted(self._tools.keys()) if self._tools else list(_TOOL_ORDER)

    def openai_tool_definitions(
        self,
        names: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Return enabled runtime tools in OpenAI-compatible function format."""

        enabled = set(self.registered_tools)
        if names is not None:
            enabled.intersection_update(names)
        return [
            item
            for item in _openai_tool_definitions()
            if str(item.get("function", {}).get("name") or "") in enabled
        ]

    def suggested_tool_names(
        self,
        query: str,
        *,
        default_timeout_seconds: float = 6.0,
    ) -> list[str]:
        """Return deterministic tool candidates without executing them."""

        calls = self._select_tool_calls(
            {
                "query": query,
                "default_timeout_seconds": default_timeout_seconds,
            }
        )
        return [call.name for call in calls]

    async def collect(
        self,
        *,
        query: str,
        default_timeout_seconds: float,
        planned_calls: list[dict[str, Any] | RuntimeToolCall] | None = None,
    ) -> RuntimeToolAgentResult:
        state: RuntimeToolAgentState = {
            "query": query,
            "default_timeout_seconds": default_timeout_seconds,
        }
        if planned_calls is not None:
            selected_tools = self._normalize_planned_tool_calls(
                planned_calls,
                query=query,
                default_timeout=default_timeout_seconds,
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

        if not web_query and _read_bool_env("ENCYCLOPEDIA_TOOL_ENABLED", True):
            encyclopedia_query = self.encyclopedia_tool.extract_query(query)
            if encyclopedia_query:
                calls.append(
                    RuntimeToolCall(
                        "encyclopedia",
                        {"query": encyclopedia_query},
                        self._timeout_for("ENCYCLOPEDIA", default_timeout),
                    )
                )

        if _read_bool_env("WEATHER_TOOL_ENABLED", True):
            weather_query = self.weather_tool.extract_query(query)
            if weather_query:
                city, day = weather_query
                weather_default = max(default_timeout, float(getattr(self.weather_tool, "timeout_seconds", default_timeout)) + 1.0)
                calls.append(RuntimeToolCall("weather", {"city": city, "day": day}, self._timeout_for("WEATHER", weather_default)))

        return _dedupe_tool_calls(calls)

    def _normalize_planned_tool_calls(
        self,
        planned_calls: list[dict[str, Any] | RuntimeToolCall],
        *,
        query: str,
        default_timeout: float,
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
            normalized = self._normalize_tool_args(name, args, query=query)
            if normalized is None:
                continue
            calls.append(RuntimeToolCall(name, normalized, self._timeout_for(_TIMEOUT_PREFIX_BY_TOOL[name], default_timeout)))
        return calls

    def _normalize_tool_args(self, name: str, args: dict[str, Any], *, query: str) -> dict[str, Any] | None:
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
        if name == "encyclopedia":
            tool_query = str(args.get("query") or args.get("topic") or "").strip()
            if not tool_query:
                tool_query = str(self.encyclopedia_tool.extract_query(query) or "").strip()
            return {"query": tool_query} if tool_query else None
        if name == "web_search":
            tool_query = str(args.get("query") or "").strip()
            if not tool_query and self.web_search_tool is not None:
                tool_query = str(self.web_search_tool.extract_query(query) or "").strip()
            return {"query": tool_query} if tool_query else None
        if name == "weather":
            city = str(args.get("city") or args.get("location") or "").strip()
            day = str(args.get("day") or "today").strip() or "today"
            if not city:
                weather_query = self.weather_tool.extract_query(query)
                if weather_query:
                    city, day = weather_query
            return {"city": city, "day": day} if city else None
        return None

    def _tool_enabled(self, name: str) -> bool:
        if name == "time":
            return _read_bool_env("TIME_TOOL_ENABLED", True)
        if name == "calculator":
            return _read_bool_env("CALCULATOR_TOOL_ENABLED", True)
        if name == "unit_converter":
            return _read_bool_env("UNIT_TOOL_ENABLED", True)
        if name == "dictionary":
            return _read_bool_env("DICTIONARY_TOOL_ENABLED", True)
        if name == "encyclopedia":
            return _read_bool_env("ENCYCLOPEDIA_TOOL_ENABLED", True)
        if name == "web_search":
            return self.web_search_tool is not None and _read_bool_env("BOCHA_WEB_SEARCH_ENABLED", True)
        if name == "weather":
            return _read_bool_env("WEATHER_TOOL_ENABLED", True)
        return False

    def _selection_metadata(self, selected_tools: list[RuntimeToolCall], *, selector: str) -> dict[str, Any]:
        return {
            "agent": "LangChainRuntimeToolAgent",
            "agent_mode": "langgraph_tools" if self.uses_langgraph else self.agent_mode,
            "selector": selector,
            "langchain_enabled": self.uses_langchain,
            "langgraph_enabled": self.uses_langgraph,
            "registered_tools": self.registered_tools,
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
        if call.name == "encyclopedia":
            return await self._encyclopedia_lookup(str(call.args.get("query") or ""))
        if call.name == "web_search":
            return await self._web_search_lookup(str(call.args.get("query") or ""))
        if call.name == "weather":
            return await self._weather_lookup(str(call.args.get("city") or ""), str(call.args.get("day") or "today"))
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
            "encyclopedia": StructuredTool.from_function(
                coroutine=self._encyclopedia_lookup,
                name="encyclopedia",
                description="Look up a concise encyclopedia summary.",
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
                description="Look up current or forecast weather for a city.",
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

    async def _encyclopedia_lookup(self, query: str) -> dict[str, Any]:
        result = await self.encyclopedia_tool.lookup(query=query)
        return {"prompt": self.format_encyclopedia(result), "result": result.to_dict()}

    async def _web_search_lookup(self, query: str) -> dict[str, Any]:
        if self.web_search_tool is None or self.format_web_search is None:
            return {"prompt": None, "result": None}
        result = await self.web_search_tool.lookup(query=query)
        return {"prompt": self.format_web_search(result), "result": result.to_dict()}

    async def _weather_lookup(self, city: str, day: str = "today") -> dict[str, Any]:
        result = await self.weather_tool.lookup(city=city, day=day)
        return {"prompt": self.format_weather(result), "result": result.to_dict()}

    def _has_unit_conversion(self, query: str) -> bool:
        try:
            return self.unit_converter_tool.convert_from_text(query) is not None
        except Exception:
            return False

    def _timeout_for(self, prefix: str, default: float) -> float:
        return _read_float_env(f"LLM_{prefix}_TOOL_TIMEOUT_SECONDS", _read_float_env(f"{prefix}_TOOL_TIMEOUT_SECONDS", default))


_TOOL_ORDER = ["time", "calculator", "unit_converter", "dictionary", "web_search", "encyclopedia", "weather"]
_TIMEOUT_PREFIX_BY_TOOL = {
    "time": "TIME",
    "calculator": "CALCULATOR",
    "unit_converter": "UNIT",
    "dictionary": "DICTIONARY",
    "web_search": "BOCHA_SEARCH",
    "encyclopedia": "ENCYCLOPEDIA",
    "weather": "WEATHER",
}


def _openai_tool_definitions() -> list[dict[str, Any]]:
    def function(
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
    ) -> dict[str, Any]:
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
            "encyclopedia",
            "Look up stable encyclopedia facts for a person, place, event, object, or concept.",
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
