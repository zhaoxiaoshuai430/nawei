"""Conversation policies adapted from agent02 08 (9d0996c); knowledge stays local."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any

from services.llm.agents.tutor_graph import is_contextual_followup
from services.llm.memory import MemoryStore, _safe_zoneinfo
from services.llm.tools.calculator import format_expression_for_display


_REFERENCE_HEADING = re.compile(
    r"(?im)^[ \t]*(?:[-*•][ \t]*)?(?:#{1,6}[ \t]*)?(?:\*\*)?"
    r"(?:参考(?:资料|来源|文献)|(?:资料|来源)[ \t]*\d+|sources?|references?)"
    r"(?:\*\*)?[ \t]*(?:[：:]|$)"
)
MEMORY_TOOL_NAMES = {"memory_search", "history_search", "profile_lookup", "concept_lookup"}
_INVITATION_START = re.compile(r"(?:^|\r?\n)(?:你想|想不想|要不要|接下来你|你现在是|你是)")
_ALLOWED_RUNTIME_TOOL_NAMES = MEMORY_TOOL_NAMES | {"time", "calculator", "unit_converter", "dictionary", "weather", "web_search"}


def _current_date_context() -> str:
    now = datetime.now(_safe_zoneinfo("Asia/Shanghai"))
    return f"{now:%Y-%m-%d} {now:%A} Asia/Shanghai"


def _runtime_tool_planner_prompt() -> str:
    return f"""
You are a tool-selection planner for Xiaowei Gateway.
Return only strict JSON. No markdown, no explanation.
Current date: {_current_date_context()}.

Available tools:
- time: current local date/time. args: {{}}
- calculator: numeric calculation. args: {{"text": original user text}}
- unit_converter: unit conversion. args: {{"text": original user text}}
- dictionary: English word definition. args: {{"word": English word}}
- weather: weather lookup. args: {{"city": explicit city or region name, "day": "today" | "tomorrow" | "day_after_tomorrow"}}
- web_search: live/latest web search. args: {{"query": search query}}
- memory_search: relevant long-term student memories. args: {{"query": memory topic}}
- history_search: older conversation history across sessions. args: {{"query": prior topic}}
- profile_lookup: full student profile for identity/preferences/background. args: {{}}
- concept_lookup: tracked learning concepts, mastery, weaknesses, and strategy. args: {{"query": concept or subject}}

Choose tools only when the user explicitly needs fresh facts, current-year facts,
weather, time, calculation, conversion, dictionary, or live web search.
Use web_search for questions about this year, current/recent/latest information,
news, sports tournaments, World Cup/世界杯, scores, winners, schedules, rankings,
teams, players, or any event whose answer may have changed.
For weather, never guess or infer a city. If the user did not name one, choose no
weather tool; Gateway will ask the user to provide a city or region.
When planning web_search for "this year"/今年 or World Cup/世界杯, include the
current year and current date in the search query, plus words such as 结果,
冠军, 赛程, 最新, or status as appropriate.
For a World Cup/世界杯 status question, prefer a query like
"{{current year}} 世界杯 冠军 决赛 结果 最新 {{current date}}" over a vague query like
"2026 世界杯 最新情况".
For tutoring, chatting, or writing, choose no tools unless prior student context is
needed. Use profile_lookup for questions about the student's name, grade, preferences,
or learning background. Use history_search for a specific older conversation or answer.
Use memory_search for relevant long-term facts, preferences, weaknesses, plans, or
learning events. Use concept_lookup for mastery, weak points, learning records, or
how to personalize teaching. For mistake review, use history_search when the user
refers to a prior concrete question or answer, and concept_lookup when recurring
weaknesses or mastery should affect the diagnosis. For personalized practice, use
concept_lookup only when the user asks to target weak points or prior performance.
Do not choose more than two memory tools in one plan.

Output schema:
{{"tools":[{{"name":"weather","args":{{"city":"北京","day":"today"}}}}]}}
If no tool is needed, output {{"tools":[]}}.
""".strip()


def _parse_runtime_tool_plan(text: str) -> list[dict[str, Any]]:
    payload = _parse_json_object_or_array(text)
    if payload is None:
        return []
    if isinstance(payload, dict):
        raw_tools = payload.get("tools")
        if raw_tools is None and ("name" in payload or "tool" in payload):
            raw_tools = [payload]
    elif isinstance(payload, list):
        raw_tools = payload
    else:
        raw_tools = []

    calls: list[dict[str, Any]] = []
    for item in raw_tools if isinstance(raw_tools, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("tool") or "").strip().lower().replace("-", "_")
        if name not in _ALLOWED_RUNTIME_TOOL_NAMES:
            continue
        raw_args = item.get("args") or item.get("arguments") or {}
        args = dict(raw_args) if isinstance(raw_args, dict) else {}
        calls.append({"name": name, "args": args})
        if len(calls) >= _read_int_env("LLM_TOOL_PLANNER_MAX_CALLS", 3):
            break
    return calls


def _parse_json_object_or_array(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    for candidate in (stripped, _first_json_span(stripped, "{", "}"), _first_json_span(stripped, "[", "]")):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _first_json_span(text: str, open_char: str, close_char: str) -> str | None:
    start = text.find(open_char)
    end = text.rfind(close_char)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def normalize_followup_answer(text: str) -> str:
    reference = _REFERENCE_HEADING.search(text)
    body = text[:reference.start()] if reference is not None else text
    stripped = body.rstrip()
    invitation = _INVITATION_START.search(stripped)
    if invitation is None:
        return text
    normalized = stripped[:invitation.start()] + _normalize_followup_ending(stripped[invitation.start():])
    return normalized + body[len(stripped):] + (text[reference.start():] if reference is not None else "")


class FollowupStreamTailBuffer:
    """Stream the body promptly and retain the invitation without altering citations."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.pending = ""
        self.references_started = False
        self.holding_ending = False

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        if not self.enabled or self.references_started:
            return [text]
        self.pending += text
        reference = _REFERENCE_HEADING.search(self.pending)
        if reference is not None:
            output = normalize_followup_answer(self.pending)
            self.pending = ""
            self.references_started = True
            return [output]
        if self.holding_ending:
            return []
        invitation = _INVITATION_START.search(self.pending)
        if invitation is not None:
            split_at = invitation.start()
            output, self.pending = self.pending[:split_at], self.pending[split_at:]
            self.holding_ending = True
            return [output] if output else []
        split_at = max(0, len(self.pending) - 24)
        output, self.pending = self.pending[:split_at], self.pending[split_at:]
        return [output] if output else []

    def finish(self) -> list[str]:
        output = normalize_followup_answer(self.pending) if self.enabled else self.pending
        self.pending = ""
        return [output] if output else []


def direct_tool_answer(tool_results: dict[str, Any], tool_errors: dict[str, Any]) -> str | None:
    if not _read_bool_env("LLM_DIRECT_TOOL_ANSWER_ENABLED", True):
        return None
    formatters = {
        "time": _time_direct_answer, "calculator": _calculator_direct_answer,
        "unit_converter": _unit_direct_answer, "weather": _weather_direct_answer,
        "dictionary": _dictionary_direct_answer,
    }
    if _read_bool_env("LLM_DIRECT_WEB_SEARCH_ANSWER_ENABLED", True):
        formatters["web_search"] = _web_search_direct_answer
    for name, formatter in formatters.items():
        if name in tool_results:
            answer = formatter(tool_results[name])
            if answer:
                return answer
    if "weather" in tool_errors:
        return _weather_error_direct_answer(tool_errors["weather"])
    if "web_search" in tool_errors:
        return _web_search_error_direct_answer(tool_errors["web_search"])
    return None


def _normalize_followup_ending(text: str) -> str:
    """Normalize only the trailing exploration question of a follow-up answer."""

    if not text or not re.search(r"[？?]", text):
        return text
    paragraph_start = max(text.rfind("\n\n"), text.rfind("\r\n\r\n"))
    prefix = text[: paragraph_start + 2] if paragraph_start >= 0 else ""
    ending = text[paragraph_start + 2 :] if paragraph_start >= 0 else text
    forbidden = ("还是", "或者", "或是", "二选一")
    if not any(marker in ending for marker in forbidden):
        return text

    if not re.search(r"[？?][”\"’'）)]*\s*$", ending):
        questions = list(re.finditer(r"[^。！？!?\n]*[？?][”\"’'）)]*", ending))
        if questions:
            last_question = questions[-1]
            before = ending[: last_question.start()].rstrip()
            after = ending[last_question.end() :].strip()
            question = last_question.group(0).strip()
            ending = "\n\n".join(item for item in (before, after, question) if item)

    binary = re.fullmatch(
        r"(?P<lead>.*?)(?:你现在是|你是)(?P<first>.+?)[，,]?\s*还是(?:已经)?(?P<second>.+?)[？?]([”\"’'）)]*)\s*",
        ending,
        flags=re.DOTALL,
    )
    if binary:
        first = binary.group("first").strip(" ，,")
        second = binary.group("second").strip(" ，,")
        suffix = binary.group(4) or ""
        ending = f"{binary.group('lead')}你想继续看看{first}时的具体影响，和{second}时可以怎样应对吗？{suffix}"
        return prefix + ending

    question_marks = [match.start() for match in re.finditer(r"[？?]", ending)]
    if len(question_marks) > 1:
        last_mark = question_marks[-1]
        ending = "".join(
            "，" if character in "？?" and index != last_mark else character
            for index, character in enumerate(ending)
        )
    ending = re.sub(r"\s*(?:还是|或者|或是)\s*(?:你)?(?:想要|想先|想)?", "和", ending)
    ending = ending.replace("二选一", "两个方向")
    ending = re.sub(r"[，,]\s*和\s*", "，和", ending)
    ending = re.sub(r"和和+", "和", ending)
    return prefix + ending


async def _direct_temporal_memory_answer(memory: MemoryStore | None, *, user_id: str, user_text: str) -> str | None:
    if memory is None or not _is_temporal_memory_question(user_text):
        return None
    target_date, label = _target_memory_date(user_text)
    try:
        questions = await asyncio.wait_for(
            memory.list_user_questions_on_date(user_id=user_id, target_date=target_date, timezone="Asia/Shanghai", limit=20),
            timeout=_read_float_env("LLM_TEMPORAL_MEMORY_TIMEOUT_SECONDS", 1.2),
        )
    except asyncio.TimeoutError:
        return f"我尝试查你{label}的问题记录，但这次记忆查询超时了。你可以稍后再问，或确认数据库和 Redis 都已启动。"
    except Exception:
        return f"我尝试查你{label}的问题记录，但这次记忆查询没有成功。"
    if not questions:
        return f"我没有查到你{label}问过的问题。可能是当时用的用户 ID 不一样，或者那轮对话没有写入记忆。"
    lines = []
    for index, item in enumerate(questions[:10], start=1):
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"{index}. {content[:120]}")
    if not lines:
        return f"我没有查到你{label}问过的问题。"
    more = f" 另外还有 {len(questions) - 10} 条。" if len(questions) > 10 else ""
    return f"你{label}问过这些问题：\n" + "\n".join(lines) + more


def _is_temporal_memory_question(text: str) -> bool:
    if not any(marker in text for marker in ("昨天", "前天", "今天")):
        return False
    if not any(marker in text for marker in ("我问", "问了", "问过", "聊了", "聊过", "说了", "说过")):
        return False
    return any(marker in text for marker in ("什么", "哪些", "啥", "内容", "问题"))


def _is_lightweight_memory_query(text: str) -> bool:
    if not any(marker in text for marker in ("记得", "记住", "还记得", "记忆", "我喜欢", "我的喜好", "我叫什么", "我的名字", "我是谁")):
        return False
    return any(marker in text for marker in ("我", "我的", "喜欢", "喜好", "名字", "叫什么", "记得", "记忆"))


def _requested_memory_mode(metadata: dict[str, Any]) -> str:
    raw = str(metadata.get("memory_mode") or metadata.get("memory_policy") or "").strip().lower()
    if raw in {"none", "skip", "disabled", "disable", "false", "0"}:
        return "off"
    if raw in {"auto", "adaptive"}:
        return "auto"
    if raw in {"off", "lightweight", "light", "profile", "full"}:
        return "lightweight" if raw in {"light", "profile"} else raw
    return ""


def _fast_memory_mode() -> str:
    raw = os.getenv("LLM_FAST_MEMORY_MODE", "auto").strip().lower()
    if raw in {"none", "skip", "disabled", "disable", "false", "0"}:
        return "off"
    if raw in {"auto", "adaptive"}:
        return "auto"
    if raw in {"off", "lightweight", "light", "profile", "full"}:
        return "lightweight" if raw in {"light", "profile"} else raw
    return "auto"


def _memory_prepare_timeout_seconds(answer_mode: str, *, user_text: str = "", metadata: dict[str, Any] | None = None) -> float:
    metadata = metadata or {}
    if answer_mode == "fast":
        fast_default = 1.0 if _fast_query_needs_memory(user_text) or _requested_memory_mode(metadata) in {"lightweight", "full"} else 0.45
        minimum = 2.0 if is_contextual_followup(user_text) else 0.1
        return max(_read_float_env("LLM_FAST_MEMORY_TIMEOUT_SECONDS", fast_default), minimum)
    default = 2.5 if answer_mode == "standard" else 5.0
    return max(_read_float_env("LLM_MEMORY_PREPARE_TIMEOUT_SECONDS", default), 0.1)


def _should_skip_memory_prepare(*, user_text: str, answer_mode: str, metadata: dict[str, Any]) -> bool:
    requested = _requested_memory_mode(metadata)
    if requested == "off":
        return True
    if requested == "auto":
        requested = ""
    if requested in {"lightweight", "full"}:
        return False
    if _is_temporal_memory_question(user_text) or _is_lightweight_memory_query(user_text):
        return False
    if answer_mode != "fast":
        return False
    mode = _fast_memory_mode()
    if mode == "off":
        return True
    if mode == "auto":
        return not _fast_query_needs_memory(user_text)
    return False


def _should_use_lightweight_memory(*, user_text: str, answer_mode: str, metadata: dict[str, Any]) -> bool:
    requested = _requested_memory_mode(metadata)
    if requested == "auto":
        requested = ""
    if requested == "full":
        return False
    if requested == "lightweight":
        return True
    if _is_lightweight_memory_query(user_text):
        return True
    if answer_mode != "fast":
        return False
    mode = _fast_memory_mode()
    if mode == "auto":
        return _fast_query_needs_memory(user_text)
    return mode == "lightweight"


def _fast_query_needs_memory(text: str) -> bool:
    if _is_temporal_memory_question(text) or _is_lightweight_memory_query(text) or is_contextual_followup(text):
        return True
    continuity_markers = (
        "上次",
        "刚才",
        "刚刚",
        "之前",
        "前面",
        "继续",
        "接着",
        "上一题",
        "这道题",
        "这个题",
        "刚才那题",
        "前一道",
        "前一个",
    )
    return any(marker in text for marker in continuity_markers)


def _target_memory_date(text: str) -> tuple[Any, str]:
    now = datetime.now(_safe_zoneinfo("Asia/Shanghai")).date()
    if "前天" in text:
        return now - timedelta(days=2), "前天"
    if "今天" in text:
        return now, "今天"
    return now - timedelta(days=1), "昨天"


def _weather_direct_answer(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    city = str(result.get("city") or "该地")
    day = _direct_day_label(str(result.get("day") or "today"))
    if str(result.get("source") or "") == "weather-unavailable":
        return f"我识别到你要查{city}{day}天气，但实时天气源这次不可用，暂时没有可靠的降雨和温度数据。可以稍后再试，或换成更具体的城市名再查。"
    condition = str(result.get("condition") or "天气情况未知")
    temp = _compact_value(result.get("temperature_c"), "°C")
    feels = _compact_value(result.get("apparent_temperature_c"), "°C")
    temp_min = _compact_value(result.get("temp_min_c"), "°C")
    temp_max = _compact_value(result.get("temp_max_c"), "°C")
    rain_prob = _compact_value(result.get("precipitation_probability_max_percent"), "%")
    base = f"{city}{day}天气是{condition}"
    if temp != "-":
        base += f"，当前约{temp}"
        if feels != "-":
            base += f"，体感约{feels}"
    if temp_min != "-" or temp_max != "-":
        base += f"，气温大约{temp_min}到{temp_max}"
    if rain_prob != "-":
        base += f"，最大降水概率约{rain_prob}"
    umbrella = ""
    try:
        probability = float(result.get("precipitation_probability_max_percent") or 0)
        precipitation = float(result.get("precipitation_mm") or 0)
        has_rain_condition = any(marker in condition for marker in ("雨", "雪", "冰雹", "雷暴", "降水"))
        umbrella = "建议带伞。" if probability >= 50 or precipitation > 0 or has_rain_condition else "一般不需要带伞。"
    except (TypeError, ValueError):
        umbrella = ""
    return f"{base}。{umbrella}".strip()


def _weather_error_direct_answer(error: Any) -> str:
    detail = str(error or "上游没有返回可用天气数据").strip()
    return f"我已尝试查询实时天气，但这次查询没有成功：{detail[:120]}。可以稍后再试，或换一个城市再查。"


def _time_direct_answer(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    date = str(result.get("date") or "")
    weekday = str(result.get("weekday") or "")
    current_time = str(result.get("time") or "")
    if not date and not current_time:
        return None
    parts = [part for part in (date, weekday, current_time) if part]
    return "现在是" + " ".join(parts) + "。"


def _calculator_direct_answer(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    expression = str(result.get("expression") or "").strip()
    result_text = str(result.get("result_text") or result.get("result") or "").strip()
    if not result_text:
        return None
    if not expression:
        return f"计算结果：{result_text}。"
    return f"算式：{format_expression_for_display(expression)}\n结果：{result_text}。"


def _unit_direct_answer(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    value = _compact_value(result.get("value"))
    from_unit = str(result.get("from_unit") or "")
    to_unit = str(result.get("to_unit") or "")
    result_text = str(result.get("result_text") or result.get("result") or "").strip()
    if not result_text or not from_unit or not to_unit:
        return None
    return f"{value} {from_unit} = {result_text} {to_unit}。"


def _dictionary_direct_answer(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    word = str(result.get("word") or "这个单词")
    phonetic = str(result.get("phonetic") or "")
    definitions = result.get("definitions")
    if not isinstance(definitions, list):
        definitions = []
    lines = []
    for item in definitions[:2]:
        if isinstance(item, dict) and item.get("definition"):
            part = str(item.get("part_of_speech") or "")
            prefix = f"{part}，" if part else ""
            lines.append(prefix + str(item["definition"]))
    if not lines:
        return None
    phonetic_text = f"，音标 {phonetic}" if phonetic else ""
    return f"{word}{phonetic_text}。主要意思是：{'；'.join(lines)}。"


def _web_search_direct_answer(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    query = str(result.get("query") or "查询").strip()
    results = result.get("results")
    if not isinstance(results, list):
        results = []
    lines = []
    for item in results[:2]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "搜索结果").strip()
        summary = _clean_tool_extract(str(item.get("summary") or item.get("snippet") or ""))
        url = str(item.get("url") or "").strip()
        if summary:
            lines.append(f"{title}：{summary[:140].rstrip('，,；;。 ')}。")
        elif url:
            lines.append(f"{title}：{url}")
    if not lines:
        return None
    return f"根据 Bocha 联网搜索“{query}”，" + " ".join(lines)


def _web_search_error_direct_answer(error: Any) -> str:
    detail = str(error or "上游没有返回可用结果").strip()
    return f"我已尝试 Bocha 联网搜索，但这次查询没有成功：{detail[:120]}。可以稍后再试，或换一个更具体的关键词。"


def _clean_tool_extract(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"（[^）]{0,80}）", "", cleaned)
    return cleaned


def _compact_value(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
        text = str(int(number)) if number.is_integer() else f"{number:.1f}".rstrip("0").rstrip(".")
        return f"{text}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def _direct_day_label(day: str) -> str:
    return {
        "today": "今天",
        "tomorrow": "明天",
        "day_after_tomorrow": "后天",
    }.get(day, day or "今天")


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _normalize_answer_mode(value: object) -> str:
    mode = str(value or os.getenv("LLM_DEFAULT_ANSWER_MODE", "standard")).strip().lower()
    if mode in {"fast", "quick", "speed"}:
        return "fast"
    if mode in {"normal", "standard"}:
        return "standard"
    if mode in {"deep", "detailed"}:
        return "deep"
    return "standard"


def _answer_mode_generation_config(mode: str, metadata: dict[str, Any]) -> dict[str, Any]:
    if mode == "deep":
        defaults = {"temperature": 0.35}
    elif mode == "standard":
        defaults = {"temperature": 0.32}
    else:
        defaults = {"temperature": 0.25}
    return {key: metadata.get(key, value) for key, value in defaults.items()}


def _apply_answer_mode_prompt(system_prompt: str, mode: str) -> str:
    if mode == "deep":
        instruction = "深度模式：在更高优先级的年龄长度限制内充分展开，先给结论，再解释原因、例子和可操作建议；学习问题讲清思路、完整步骤、易错点和练习方向。"
    elif mode == "standard":
        instruction = "标准模式：在更高优先级的年龄长度限制内完整回答当前问题，讲清关键原因、必要步骤和例子；只有用户本轮明确要求简短时才进一步压缩。"
    else:
        instruction = "快速模式：在更高优先级的年龄长度限制内直接回答核心问题，保留必要内容；用户明确要求极简时才进一步压缩。"
    return f"{instruction}\n{system_prompt}"


def _native_tool_policy_prompt() -> str:
    return (
        "工具调用规则（事实可靠性优先级高于凭记忆直接回答）：\n"
        "- 询问当前、最新、今年、近年或明确年份的赛事结果、冠军、奖项得主、新闻、政策等时效性事实时，"
        "必须先调用 web_search，即使你认为自己知道答案；不得编造检索结果或来源。\n"
        "- 明确的算术计算必须调用 calculator；单位换算必须调用 unit_converter；当前日期或时间必须调用 time。\n"
        "- 查询天气必须调用 weather；用户未提供城市或地区时先询问位置，不得猜测。\n"
        "- 查询英文单词释义和用法时调用 dictionary。\n"
        "- 需要更早的用户上下文时，按需调用 history_search、memory_search、profile_lookup 或 concept_lookup。\n"
        "- 工具不可用或返回不足时明确说明，不得用猜测补齐；确实不需要工具的问题可以直接回答。"
    )
