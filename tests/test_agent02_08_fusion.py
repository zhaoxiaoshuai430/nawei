"""Regression coverage for target conversation policies and local knowledge."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from unittest.mock import AsyncMock

import pytest

from app_protocols import EventType, LLMRequest, LLMResponse, Message, MessageRole, StreamEvent
from services.knowledge.schemas import KnowledgeChunkHit, KnowledgeContext, KnowledgeGraphRelation
from services.llm import main as llm_main
from services.llm.agent_runtime import LLMAgentRuntime
from services.llm.agents.tutor_graph import is_contextual_followup
from services.llm.conversation import (
    FollowupStreamTailBuffer, MEMORY_TOOL_NAMES, normalize_followup_answer,
    _should_skip_memory_prepare, _should_use_lightweight_memory,
)
from services.llm.memory import MemoryStore


class KnowledgeFixture:
    def __init__(self, delay: float = 0) -> None:
        self.calls = 0
        self.delay = delay

    async def retrieve(self, *, knowledge_base_id: str, query: str, **kwargs: object) -> KnowledgeContext:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return KnowledgeContext(
            knowledge_base_id=knowledge_base_id, query=query,
            chunks=[KnowledgeChunkHit(chunk_id="light-1", document_id="book", title="光与健康",
                                      text="光照影响昼夜节律。", source_name="现有知识库", score=0.9)],
            relations=[KnowledgeGraphRelation(source_id="light", source_name="光照", relation_type="AFFECTS",
                                               target_id="rhythm", target_name="昼夜节律")],
        )


def request(text: str = "为什么光照影响昼夜节律", **options: object) -> LLMRequest:
    return LLMRequest(request_id="fusion", trace_id="fusion", session_id="session", user_id="owner",
                      messages=[Message(role=MessageRole.USER, content=text)], generation_config=options)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {"LLM_PROVIDER": "deepseek", "LLM_TOOL_SELECTION_MODE": "native",
                        "LLM_AGENT_ENABLED": "true", "MEMORY_TOOLS_ENABLED": "true",
                        "LLM_FAST_MEMORY_MODE": "auto", "SAFETY_ENABLED": "false"}.items():
        monkeypatch.setenv(name, value)


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "temperature"), [("fast", 0.25), ("standard", 0.32), ("deep", 0.35)])
async def test_answer_modes_preserve_knowledge_and_remove_agent_token_limit(mode: str, temperature: float) -> None:
    knowledge = KnowledgeFixture()
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False), knowledge_client=knowledge)
    turn = await runtime.prepare(request(answer_mode=mode, max_tokens=10, memory_mode="off"))
    assert turn.request.generation_config["temperature"] == temperature
    assert "max_tokens" not in turn.request.generation_config
    assert turn.metadata["knowledge"]["chunk_count"] == 1
    assert turn.metadata["knowledge"]["relation_count"] == 1
    assert "参考来源" in turn.request.messages[0].content
    assert "光照 --AFFECTS--> 昼夜节律" in turn.request.messages[0].content
    assert "更高优先级的年龄长度限制内" in turn.request.messages[0].content
    assert knowledge.calls == 1
    assert turn.metadata["agent"]["context_mode"] == "off"
    assert "memory" not in turn.metadata["skills"]["active"]
    assert not MEMORY_TOOL_NAMES.intersection(item["function"]["name"] for item in turn.request.tools)


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "请把第二步再讲清楚。", "请解释一下第2步", "第三点是什么意思？", "第二步为什么这样做", "第十步",
])
async def test_numbered_step_followup_keeps_previous_answer_in_fast_mode(query: str) -> None:
    store = MemoryStore(enabled=False)
    previous = "第一步检查符号。第二步估算数量级，判断答案的大致范围。"
    store.cache_message(user_id="owner", session_id="session", role="user", content="如何检查计算结果")
    store.cache_message(user_id="owner", session_id="session", role="assistant", content=previous)
    knowledge = KnowledgeFixture()
    runtime = LLMAgentRuntime(memory_store=store, knowledge_client=knowledge)

    turn = await runtime.prepare(request(query, answer_mode="fast"))

    assert turn.metadata["memory"]["history_count"] == 2
    assert turn.metadata["skills"]["tutor_strategy"]["intent"]["category"] == "followup"
    assert previous in turn.request.messages[0].content
    assert knowledge.calls == 0


@pytest.mark.parametrize("query", ["请用五个短句说明如何检查计算结果", "一加一等于几", "解释一下二次函数", "牛顿第二定律是什么"])
def test_standalone_questions_are_not_numbered_step_followups(query: str) -> None:
    assert not is_contextual_followup(query)


@pytest.mark.asyncio
async def test_step_followup_respects_explicit_memory_off() -> None:
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False), knowledge_client=KnowledgeFixture())
    turn = await runtime.prepare(request("请把第二步再讲清楚", answer_mode="fast", memory_mode="off"))
    assert turn.metadata["memory"]["mode"] == "off"
    assert turn.metadata["memory"]["history_count"] == 0


@pytest.mark.asyncio
async def test_generation_override_and_raw_inference_bypass() -> None:
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False), knowledge_client=KnowledgeFixture())
    turn = await runtime.prepare(request("你好", answer_mode="fast", temperature=0.7))
    assert turn.request.generation_config["temperature"] == 0.7
    raw = request(agent_enabled=False, max_tokens=42)
    turn = await runtime.prepare(raw)
    assert turn.request is raw
    assert turn.context is None
    assert turn.request.generation_config["max_tokens"] == 42


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_enabled", [True, False])
async def test_memory_timeout_never_cancels_knowledge(graph_enabled: bool) -> None:
    knowledge = KnowledgeFixture(delay=0.04)
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False), knowledge_client=knowledge)
    if not graph_enabled:
        runtime.skill_manager.tutor_graph = None

    async def slow_memory(**kwargs: object) -> None:
        await asyncio.sleep(2)

    runtime.skill_manager.memory_skill.prepare = slow_memory
    context = await runtime.skill_manager.prepare_conversation_context(
        user_id="owner", session_id="session", query="为什么光照影响昼夜节律",
        knowledge_base_id="default", memory_mode="full", memory_timeout=0.01,
    )
    assert context.skill_metadata["memory_preparation"]["timed_out"]
    assert len(context.knowledge.chunks) == 1
    assert len(context.knowledge.relations) == 1
    assert knowledge.calls == 1


def test_adaptive_memory_and_explicit_full() -> None:
    assert _should_skip_memory_prepare(user_text="什么是光合作用", answer_mode="fast", metadata={})
    assert not _should_skip_memory_prepare(user_text="继续", answer_mode="fast", metadata={})
    assert _should_use_lightweight_memory(user_text="继续", answer_mode="fast", metadata={})
    assert not _should_use_lightweight_memory(user_text="我叫什么", answer_mode="fast", metadata={"memory_mode": "full"})


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_mode", ["off", "langgraph"])
async def test_memory_tool_identity_is_bound_outside_model(monkeypatch: pytest.MonkeyPatch, agent_mode: str) -> None:
    monkeypatch.setenv("LLM_TOOL_AGENT_MODE", agent_mode)
    store = MemoryStore(enabled=False)
    store.get_profile = AsyncMock(return_value={"name": "owner-name"})
    runtime = LLMAgentRuntime(memory_store=store, knowledge_client=KnowledgeFixture())
    result = await runtime.tool_agent.collect(query="我叫什么", user_id="owner", session_id="session",
        default_timeout_seconds=1, planned_calls=[{"name": "profile_lookup", "args": {"user_id": "victim", "session_id": "victim-session"}}])
    store.get_profile.assert_awaited_once_with("owner")
    assert result.outputs[0].result["profile"]["name"] == "owner-name"
    assert "user_id" not in next(item for item in runtime.tool_agent.openai_tool_definitions()
                                 if item["function"]["name"] == "profile_lookup")["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_invalid_native_call_does_not_shift_subsequent_results() -> None:
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False))
    active = request("测试").model_copy(update={"tools": runtime.tool_agent.openai_tool_definitions()})
    followup, metadata = await runtime.execute_native_tool_calls(active, first_text="", calls=[
        {"id": "invalid", "name": "dictionary", "args": {"word": ""}},
        {"id": "valid", "name": "calculator", "args": {"text": "13*27"}},
    ])
    messages = [item for item in followup.messages if item.role == MessageRole.TOOL]
    assert json.loads(messages[0].content)["error"]
    assert json.loads(messages[1].content)["result"]["result_text"] == "351"
    assert metadata["tool_results"]["calculator"]["result_text"] == "351"


@pytest.mark.asyncio
async def test_memory_off_rejects_forced_native_memory_call() -> None:
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False))
    runtime.memory_store.get_profile = AsyncMock()
    active = request("你好", memory_mode="off").model_copy(update={"tools": runtime.tool_agent.openai_tool_definitions()})
    _, metadata = await runtime.execute_native_tool_calls(active, first_text="", calls=[
        {"id": "forged", "name": "profile_lookup", "args": {"user_id": "victim"}},
    ])
    runtime.memory_store.get_profile.assert_not_awaited()
    assert "profile_lookup" in metadata["tool_errors"]


@pytest.mark.asyncio
async def test_history_search_scopes_user_and_excludes_current_session() -> None:
    store = MemoryStore(enabled=False)
    for user, session in [("owner", "old"), ("victim", "old"), ("owner", "current"), ("owner:other", "old")]:
        store.cache_message(user_id=user, session_id=session, role="user", content=f"{user} {session} 光照")
    rows = await store.search_history(user_id="owner", session_id="current", query="光照")
    assert [item["content"] for item in rows] == ["owner old 光照"]


@pytest.mark.asyncio
async def test_weather_clarification_and_session_isolation() -> None:
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False), knowledge_client=KnowledgeFixture())
    turn = await runtime.prepare(request("明天会下雨吗"))
    assert "城市或地区" in turn.direct_answer
    assert runtime._weather_tool_query(user_id="victim", session_id="session", user_text="上海") == "上海"
    assert runtime._weather_tool_query(user_id="owner", session_id="other", user_text="上海") == "上海"
    assert runtime._weather_tool_query(user_id="owner", session_id="session", user_text="上海") == "上海明天天气"
    runtime._weather_tool_query(user_id="owner", session_id="session", user_text="后天天气")
    runtime._weather_tool_query(user_id="owner", session_id="session", user_text="算了")
    assert runtime._weather_tool_query(user_id="owner", session_id="session", user_text="上海") == "上海"
    with pytest.raises(ValueError):
        await runtime.tool_agent.weather_tool.lookup(city="")


@pytest.mark.asyncio
async def test_weather_pending_location_is_shared_across_workers() -> None:
    values = {}

    class RedisFixture:
        async def get(self, key: str) -> str | None:
            return values.get(key)

        async def setex(self, key: str, ttl: int, value: str) -> None:
            assert ttl == 300
            values[key] = value

        async def delete(self, key: str) -> None:
            values.pop(key, None)

    first_store, second_store = MemoryStore(enabled=False), MemoryStore(enabled=False)
    first_store._redis = RedisFixture()
    second_store._redis = RedisFixture()
    first, second = LLMAgentRuntime(memory_store=first_store), LLMAgentRuntime(memory_store=second_store)
    await first._resolve_weather_query(user_id="owner", session_id="weather", user_text="明天天气如何")
    assert await second._resolve_weather_query(user_id="other", session_id="weather", user_text="上海") == "上海"
    assert await second._resolve_weather_query(user_id="owner", session_id="weather", user_text="上海") == "上海明天天气"
    assert not values


@pytest.mark.asyncio
async def test_database_history_query_binds_trusted_owner() -> None:
    store = MemoryStore(enabled=False)
    store.initialize = AsyncMock()
    connection = AsyncMock()
    connection.fetch.return_value = [{"session_id": "old", "role": "user", "content": "光照", "created_at": "now"}]

    class Acquire:
        async def __aenter__(self):
            return connection

        async def __aexit__(self, *args: object):
            return None

    class Pool:
        def acquire(self):
            return Acquire()

    store._pool = Pool()
    rows = await store.search_history(user_id="owner", session_id="current", query="光照", limit=8)
    arguments = connection.fetch.call_args.args
    assert "user_id = $1" in arguments[0]
    assert arguments[1:] == ("owner", "光照", "current", 8)
    assert rows[0]["content"] == "光照"


@pytest.mark.asyncio
async def test_rules_direct_answer_and_temporal_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_TOOL_SELECTION_MODE", "rules")
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False), knowledge_client=KnowledgeFixture())
    turn = await runtime.prepare(request("13乘以27等于多少"))
    assert "351" in turn.direct_answer
    runtime.memory_store.list_user_questions_on_date = AsyncMock(return_value=[{"content": "光照是什么"}])
    turn = await runtime.prepare(request("我昨天问过哪些问题"))
    assert "光照是什么" in turn.direct_answer
    assert runtime.memory_store.list_user_questions_on_date.call_args.kwargs["user_id"] == "owner"


@pytest.mark.asyncio
async def test_planner_failure_falls_back_to_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_TOOL_SELECTION_MODE", "llm")
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False), knowledge_client=KnowledgeFixture())
    runtime._plan_runtime_tools = AsyncMock(side_effect=TimeoutError())
    turn = await runtime.prepare(request("13乘以27等于多少"))
    assert "351" in turn.direct_answer
    assert turn.metadata["tools"]["planner_fallback"] == "rules"


@pytest.mark.parametrize("chunk_size", [1, 2, 8, 31, 1000])
@pytest.mark.parametrize("references", ["", "\n\n参考来源：\n- 资料1：现有知识库《光与健康》；链接：https://example.test/?a=1", "\n**参考来源**\n- 资料1：光与健康"])
def test_stream_ending_matches_complete_and_preserves_references(chunk_size: int, references: str) -> None:
    text = "光照影响昼夜节律。\n\n规律作息有助于稳定节律。\n\n你想了解光照的影响，还是想了解作息建议？" + references
    buffer = FollowupStreamTailBuffer(enabled=True)
    output = []
    for offset in range(0, len(text), chunk_size):
        output.extend(buffer.feed(text[offset:offset + chunk_size]))
    output.extend(buffer.finish())
    assert "".join(output) == normalize_followup_answer(text)
    assert "还是" not in "".join(output)
    assert "".join(output).endswith(references)


def test_followup_stream_emits_body_before_invitation() -> None:
    buffer = FollowupStreamTailBuffer(enabled=True)
    assert buffer.feed("这是可以立即输出的正文。" * 20)


@pytest.mark.asyncio
async def test_background_history_is_immediately_visible_and_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_POST_TURN_ASYNC", "true")
    store = MemoryStore(enabled=False)
    runtime = LLMAgentRuntime(memory_store=store, knowledge_client=KnowledgeFixture())
    persisted = []

    async def persist(**kwargs: object) -> None:
        await asyncio.sleep(0.01)
        persisted.append((kwargs["role"], kwargs["content"]))

    store.persist_message = persist
    runtime.skill_manager.after_turn = AsyncMock(return_value={})
    first = await runtime.prepare(request("你好", answer_mode="fast"))
    second = await runtime.prepare(request("继续", answer_mode="fast"))
    result = await runtime.after_turn(first, "第一轮")
    await runtime.after_turn(second, "第二轮")
    assert result["saved_turn"]
    assert not runtime._persistence_tasks
    assert [item["content"] for item in await store.get_history(user_id="owner", session_id="session")] == ["你好", "第一轮", "继续", "第二轮"]
    await asyncio.gather(*list(runtime._persistence_tasks))
    assert persisted == [("user", "你好"), ("assistant", "第一轮"), ("user", "继续"), ("assistant", "第二轮")]
    runtime.skill_manager.after_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_temporal_memory_reads_actual_local_cache() -> None:
    store = MemoryStore(enabled=False)
    store.cache_message(user_id="owner", session_id="old", role="user", content="光照是什么")
    store.cache_message(user_id="victim", session_id="old", role="user", content="别人的内容")
    runtime = LLMAgentRuntime(memory_store=store, knowledge_client=KnowledgeFixture())
    turn = await runtime.prepare(request("我今天问过哪些问题"))
    assert "光照是什么" in turn.direct_answer
    assert "别人的内容" not in turn.direct_answer


@pytest.mark.asyncio
async def test_completion_stream_done_and_memory_have_identical_final_text(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False), knowledge_client=KnowledgeFixture())
    runtime.after_turn = AsyncMock(return_value={"saved": True})
    monkeypatch.setattr(llm_main, "agent_runtime", runtime)
    raw = "正文。\n\n你想了解光照，还是作息？\n\n参考来源：\n- 资料1：光与健康"
    monkeypatch.setattr(llm_main, "complete_deepseek", AsyncMock(return_value=LLMResponse(request_id="fusion", text=raw)))

    async def backend(active: LLMRequest, settings: object):
        for text in raw:
            yield StreamEvent(event=EventType.MESSAGE_DELTA, request_id=active.request_id,
                              trace_id=active.trace_id, session_id=active.session_id, data={"text": text})
        yield StreamEvent(event=EventType.MESSAGE_DONE, request_id=active.request_id,
                          trace_id=active.trace_id, session_id=active.session_id, data={"text": raw})

    monkeypatch.setattr(llm_main, "_stream_backend_with_fallback", backend)
    complete = await llm_main.complete_llm(request("好的"))
    stream = await llm_main.stream_llm(request("好的"))
    events = []
    async for payload in stream.body_iterator:
        for line in payload.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    deltas = "".join(item["data"]["text"] for item in events if item["event"] == "message.delta")
    done = next(item for item in events if item["event"] == "message.done")
    assert deltas == complete.text == done["data"]["text"] == normalize_followup_answer(raw)
    assert [call.args[1] for call in runtime.after_turn.await_args_list] == [complete.text, complete.text]


@pytest.mark.asyncio
async def test_age_constraints_match_for_completion_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryStore(enabled=False)
    today = date.today()
    await store.update_profile(
        user_id="owner",
        session_id="session",
        reason="test",
        changes={"birth_date": today.replace(year=today.year - 4).isoformat()},
    )
    runtime = LLMAgentRuntime(memory_store=store, knowledge_client=KnowledgeFixture())
    runtime.after_turn = AsyncMock(return_value={"saved": True})
    monkeypatch.setattr(llm_main, "agent_runtime", runtime)
    raw = (
        "天空看起来是蓝色，因为阳光里不同颜色的光会被空气分开。"
        "蓝色的光更容易向四周散开，所以我们的眼睛从各处都能看到它。"
        "太阳快落山时，光走的路更长，天空的颜色还会变化。\n\n"
        "你想看看晚霞为什么是红色吗？"
    )
    complete_backend = AsyncMock(return_value=LLMResponse(request_id="fusion", text=raw))
    monkeypatch.setattr(llm_main, "complete_deepseek", complete_backend)
    streamed_requests: list[LLMRequest] = []

    async def backend(active: LLMRequest, settings: object):
        streamed_requests.append(active)
        for text in raw:
            yield StreamEvent(
                event=EventType.MESSAGE_DELTA,
                request_id=active.request_id,
                trace_id=active.trace_id,
                session_id=active.session_id,
                data={"text": text},
            )
        yield StreamEvent(
            event=EventType.MESSAGE_DONE,
            request_id=active.request_id,
            trace_id=active.trace_id,
            session_id=active.session_id,
            data={"text": raw},
        )

    monkeypatch.setattr(llm_main, "_stream_backend_with_fallback", backend)
    complete = await llm_main.complete_llm(request("为什么天空是蓝色的"))
    stream = await llm_main.stream_llm(request("为什么天空是蓝色的"))
    events = []
    async for payload in stream.body_iterator:
        for line in payload.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    done = next(item for item in events if item["event"] == "message.done")
    streamed_text = done["data"]["text"]
    body = streamed_text.rsplit("\n\n", 1)[0]

    assert complete.text == streamed_text
    assert len(re.findall(r"[\u3400-\u9fff]", body)) <= 80
    assert len([item for item in re.split(r"[。！？!?]+", body) if item.strip()]) <= 2
    assert streamed_text.count("？") <= 1
    assert complete.metadata["age_constraints"]["stage"] == "S0"
    assert done["data"]["runtime"]["age_constraints"]["stage"] == "S0"
    assert complete_backend.await_args.args[0].generation_config["max_tokens"] == 108
    assert streamed_requests[0].generation_config["max_tokens"] == 108
