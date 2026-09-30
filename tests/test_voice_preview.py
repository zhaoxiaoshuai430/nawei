"""Speculation must not speak, persist provisional turns, or reuse revised input."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app_protocols import EventType, LLMRequest, LLMResponse, Message, MessageRole
from services.gateway import main as gateway
from services.gateway.voice_preview import VoicePreview
from services.llm import main as llm
from services.llm.agent_runtime import AgentTurn, LLMAgentRuntime
from services.llm.voice_preview import VoicePreviewCache


def llm_request(text="Explain how rain forms", **updates):
    return LLMRequest(
        request_id="request", trace_id="trace", session_id="session", user_id="owner",
        messages=[Message(role=MessageRole.USER, content=text)], **updates,
    )


def voice_request(**updates):
    return gateway.RealtimeVoiceRequest(
        request_id="request", trace_id="trace", session_id="session", user_id="owner",
        output_audio=False, **updates,
    )


@pytest.fixture
def preview_env(monkeypatch):
    monkeypatch.setenv("GATEWAY_VOICE_PREVIEW_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_VOICE_PREVIEW_STABLE_SECONDS", "0")
    monkeypatch.setenv("GATEWAY_KEYWORD_FILTER_ENABLED", "false")


@pytest.fixture
def local_llm(monkeypatch):
    prepared, persisted = [], []
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setattr(llm, "voice_previews", VoicePreviewCache())
    monkeypatch.setattr(llm, "_resolve_request_safety_age", AsyncMock(return_value=None))
    monkeypatch.setattr(llm.agent_runtime, "_resolve_weather_query", AsyncMock())

    async def prepare(request):
        prepared.append(request)
        return AgentTurn(
            request=request, original_request=request, user_id=request.user_id,
            user_text=request.messages[-1].content, direct_answer="Water vapor cools and condenses.",
        )

    async def persist(turn, text):
        assert not turn.original_request.defer_persistence
        persisted.append((turn.user_text, text))
        return {"saved_turn": True}

    monkeypatch.setattr(llm.agent_runtime, "prepare", prepare)
    monkeypatch.setattr(llm.agent_runtime, "after_turn", persist)
    return prepared, persisted


async def read_stream(response):
    events = []
    async for chunk in response.body_iterator:
        text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
        for line in text.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


async def test_preview_generates_without_memory_then_reuses_once(local_llm):
    prepared, persisted = local_llm
    preview = await llm.complete_llm(llm_request(defer_persistence=True))
    token = preview.metadata["voice_preview_token"]
    assert len(prepared) == 1
    assert persisted == []
    assert preview.metadata["persistence"]["saved_turn"] is False
    request = llm_request(voice_preview_token=token)
    events = await read_stream(await llm.stream_llm(request))
    assert len(prepared) == 1
    assert len(persisted) == 1
    assert events[-1]["data"]["runtime"]["voice_preview_used"] is True
    assert "Water vapor" in "".join(event["data"].get("text", "") for event in events[:-1])
    await llm.complete_llm(request)
    assert len(prepared) == 2, "A receipt must not replay an old turn twice"


@pytest.mark.parametrize("updates", [
    {"user_id": "other"}, {"session_id": "other"},
    {"metadata": {"device_role": "other"}},
    {"messages": [Message(role=MessageRole.USER, content="Explain snow instead")]},
])
async def test_preview_cannot_cross_identity_or_question(local_llm, updates):
    prepared, _ = local_llm
    preview = await llm.complete_llm(llm_request(defer_persistence=True))
    request = llm_request(voice_preview_token=preview.metadata["voice_preview_token"]).model_copy(update=updates)
    await llm.complete_llm(request)
    assert len(prepared) == 2


async def test_final_safety_check_can_reject_a_preview(local_llm, monkeypatch):
    from services.llm.safety import SafetyDecision

    _, persisted = local_llm
    preview = await llm.complete_llm(llm_request(defer_persistence=True))
    request = llm_request(voice_preview_token=preview.metadata["voice_preview_token"])
    monkeypatch.setattr(llm, "_audit_request_input", lambda request, **kwargs: (request, SafetyDecision("block", "Blocked")))
    events = await read_stream(await llm.stream_llm(request))
    assert events[-1]["data"]["finish_reason"] == "safety"
    assert persisted == []


def test_preview_cache_is_bounded_and_expires(monkeypatch):
    now = [1.0]
    monkeypatch.setattr("services.llm.voice_preview.time.monotonic", lambda: now[0])
    cache = VoicePreviewCache(limit=1, ttl=5)
    request = llm_request(defer_persistence=True)
    first = cache.put(request, None, LLMResponse(request_id="request", text="first"))
    second = cache.put(request, None, LLMResponse(request_id="request", text="second"))
    assert cache.take(llm_request(voice_preview_token=first)) is None
    now[0] += 6
    assert cache.take(llm_request(voice_preview_token=second)) is None
    assert not cache.entries


async def test_runtime_never_writes_deferred_turns():
    runtime = object.__new__(LLMAgentRuntime)
    runtime.memory_store = SimpleNamespace(cache_message=lambda **kwargs: pytest.fail("Provisional memory write"))
    request = llm_request(defer_persistence=True)
    turn = AgentTurn(request, request, "owner", "question", context=SimpleNamespace())
    assert (await runtime.after_turn(turn, "answer"))["saved_turn"] is False


async def test_preview_weather_resolution_does_not_consume_pending_location():
    runtime = object.__new__(LLMAgentRuntime)
    runtime._pending_weather_locations = {}
    runtime.memory_store = SimpleNamespace(get_pending_weather_day=AsyncMock(return_value="tomorrow"))
    runtime.tool_agent = SimpleNamespace(weather_tool=SimpleNamespace(
        extract_query=lambda text: None, extract_location_reply=lambda text: "Beijing",
    ))
    assert "Beijing" in await runtime._preview_weather_query(user_id="owner", session_id="session", user_text="Beijing")
    assert runtime._pending_weather_locations == {}


async def test_generation_starts_before_final_and_waits_without_tts(preview_env, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()

    async def complete(request):
        assert request.defer_persistence
        started.set()
        await release.wait()
        return LLMResponse(request_id=request.request_id, text="answer", metadata={"voice_preview_token": "receipt"})

    monkeypatch.setattr(gateway.orchestrator.llm_provider, "complete", complete)
    preview = VoicePreview(voice_request())
    try:
        await preview.update("Explain how rain forms")
        await asyncio.wait_for(started.wait(), 1)
        finishing = asyncio.create_task(preview.finish("Explain how rain forms"))
        await asyncio.sleep(0)
        assert not finishing.done()
        release.set()
        assert await finishing == "receipt"
    finally:
        await preview.close()


async def test_changed_partial_cancels_old_work_and_final_mismatch_discards(preview_env, monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def complete(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(gateway.orchestrator.llm_provider, "complete", complete)
    preview = VoicePreview(voice_request())
    await preview.update("Explain how rain forms")
    await asyncio.wait_for(started.wait(), 1)
    await preview.update("Explain how snow forms")
    assert cancelled.is_set()
    assert await preview.finish("Actually explain clouds") is None
    assert preview.task is None


async def test_debounce_disabled_and_attempt_budget(preview_env, monkeypatch):
    calls = AsyncMock(return_value=LLMResponse(request_id="request", text="answer"))
    monkeypatch.setattr(gateway.orchestrator.llm_provider, "complete", calls)
    monkeypatch.setenv("GATEWAY_VOICE_PREVIEW_MAX_ATTEMPTS", "1")
    preview = VoicePreview(voice_request())
    await preview.update("Explain how rain forms")
    await preview.task
    await preview.update("Explain how clouds form")
    assert preview.task is None
    assert calls.await_count == 1
    monkeypatch.setenv("GATEWAY_VOICE_PREVIEW_ENABLED", "false")
    disabled = VoicePreview(voice_request())
    await disabled.update("Explain how rain forms")
    assert disabled.task is None
    monkeypatch.setenv("GATEWAY_VOICE_PREVIEW_STABLE_SECONDS", "5")
    waiting = VoicePreview(voice_request())
    waiting.enabled = True
    await waiting.update("Explain how rain forms")
    assert await waiting.finish("Explain how rain forms") is None
    assert calls.await_count == 1


async def test_preview_is_opt_in_and_does_not_generate_by_default(monkeypatch):
    monkeypatch.delenv("GATEWAY_VOICE_PREVIEW_ENABLED", raising=False)
    calls = AsyncMock()
    monkeypatch.setattr(gateway.orchestrator.llm_provider, "complete", calls)
    preview = VoicePreview(voice_request())
    await preview.update("Explain how rain forms")
    assert await preview.finish("Explain how rain forms") is None
    assert preview.attempts == 0
    calls.assert_not_called()


@pytest.mark.parametrize("continuous", [False, True])
async def test_browser_does_not_deliver_preview_before_final(preview_env, monkeypatch, continuous):
    generated = asyncio.Event()
    messages, used_tokens = [], []

    async def complete(request):
        generated.set()
        return LLMResponse(request_id=request.request_id, text="answer", metadata={"voice_preview_token": "receipt"})

    async def respond(request, message):
        used_tokens.append(request.voice_preview_token)
        yield gateway._gateway_event(request, EventType.MESSAGE_DELTA, {"text": "answer"})

    async def send_json(payload):
        messages.append(payload)

    class Upstream:
        def __aiter__(self):
            return self.events()

        async def events(self):
            yield json.dumps({"type": "transcript.partial", "text": "Explain how rain forms"})
            await asyncio.wait_for(generated.wait(), 1)
            assert all(item["event"] == "transcript.partial" for item in messages)
            yield json.dumps({"type": "transcript.final", "text": "Explain how rain forms"})
            yield json.dumps({"type": "session.done"})

    monkeypatch.setattr(gateway.orchestrator.llm_provider, "complete", complete)
    monkeypatch.setattr(gateway, "_stream_voice_response", respond)
    request = voice_request(metadata={"continuous_turns": continuous})
    websocket = SimpleNamespace(send_json=send_json)
    segments = await gateway._collect_realtime_asr(websocket, Upstream(), request)
    if not continuous:
        await gateway._complete_realtime_voice_turn(websocket, request, segments)
    assert used_tokens == ["receipt"]
    assert any(item["event"] == "message.delta" for item in messages)


async def test_gateway_rechecks_final_keyword_filter(preview_env, monkeypatch):
    from services.gateway.keyword_filter import KeywordFilterDecision

    monkeypatch.setattr(gateway, "_evaluate_asr_keyword_filter", AsyncMock(return_value=KeywordFilterDecision(blocked=True)))
    stream = AsyncMock(side_effect=AssertionError("Blocked final must not invoke LLM"))
    monkeypatch.setattr(gateway.orchestrator.llm_provider, "stream_complete", stream)
    request = voice_request(voice_preview_token="receipt")
    events = [event async for event in gateway._stream_voice_response(request, Message(role=MessageRole.USER, content="question"))]
    assert any(event.event == EventType.MESSAGE_DELTA for event in events)
    assert not stream.called


@pytest.mark.parametrize("failure", ["timeout", "upstream"])
async def test_preview_failure_releases_task_and_falls_back(preview_env, monkeypatch, failure):
    started = asyncio.Event()
    monkeypatch.setenv("GATEWAY_VOICE_PREVIEW_FINAL_WAIT_SECONDS", "0.01")

    async def complete(request):
        started.set()
        if failure == "upstream":
            raise RuntimeError("Upstream unavailable")
        await asyncio.Event().wait()

    monkeypatch.setattr(gateway.orchestrator.llm_provider, "complete", complete)
    preview = VoicePreview(voice_request())
    await preview.update("Explain how rain forms")
    await asyncio.wait_for(started.wait(), 1)
    assert await preview.finish("Explain how rain forms") is None
    assert preview.task is None


async def test_preview_keyword_block_never_calls_model(preview_env, monkeypatch):
    from services.gateway.keyword_filter import KeywordFilterDecision

    monkeypatch.setattr(gateway, "_evaluate_asr_keyword_filter", AsyncMock(return_value=KeywordFilterDecision(blocked=True)))
    complete = AsyncMock(side_effect=AssertionError("Blocked partial reached model"))
    monkeypatch.setattr(gateway.orchestrator.llm_provider, "complete", complete)
    preview = VoicePreview(voice_request())
    await preview.update("Explain how rain forms")
    await preview.task
    assert await preview.finish("Explain how rain forms") is None
    assert not complete.called


async def test_gateway_to_llm_preview_preserves_final_conversation(local_llm, preview_env, monkeypatch):
    from app_protocols import StreamEvent

    prepared, persisted = local_llm
    monkeypatch.setattr(gateway.orchestrator.llm_provider, "complete", llm.complete_llm)

    async def stream(request):
        for event in await read_stream(await llm.stream_llm(request)):
            yield StreamEvent(**event)

    monkeypatch.setattr(gateway.orchestrator.llm_provider, "stream_complete", stream)
    request = voice_request()
    preview = VoicePreview(request)
    await preview.update("Explain how rain forms")
    await preview.task
    assert persisted == []
    request.voice_preview_token = await preview.finish("Explain how rain forms")
    message = Message(role=MessageRole.USER, content="Explain how rain forms", metadata={"asr": {"language": "en"}})
    events = [event async for event in gateway._stream_voice_response(request, message)]
    assert len(prepared) == len(persisted) == 1
    assert events[-1].data["runtime"]["voice_preview_used"] is True


async def test_robot_only_plays_after_transcription_finishes(preview_env, monkeypatch):
    from services.gateway.xiaozhi import pipeline
    from services.gateway.xiaozhi.session import DeviceSession

    transcript = "Explain how rain forms"
    writer = SimpleNamespace(generation=0, send=AsyncMock(), flush=AsyncMock())
    session = DeviceSession(
        {"user_id": "owner", "voice": "liliya", "device_id": "device", "client_id": "client", "role": "tutor"},
        writer, 24000,
    )
    complete = AsyncMock(return_value=LLMResponse(request_id="request", text="answer", metadata={"voice_preview_token": "receipt"}))
    monkeypatch.setattr(gateway.orchestrator.llm_provider, "complete", complete)

    async def transcribe(session, mode):
        await session.preview.update(transcript)
        await session.preview.task
        assert writer.send.await_count == 0
        return transcript

    async def response(request, message):
        assert request.voice_preview_token == "receipt"
        assert writer.send.call_args.args[0]["final"] is True
        yield gateway._gateway_event(request, EventType.AUDIO_STARTED, {})
        yield gateway._gateway_event(request, EventType.AUDIO_DONE, {})

    monkeypatch.setattr(pipeline, "transcribe", transcribe)
    monkeypatch.setattr(gateway, "_stream_voice_response", response)
    await pipeline.run_turn(session, "manual")
    assert [call.args[0].get("state") for call in writer.send.call_args_list] == [None, "start", "stop"]
    assert session.preview is None


async def test_llm_limits_speculation_and_releases_cancelled_slots(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(llm, "voice_preview_slots", asyncio.BoundedSemaphore(2))
    started = asyncio.Queue()

    async def complete(request):
        await started.put(True)
        await asyncio.Event().wait()

    monkeypatch.setattr(llm, "_complete_llm", complete)
    tasks = [asyncio.create_task(llm.complete_llm(llm_request(defer_persistence=True))) for _ in range(2)]
    try:
        await asyncio.wait_for(started.get(), 1)
        await asyncio.wait_for(started.get(), 1)
        with pytest.raises(HTTPException) as error:
            await llm.complete_llm(llm_request(defer_persistence=True))
        assert error.value.status_code == 429
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert not llm.voice_preview_slots.locked()
