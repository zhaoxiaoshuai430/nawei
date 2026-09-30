"""Low-latency realtime Gateway streaming regression tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from app_protocols import EventType, LLMRequest, Message, StreamEvent, TTSRequest
from services.gateway import main as gateway_main


class StreamingLLMProvider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        for text in (
            "第一段已经完整。",
            "第二段仍在生成。",
            "\n- 资料1：OpenStax（莱斯大学）《Precalculus 2e》，路径：第5章第2节·Unit Circle，链接：https://openstax.org/\n",
        ):
            yield StreamEvent(
                event=EventType.MESSAGE_DELTA,
                request_id=request.request_id,
                trace_id=request.trace_id,
                session_id=request.session_id,
                data={"text": text},
            )
        yield StreamEvent(
            event=EventType.MESSAGE_DONE,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"finish_reason": "stop"},
        )


class BlockingStreamingTTSProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.requests: list[TTSRequest] = []

    async def stream_synthesize(self, request: TTSRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        yield StreamEvent(
            event=EventType.AUDIO_DELTA,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"audio_base64": "UklGRg==", "ttfa_ms": 10},
        )


class GatedStreamingLLMProvider:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(
            event=EventType.MESSAGE_DELTA,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"text": "第一段文字开始了"},
        )
        await self.release.wait()
        yield StreamEvent(
            event=EventType.MESSAGE_DELTA,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"text": "，后续内容仍在生成。"},
        )
        self.finished.set()
        yield StreamEvent(
            event=EventType.MESSAGE_DONE,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"finish_reason": "stop"},
        )


def test_tts_segments_flush_on_soft_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_main, "_TTS_SOFT_SEGMENT_MIN_CHARS", 10)
    monkeypatch.setattr(gateway_main, "_TTS_EAGER_SEGMENT_CHARS", 20)

    segments, remainder = gateway_main._extract_tts_segments("abcdefghij,tail")

    assert segments == ["abcdefghij,"]
    assert remainder == "tail"


def test_tts_segments_flush_without_punctuation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_main, "_TTS_SOFT_SEGMENT_MIN_CHARS", 10)
    monkeypatch.setattr(gateway_main, "_TTS_EAGER_SEGMENT_CHARS", 12)

    segments, remainder = gateway_main._extract_tts_segments("abcdefghijklmnop")

    assert segments == ["abcdefghijkl"]
    assert remainder == "mnop"


def test_first_tts_segment_uses_lower_latency_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_main, "_TTS_FIRST_SEGMENT_CHARS", 6)
    monkeypatch.setattr(gateway_main, "_TTS_EAGER_SEGMENT_CHARS", 16)

    segments, remainder = gateway_main._extract_tts_segments("abcdef", first=True)

    assert segments == ["abcdef"]
    assert remainder == ""


def test_sentence_boundary_ends_first_segment_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_main, "_TTS_FIRST_SEGMENT_CHARS", 8)
    monkeypatch.setattr(gateway_main, "_TTS_EAGER_SEGMENT_CHARS", 16)

    segments, remainder = gateway_main._extract_tts_segments("第一句。abcdefgh", first=True)

    assert segments == ["第一句。"]
    assert remainder == "abcdefgh"


@pytest.mark.parametrize("chunks, number", [
    (["圆周率约为3.14。"], "3.14"),
    (["圆周率约为3.", "14。"], "3.14"),
    (["结果为-12", ".", "3456789", "。"], "-12.3456789"),
    (["123456", ".", "789", "。"], "123456.789"),
    (["结果为.5。"], ".5"),
    (["值为3.14"], "3.14"),
    (list("数值大约为-3.14。"), "-3.14"),
    (list("数值大约为.5。"), ".5"),
    (["a" * 449 + "3.14。"], "3.14"),
])
def test_streaming_tts_keeps_decimal_numbers_in_one_segment(chunks, number):
    output = []
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        segments, buffer = gateway_main._extract_tts_segments(buffer, first=not output)
        output.extend(segments)
    segments, buffer = gateway_main._extract_tts_segments(buffer, flush=True)
    output.extend(segments)
    assert "".join(output) == "".join(chunks)
    assert any(number in segment for segment in output)
    assert buffer == ""


def test_tts_keeps_english_sentence_boundaries_and_flushes_numeric_period():
    segments, buffer = gateway_main._extract_tts_segments("Hello. Next sentence.")
    assert segments == ["Hello.", "Next sentence."]
    assert buffer == ""
    segments, buffer = gateway_main._extract_tts_segments("123.", flush=True)
    assert segments == ["123."]
    assert buffer == ""


def test_long_tts_segments_remain_bounded_and_make_progress():
    for text in ["a" * 450 + ".", "1" * 900 + ".", "a" * 449 + "3.14."]:
        segments = gateway_main._split_long_tts_text(text)
        assert "".join(segments) == text
        assert all(0 < len(segment) <= 450 for segment in segments)


def test_tts_reference_filter_hides_sources_across_stream_chunks() -> None:
    reference_filter = gateway_main._TtsReferenceTextFilter()

    spoken = reference_filter.feed("这是可朗读答案。\n- 资")
    spoken += reference_filter.feed("料1：OpenStax（莱斯大学）《Precalculus 2e》，路径：第5章第2节·Unit Circle\n继续说明。")
    spoken += reference_filter.feed("", flush=True)

    assert "这是可朗读答案。" in spoken
    assert "继续说明。" in spoken
    assert "资料1" not in spoken
    assert "OpenStax" not in spoken


def test_tts_spoken_text_removes_markdown_and_urls() -> None:
    spoken = gateway_main._normalize_tts_spoken_text(
        "## **核心结论**\n- 查看 [OpenStax](https://openstax.org/) 的 `单位圆`，详情 https://example.com/page"
    )

    assert spoken == "核心结论，查看 OpenStax 的单位圆，详情"


def test_tts_spoken_text_keeps_sentence_punctuation_after_url() -> None:
    spoken = gateway_main._normalize_tts_spoken_text("详情 https://example.com/page。下一句。")

    assert spoken == "详情。下一句。"


def test_inline_reference_filter_removes_markers_across_stream_chunks() -> None:
    reference_filter = gateway_main._InlineReferenceMarkerFilter()

    displayed = reference_filter.feed("同一个三角形里换一个角，各边角色就互换了[资")
    displayed += reference_filter.feed("料5]。\n参考来源\n- 资料5：OpenStax《Precalculus 2e》")
    displayed += reference_filter.feed("", flush=True)

    assert "互换了。" in displayed
    assert "[资料5]" not in displayed
    assert "- 资料5：OpenStax" in displayed


@pytest.mark.asyncio
async def test_llm_text_continues_while_tts_is_generating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_provider = StreamingLLMProvider()
    tts_provider = BlockingStreamingTTSProvider()
    monkeypatch.setattr(gateway_main.orchestrator, "llm_provider", llm_provider)
    monkeypatch.setattr(gateway_main.orchestrator, "tts_provider", tts_provider)
    monkeypatch.delenv("GATEWAY_TTS_STREAM_CHUNK_SIZE", raising=False)
    request = gateway_main.RealtimeVoiceRequest(
        request_id="request-1",
        trace_id="trace-1",
        session_id="session-1",
        user_id="user-1",
    )
    events: list[StreamEvent] = []

    async def collect_events() -> None:
        async for event in gateway_main._stream_voice_response(
            request,
            Message(role="user", content="你好"),
        ):
            events.append(event)

    collector = asyncio.create_task(collect_events())
    try:
        await asyncio.wait_for(tts_provider.started.wait(), timeout=1)
        for _ in range(20):
            message_text = "".join(
                str(event.data.get("text") or "")
                for event in events
                if event.event == EventType.MESSAGE_DELTA
            )
            if "第二段仍在生成。" in message_text:
                break
            await asyncio.sleep(0)

        assert "第二段仍在生成。" in message_text
        assert not any(event.event == EventType.AUDIO_DELTA for event in events)
    finally:
        tts_provider.release.set()
        await asyncio.wait_for(collector, timeout=2)

    assert any(event.event == EventType.AUDIO_DELTA for event in events)
    assert any(event.event == EventType.AUDIO_DONE for event in events)
    assert llm_provider.requests
    assert llm_provider.requests[0].metadata["include_references"] is True
    assert tts_provider.requests
    assert tts_provider.requests[0].metadata["chunk_size"] == 8
    assert all(request.metadata["chunk_size"] == 12 for request in tts_provider.requests[1:])
    displayed_text = "".join(
        str(event.data.get("text") or "")
        for event in events
        if event.event == EventType.MESSAGE_DELTA
    )
    spoken_text = "".join(request.text for request in tts_provider.requests)
    assert "资料1" in displayed_text
    assert "OpenStax" in displayed_text
    assert "资料1" not in spoken_text
    assert "OpenStax" not in spoken_text


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata,expected", [({}, [8, 12]), ({"chunk_size": 16}, [16, 16])])
async def test_tts_chunk_policy_preserves_first_audio_latency_and_explicit_override(monkeypatch, metadata, expected):
    provider = BlockingStreamingTTSProvider()
    provider.release.set()
    monkeypatch.setattr(gateway_main.orchestrator, "tts_provider", provider)
    monkeypatch.setenv("GATEWAY_TTS_STREAM_CHUNK_SIZE", "8")
    monkeypatch.setenv("GATEWAY_TTS_CONTINUATION_CHUNK_SIZE", "12")
    request = gateway_main.RealtimeVoiceRequest(
        request_id="chunks", trace_id="chunks", session_id="chunks", user_id="chunks", metadata=metadata,
    )
    for index in range(2):
        events = [event async for event in gateway_main._stream_tts_segment(request, "A complete sentence.", index)]
        assert events[0].event == EventType.AUDIO_DELTA
    assert [item.metadata["chunk_size"] for item in provider.requests] == expected
    assert request.metadata == metadata


@pytest.mark.asyncio
async def test_tts_starts_before_llm_stream_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_provider = GatedStreamingLLMProvider()
    tts_provider = BlockingStreamingTTSProvider()
    monkeypatch.setattr(gateway_main.orchestrator, "llm_provider", llm_provider)
    monkeypatch.setattr(gateway_main.orchestrator, "tts_provider", tts_provider)
    monkeypatch.setattr(gateway_main, "_TTS_FIRST_SEGMENT_CHARS", 8)
    request = gateway_main.RealtimeVoiceRequest(
        request_id="request-early-tts",
        trace_id="trace-early-tts",
        session_id="session-early-tts",
        user_id="user-early-tts",
    )

    async def consume() -> None:
        async for _ in gateway_main._stream_voice_response(
            request,
            Message(role="user", content="你好"),
        ):
            pass

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(tts_provider.started.wait(), timeout=1)
        assert not llm_provider.finished.is_set()
    finally:
        llm_provider.release.set()
        tts_provider.release.set()
        await asyncio.wait_for(consumer, timeout=2)

    assert llm_provider.finished.is_set()
    assert tts_provider.requests[0].text == "第一段文字开始了"


@pytest.mark.asyncio
@pytest.mark.parametrize(("limit", "expected"), [
    (1, ["first.", "two.", "three.", "four."]),
    (12, ["first.", "two.three.", "four."]),
    (80, ["first.", "two.three.four."]),
])
async def test_tts_batch_preserves_first_segment_order_and_text(monkeypatch, limit, expected):
    monkeypatch.setenv("GATEWAY_TTS_BATCH_MAX_CHARS", str(limit))
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    for segment in ["", "first.", "two.", "three.", "four.", None]:
        queue.put_nowait(segment)

    async with asyncio.timeout(1):
        segments = [segment async for segment in gateway_main._queued_tts_segments(queue)]

    assert segments == expected
    assert "".join(segments) == "first.two.three.four."
    assert queue.empty()


@pytest.mark.asyncio
async def test_tts_batch_does_not_wait_for_more_text(monkeypatch):
    monkeypatch.setenv("GATEWAY_TTS_BATCH_MAX_CHARS", "80")
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    stream = gateway_main._queued_tts_segments(queue)
    try:
        async with asyncio.timeout(1):
            queue.put_nowait("first.")
            assert await anext(stream) == "first."
            queue.put_nowait("second.")
            assert await anext(stream) == "second."
            queue.put_nowait(None)
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_tts_batch_normalizes_each_fragment_before_joining(monkeypatch):
    monkeypatch.setenv("GATEWAY_TTS_BATCH_MAX_CHARS", "80")
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    for segment in ["**first.**", "- second.", "", "## third.", None]:
        queue.put_nowait(segment)
    segments = [segment async for segment in gateway_main._queued_tts_segments(queue)]
    assert segments == ["first.", "second.third."]


@pytest.mark.asyncio
async def test_tts_batch_caps_configuration_at_provider_text_limit(monkeypatch):
    monkeypatch.setenv("GATEWAY_TTS_BATCH_MAX_CHARS", "10000")
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    expected = ["first.", "a" * 300, "b" * 200]
    for segment in [*expected, None]:
        queue.put_nowait(segment)
    segments = [segment async for segment in gateway_main._queued_tts_segments(queue)]
    assert segments == expected


class RealtimeASRUpstream:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        for event in self.events:
            yield gateway_main.json.dumps(event)


class RealtimeClientWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.messages.append(payload)


@pytest.mark.asyncio
async def test_continuous_realtime_transcript_starts_turn_before_session_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed_turns: list[tuple[gateway_main.RealtimeVoiceRequest, list[dict[str, object]]]] = []

    async def complete_turn(websocket, request, segments, send_lock=None) -> None:
        completed_turns.append((request, segments))

    monkeypatch.setattr(gateway_main, "_complete_realtime_voice_turn", complete_turn)
    request = gateway_main.RealtimeVoiceRequest(
        request_id="request-live",
        trace_id="trace-live",
        session_id="session-live",
        user_id="user-live",
        metadata={"continuous_turns": True},
    )
    websocket = RealtimeClientWebSocket()
    upstream = RealtimeASRUpstream(
        [
            {"type": "transcript.partial", "segment_index": 0, "text": "第一"},
            {"type": "transcript.partial", "segment_index": 0, "text": "第一句"},
            {"type": "transcript.final", "segment_index": 0, "text": "第一句话"},
            {"type": "session.done", "segments": 1},
        ]
    )

    segments = await gateway_main._collect_realtime_asr(websocket, upstream, request)

    assert [segment["text"] for segment in segments] == ["第一句话"]
    assert len(completed_turns) == 1
    assert [message["event"] for message in websocket.messages[:2]] == ["transcript.partial", "transcript.partial"]
    turn_request, turn_segments = completed_turns[0]
    assert turn_request.request_id == "request-live-turn-1"
    assert turn_request.session_id == "session-live"
    assert turn_request.metadata["turn_number"] == 1
    assert turn_segments[0]["text"] == "第一句话"


@pytest.mark.asyncio
async def test_continuous_realtime_barge_in_cancels_active_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    completed_turns: list[int] = []

    async def complete_turn(websocket, request, segments, send_lock=None) -> None:
        turn_number = int(request.metadata["turn_number"])
        if turn_number == 1:
            first_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_cancelled.set()
                raise
        completed_turns.append(turn_number)

    class BargeInUpstream:
        def __aiter__(self) -> AsyncIterator[str]:
            return self._iterate()

        async def _iterate(self) -> AsyncIterator[str]:
            yield gateway_main.json.dumps({"type": "transcript.final", "segment_index": 0, "text": "第一句话"})
            await first_started.wait()
            yield gateway_main.json.dumps({"type": "speech.started", "segment_index": 1})
            await first_cancelled.wait()
            yield gateway_main.json.dumps({"type": "transcript.final", "segment_index": 1, "text": "第二句话"})
            yield gateway_main.json.dumps({"type": "session.done", "segments": 2})

    monkeypatch.setattr(gateway_main, "_complete_realtime_voice_turn", complete_turn)
    request = gateway_main.RealtimeVoiceRequest(
        request_id="request-barge",
        trace_id="trace-barge",
        session_id="session-barge",
        user_id="user-barge",
        metadata={"continuous_turns": True, "barge_in": True},
    )
    websocket = RealtimeClientWebSocket()

    segments = await asyncio.wait_for(
        gateway_main._collect_realtime_asr(websocket, BargeInUpstream(), request),
        timeout=2,
    )

    assert [segment["text"] for segment in segments] == ["第一句话", "第二句话"]
    assert completed_turns == [2]
    interrupted = [message for message in websocket.messages if message["event"] == "response.interrupted"]
    assert interrupted[0]["data"] == {"turn_number": 1, "reason": "barge_in", "audio_cleared": True}
