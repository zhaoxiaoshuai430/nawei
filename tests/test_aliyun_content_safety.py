"""Moderation policy, failures, and HTTP/SSE release boundaries."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app_protocols import EventType, LLMRequest, LLMResponse, Message, MessageRole, StreamEvent, Usage
from services.llm import main
from services.llm.agent_runtime import AgentTurn
from services.llm.content_safety import AliyunContentSafety, MAX_TEXT_CHARS, REPLIES, fallback_for, parse_result
from services.llm.safety import SafetyDecision


def body(labels="", risk="", **updates):
    return {"Code": 200, "Data": {"labels": labels, "reason": json.dumps({"riskLevel": risk})},
            "RequestId": "vendor-request", **updates}


def request():
    return LLMRequest(request_id="review-request", trace_id="review-trace", session_id="review-session",
                      user_id="review-user", messages=[Message(role=MessageRole.USER, content="hello")])


@pytest.mark.parametrize("label,category", [
    ("profanity", "profanity"), ("porn", "sexual"), ("violence", "violence"),
    ("contraband", "illegal"), ("medical", "medical"), ("political", "sensitive"), ("ad", "advertisement"),
    ("privacy", "privacy"), ("suicide", "self_harm"), ("future_vendor_label", "default"),
])
def test_category_replies(label, category):
    assert fallback_for([label]) == (category, REPLIES[category])
    assert parse_result(body(label, "high"))["blocked"]


def test_priority_and_risk_policy():
    assert fallback_for(["profanity", "porn", "violence"])[0] == "violence"
    assert fallback_for(["violence", "porn", "profanity"])[0] == "violence"
    assert not parse_result(body())["blocked"]
    assert not parse_result(body("nonLabel"))["blocked"]
    assert not parse_result(body("profanity", "low"))["blocked"]
    assert parse_result(body("profanity", "medium"))["blocked"]
    assert parse_result(body("unknown"))["blocked"]
    assert parse_result(body("", "high"))["blocked"]


@pytest.mark.parametrize("payload", [
    {}, {"Code": 200, "Data": {}}, body(Code=500),
    {"Code": 200, "Data": {"labels": [], "reason": ""}},
    {"Code": 200, "Data": {"labels": "", "reason": "bad-json"}},
    body("profanity", "unexpected"),
])
def test_invalid_response_is_not_a_pass(payload):
    with pytest.raises((ValueError, TypeError)):
        parse_result(payload)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ALIYUN_CONTENT_SAFETY_ENABLED", "true")
    client = AliyunContentSafety()
    client._request = AsyncMock(return_value=body())
    monkeypatch.setattr(main, "content_safety", client)
    monkeypatch.setattr(main, "load_settings", lambda: main.LLMSettings())
    return client


async def test_disabled_does_not_call_vendor(monkeypatch):
    client = AliyunContentSafety()
    client._request = AsyncMock(side_effect=AssertionError("Must not run"))
    decision = SafetyDecision("allow", "hello")
    assert await client.audit(decision, stage="input", request_id="r") == decision
    client._request.assert_not_called()


@pytest.mark.parametrize("error", [TimeoutError(), ValueError("secret-do-not-log"), RuntimeError("sdk failure")])
async def test_failure_has_separate_reply_and_no_secret_log(client, error, caplog):
    client._request.side_effect = error
    result = await client.audit(SafetyDecision("allow", "hello"), stage="input", request_id="r")
    assert result.blocked and result.content == REPLIES["unavailable"]
    assert result.external_review["failed"]
    assert "secret-do-not-log" not in caplog.text


async def test_actual_timeout_is_enforced(client, monkeypatch):
    monkeypatch.setenv("ALIYUN_CONTENT_SAFETY_TIMEOUT_SECONDS", "0.01")
    async def slow(*args):
        await asyncio.sleep(1)
    client._request.side_effect = slow
    result = await client.audit(SafetyDecision("allow", "hello"), stage="input", request_id="r")
    assert result.external_review["error_type"] == "TimeoutError"


async def test_cancellation_propagates(client):
    client._request.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await client.audit(SafetyDecision("allow", "hello"), stage="input", request_id="r")


async def test_all_windows_and_multiple_categories(client):
    client._request.side_effect = [body(), body("profanity", "medium"), body("violence", "high")]
    result = await client.audit(SafetyDecision("allow", "x" * 1000), stage="output", request_id="r")
    assert result.content == REPLIES["violence"]
    windows = [call.args[0] for call in client._request.call_args_list]
    assert list(map(len, windows)) == [500, 500, 200]
    assert result.external_review["category"] == "violence"


@pytest.mark.parametrize("stream", [False, True])
async def test_blocked_input_skips_agent_and_persistence(client, monkeypatch, stream):
    client._request.return_value = body("profanity", "medium")
    prepare = AsyncMock(side_effect=AssertionError("Agent must not run"))
    persist = AsyncMock(side_effect=AssertionError("Memory must not run"))
    monkeypatch.setattr(main.agent_runtime, "prepare", prepare)
    monkeypatch.setattr(main.agent_runtime, "after_turn", persist)
    if stream:
        response = await main.stream_llm(request())
        result = "".join([frame async for frame in response.body_iterator])
        assert REPLIES["profanity"] in result
        assert '"finish_reason":"safety"' in result
    else:
        response = await main.complete_llm(request())
        assert response.text == REPLIES["profanity"]
        assert response.finish_reason == "safety"
    prepare.assert_not_called()
    persist.assert_not_called()


@pytest.mark.parametrize("blocked", [False, True])
async def test_nonstream_output_and_memory_receive_only_reviewed_answer(client, monkeypatch, blocked):
    client._request.side_effect = [body(), body("porn", "high") if blocked else body()]
    req = request()
    turn = AgentTurn(req, req, req.user_id, "hello")
    monkeypatch.setattr(main.agent_runtime, "prepare", AsyncMock(return_value=turn))
    persist = AsyncMock(return_value={"saved_turn": True})
    monkeypatch.setattr(main.agent_runtime, "after_turn", persist)
    monkeypatch.setattr(main, "complete_stub", lambda r: LLMResponse(
        request_id=r.request_id, text="candidate output", usage=Usage(input_tokens=1, output_tokens=2)))
    result = await main.complete_llm(req)
    assert result.text == (REPLIES["sexual"] if blocked else "candidate output")
    if blocked:
        persist.assert_not_called()
    else:
        persist.assert_awaited_once_with(turn, "candidate output")


@pytest.mark.parametrize("mode", ["allow", "block", "error", "overflow"])
async def test_stream_releases_nothing_until_review_and_closes_source(client, monkeypatch, mode):
    client._request.side_effect = [body(), body("profanity", "medium") if mode == "block" else body()]
    req = request()
    turn = AgentTurn(req, req, req.user_id, "hello")
    monkeypatch.setattr(main.agent_runtime, "prepare", AsyncMock(return_value=turn))
    persist = AsyncMock(return_value={"saved_turn": True})
    monkeypatch.setattr(main.agent_runtime, "after_turn", persist)
    finished = False
    ids = {"request_id": req.request_id, "trace_id": req.trace_id, "session_id": req.session_id}

    async def source(*args):
        nonlocal finished
        try:
            yield StreamEvent(event=EventType.MESSAGE_DELTA, data={"text": "candidate "}, **ids)
            if mode == "error":
                yield StreamEvent(event=EventType.ERROR, data={"message": "private upstream details"}, **ids)
                return
            yield StreamEvent(event=EventType.MESSAGE_DELTA, data={"text": "x" * (MAX_TEXT_CHARS + 1) if mode == "overflow" else "output"}, **ids)
            yield StreamEvent(event=EventType.MESSAGE_DONE, data={"text": "unreviewed done payload"}, **ids)
        finally:
            finished = True

    monkeypatch.setattr(main, "_stream_backend_with_fallback", source)
    response = await main.stream_llm(req)
    frames = []
    async for frame in response.body_iterator:
        assert finished
        if mode in {"allow", "block"}:
            assert client._request.await_count == 2
        frames.append(frame)
    result = "".join(frames)
    assert "unreviewed done payload" not in result
    if mode == "allow":
        assert '"text":"candidate output"' in result
        persist.assert_awaited_once_with(turn, "candidate output")
    else:
        assert "candidate" not in result
        persist.assert_not_called()
        if mode == "block":
            assert REPLIES["profanity"] in result
        elif mode == "overflow":
            assert REPLIES["unavailable"] in result
        else:
            assert "private upstream details" not in result
