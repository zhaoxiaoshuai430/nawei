import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import anyio
import pytest
from starlette.datastructures import Headers

from app_protocols import LLMRequest, Message, MessageRole
from services.gateway import main as gateway
from services.gateway.xiaozhi import websocket
from services.gateway.xiaozhi.device_registry import DeviceRegistry
from services.gateway.xiaozhi.session import DeviceSession, SessionManager
from services.llm import main as llm
from services.llm.agent_runtime import AgentTurn, LLMAgentRuntime
from services.llm.drain import LocalDrainMiddleware, LocalDrainState
from services.llm.memory import MemoryStore


def history_client(rows):
    return SimpleNamespace(lrange=AsyncMock(return_value=[json.dumps(row) for row in rows]),
                           rpush=AsyncMock(), ltrim=AsyncMock(), expire=AsyncMock())


async def test_history_merges_pending_messages_without_overriding_other_workers():
    store = MemoryStore(enabled=False)
    store.cache_message(user_id="owner", session_id="session", role="user", content="old", message_id="old")
    await store.persist_message(user_id="owner", session_id="session", role="user", content="old", message_id="old")
    store._memory_messages[store._history_key("owner", "session")][0]["redis_persisted"] = True
    store.cache_message(user_id="owner", session_id="session", role="assistant", content="latest", message_id="new")
    store._redis = history_client([{"role": "user", "content": "other worker", "message_id": "remote", "ts": 1}])
    result = await store.get_history(user_id="owner", session_id="session")
    assert [item["content"] for item in result] == ["other worker", "latest"]


async def test_history_preserves_repeated_messages_and_deduplicates_by_id():
    store = MemoryStore(enabled=False)
    for message_id in ("one", "two"):
        store.cache_message(user_id="owner", session_id="session", role="user", content="again", message_id=message_id)
    local = store._memory_messages[store._history_key("owner", "session")]
    store._redis = history_client([local[0]])
    assert len(await store.get_history(user_id="owner", session_id="session")) == 2
    assert len(await store.get_history(user_id="owner", session_id="session", limit=1)) == 1
    store._redis = history_client([])
    assert await store.get_history(user_id="other", session_id="session") == []


async def test_history_read_racing_redis_publish_keeps_pending_snapshot():
    store = MemoryStore(enabled=False)
    store.cache_message(user_id="owner", session_id="session", role="user", content="latest", message_id="new")
    started, release = asyncio.Event(), asyncio.Event()

    async def read(*args):
        started.set()
        await release.wait()
        return [json.dumps({"role": "user", "content": "old", "ts": 1})]

    store._redis = history_client([])
    store._redis.lrange.side_effect = read
    task = asyncio.create_task(store.get_history(user_id="owner", session_id="session"))
    await started.wait()
    await store.persist_message(user_id="owner", session_id="session", role="user", content="latest", message_id="new")
    release.set()
    assert [item["content"] for item in await task] == ["old", "latest"]


@pytest.mark.parametrize("ready", [True, False])
async def test_readiness_http_codes_preserve_liveness(monkeypatch, ready):
    monkeypatch.setattr(gateway, "_dependency_status", AsyncMock(return_value={"status": "ready" if ready else "degraded"}))
    monkeypatch.setattr(llm, "health", AsyncMock(return_value={"status": "ok" if ready else "degraded"}))
    for app in (gateway.app, llm.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/ready")
            assert response.status_code == (200 if ready else 503)
    assert (await gateway.health())["status"] == "ok"


async def test_listen_authentication_does_not_block_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    called = []

    def authenticate(*args):
        called.append(threading.get_ident())
        time.sleep(0.05)
        return {"device_id": "device", "user_id": "owner"}

    socket = SimpleNamespace(receive=AsyncMock(side_effect=[
        {"type": "websocket.receive", "text": json.dumps({"type": "listen", "state": "start"})},
        {"type": "websocket.disconnect"},
    ]))
    session = SimpleNamespace(session_id="session", device={"device_id": "device"}, turn=None, turn_number=0,
                              cancel_turn=AsyncMock(), writer=SimpleNamespace(turn_complete=True), listening=False)
    monkeypatch.setattr(websocket, "run_turn", AsyncMock())
    await websocket._receive(socket, session, SimpleNamespace(authenticate=authenticate), "client", "token")
    await session.turn
    assert called and called[0] != loop_thread


def test_registry_reads_do_not_request_write_lock(tmp_path):
    registry = DeviceRegistry(tmp_path / "devices.sqlite3")
    with registry.transaction():
        assert registry.list_devices("owner") == []


async def test_websocket_cancellation_completes_session_cleanup(monkeypatch):
    connected, cleaned = asyncio.Event(), asyncio.Event()
    hello = {"type": "hello", "version": 1, "transport": "websocket",
             "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60}}
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "websocket.receive", "text": json.dumps(hello)}
        await asyncio.Event().wait()

    async def send_json(payload):
        connected.set()

    async def cancel_turn(self):
        await anyio.sleep(0.01)
        cleaned.set()

    monkeypatch.setattr(DeviceSession, "cancel_turn", cancel_turn)
    socket = SimpleNamespace(headers=Headers({"Device-Id": "aa:bb:cc:dd:ee:01", "Client-Id": "test-client-00001",
                                              "Protocol-Version": "1", "Authorization": "Bearer nwdev_test"}),
                             receive=receive, accept=AsyncMock(), send_json=send_json, close=AsyncMock())
    registry = SimpleNamespace(authenticate=lambda *args: {"device_id": "device"}, seen=lambda *args: None)
    sessions = SessionManager()
    async with anyio.create_task_group() as group:
        group.start_soon(websocket.serve, socket, registry, sessions, SimpleNamespace(hello_timeout=1, heartbeat_interval=25))
        await asyncio.wait_for(connected.wait(), timeout=1)
        group.cancel_scope.cancel()
    assert cleaned.is_set()
    assert not sessions.active


def persistence_fixture():
    store = MemoryStore(enabled=False)
    runtime = LLMAgentRuntime(memory_store=store, knowledge_client=SimpleNamespace(close=AsyncMock()))
    request = LLMRequest(request_id="request", trace_id="trace", session_id="session", user_id="owner",
                         messages=[Message(role=MessageRole.USER, content="hello")])
    turn = AgentTurn(request=request, original_request=request, user_id="owner", user_text="hello",
                     context=SimpleNamespace(), direct_answer="answer")
    return runtime, turn


async def test_persistence_backpressure_bounds_created_tasks(monkeypatch):
    monkeypatch.setenv("LLM_AGENT_ENABLED", "true")
    monkeypatch.setenv("MEMORY_POST_TURN_ASYNC", "true")
    monkeypatch.setenv("MEMORY_POST_TURN_MAX_PENDING", "2")
    runtime, turn = persistence_fixture()
    turn.direct_answer = None
    release = asyncio.Event()

    async def persist(**kwargs):
        await release.wait()

    runtime.skill_manager.after_turn = AsyncMock(side_effect=persist)
    await runtime.after_turn(turn, "one")
    await runtime.after_turn(turn, "two")
    waiting = asyncio.create_task(runtime.after_turn(turn, "three"))
    await asyncio.sleep(0)
    assert not waiting.done()
    assert runtime.persistence_status()["pending"] == 2
    release.set()
    await waiting
    await runtime.close()
    assert runtime.persistence_status()["pending"] == 0


@pytest.mark.parametrize("background", [True, False])
async def test_persistence_admission_and_sync_timeouts_prevent_safe_drain(monkeypatch, background, tmp_path):
    monkeypatch.setenv("LLM_AGENT_ENABLED", "true")
    monkeypatch.setenv("MEMORY_POST_TURN_ASYNC", str(background).lower())
    monkeypatch.setenv("MEMORY_POST_TURN_MAX_PENDING", "1")
    monkeypatch.setenv("MEMORY_POST_TURN_TIMEOUT_SECONDS", "10")
    runtime, turn = persistence_fixture()
    turn.direct_answer = None
    entered, release = asyncio.Event(), asyncio.Event()

    async def persist(**kwargs):
        entered.set()
        await release.wait()

    runtime.skill_manager.after_turn = persist
    if background:
        await runtime.after_turn(turn, "one")
        await asyncio.wait_for(entered.wait(), timeout=1)
    monkeypatch.setenv("MEMORY_POST_TURN_TIMEOUT_SECONDS", "0.01")
    with pytest.raises(TimeoutError):
        await runtime.after_turn(turn, "two")
    release.set()
    await runtime.close()
    state = LocalDrainState(str(tmp_path))
    (tmp_path / "drain.request").touch()
    assert runtime.persistence_status()["failed"] == 1
    assert runtime.persistence_status()["pending"] == 0
    assert not state.snapshot(runtime)["drained"]


async def test_persistence_timeout_is_reported_and_shutdown_is_bounded(monkeypatch):
    monkeypatch.setenv("LLM_AGENT_ENABLED", "true")
    monkeypatch.setenv("MEMORY_POST_TURN_ASYNC", "true")
    monkeypatch.setenv("MEMORY_SHUTDOWN_TIMEOUT_SECONDS", "0.01")
    runtime, turn = persistence_fixture()
    turn.direct_answer = None

    async def stalled(**kwargs):
        await asyncio.Event().wait()

    runtime.skill_manager.after_turn = stalled
    await runtime.after_turn(turn, "answer")
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(runtime.close(), timeout=1)
    assert runtime.persistence_status()["failed"] == 1
    assert runtime.persistence_status()["pending"] == 0


async def test_local_drain_counts_complete_stream_and_rejects_new_requests(tmp_path):
    state = LocalDrainState(str(tmp_path))
    entered, release = asyncio.Event(), asyncio.Event()

    async def streaming(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        entered.set()
        await release.wait()
        await send({"type": "http.response.body", "body": b"done"})

    runtime = SimpleNamespace(persistence_status=lambda: {"pending": 0, "failed": 0})
    middleware = LocalDrainMiddleware(streaming, state)
    scope = {"type": "http", "path": "/api/v1/llm/stream", "method": "POST"}
    task = asyncio.create_task(middleware(scope, AsyncMock(), AsyncMock()))
    await entered.wait()
    (tmp_path / "drain.request").touch()
    assert not state.snapshot(runtime)["drained"]
    rejected = AsyncMock()
    await middleware(scope, AsyncMock(), rejected)
    assert rejected.call_args_list[0].args[0]["status"] == 503
    release.set()
    await task
    assert state.snapshot(runtime)["drained"]
    runtime.persistence_status = lambda: {"pending": 0, "failed": 1}
    assert not state.snapshot(runtime)["drained"]


@pytest.mark.parametrize(("token", "path", "expected"), [
    ("private-worker-token", "/api/v1/llm", 200),
    ("wrong", "/api/v1/llm", 503),
    ("private-worker-token", "/api/v1/llm/stream", 503),
])
async def test_drain_allows_only_trusted_background_extraction(tmp_path, monkeypatch, token, path, expected):
    monkeypatch.setenv("LLM_DRAIN_TOKEN", "private-worker-token")
    state = LocalDrainState(str(tmp_path))
    (tmp_path / "drain.request").touch()

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    send = AsyncMock()
    await LocalDrainMiddleware(downstream, state)(
        {"type": "http", "path": path, "method": "POST", "headers": [(b"x-nawei-drain-token", token.encode())]},
        AsyncMock(), send,
    )
    assert send.call_args_list[0].args[0]["status"] == expected
    assert state.active == 0
