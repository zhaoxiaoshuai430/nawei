import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from services.llm import memory


def postgres_pool(execute):
    @asynccontextmanager
    async def transaction():
        yield

    connection = SimpleNamespace(execute=execute, transaction=transaction)

    @asynccontextmanager
    async def acquire():
        yield connection

    return SimpleNamespace(acquire=acquire, terminate=Mock(), close=AsyncMock())


def redis_client():
    return SimpleNamespace(
        ping=AsyncMock(return_value=True),
        aclose=AsyncMock(),
        connection_pool=SimpleNamespace(disconnect=AsyncMock()),
    )


@pytest.mark.parametrize("failure", ["error", "cancel", "timeout"])
async def test_postgres_schema_failure_releases_unpublished_pool(monkeypatch, failure):
    entered = asyncio.Event()

    async def execute(query):
        entered.set()
        if failure == "error":
            raise RuntimeError("schema failed")
        await asyncio.Event().wait()

    pool = postgres_pool(execute)
    factory = AsyncMock(return_value=pool)
    monkeypatch.setattr(memory, "asyncpg", SimpleNamespace(create_pool=factory))
    monkeypatch.setattr(memory, "aioredis", None)
    monkeypatch.setenv("MEMORY_CONNECT_TIMEOUT_SECONDS", "0.01")
    store = memory.MemoryStore(enabled=True)
    task = asyncio.create_task(store.initialize())
    await asyncio.wait_for(entered.wait(), timeout=1)
    if failure == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await asyncio.wait_for(task, timeout=1)
    assert store._pool is None
    pool.terminate.assert_called_once()
    if failure == "cancel":
        replacement = postgres_pool(AsyncMock())
        factory.return_value = replacement
        await store.initialize()
        assert store._pool is replacement
        await store.close()


@pytest.mark.parametrize("failure", ["error", "cancel", "timeout"])
async def test_redis_initialization_failure_closes_candidate(monkeypatch, failure):
    monkeypatch.setenv("MEMORY_CONNECT_TIMEOUT_SECONDS", "0.01")
    candidate = redis_client()
    entered = asyncio.Event()

    async def ping():
        entered.set()
        if failure == "error":
            raise ConnectionError("redis unavailable")
        await asyncio.Event().wait()

    candidate.ping.side_effect = ping
    monkeypatch.setattr(memory, "aioredis", SimpleNamespace(from_url=Mock(return_value=candidate)))
    store = memory.MemoryStore(enabled=True)
    task = asyncio.create_task(store._ensure_redis())
    await asyncio.wait_for(entered.wait(), timeout=1)
    if failure == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await asyncio.wait_for(task, timeout=1)
    assert store._redis is None
    candidate.aclose.assert_awaited_once()
    candidate.connection_pool.disconnect.assert_awaited_once()


async def test_redis_concurrent_initialization_waits_for_verified_client(monkeypatch):
    candidate = redis_client()
    entered, release = asyncio.Event(), asyncio.Event()

    async def ping():
        entered.set()
        await release.wait()
        return True

    candidate.ping.side_effect = ping
    factory = Mock(return_value=candidate)
    monkeypatch.setattr(memory, "aioredis", SimpleNamespace(from_url=factory))
    store = memory.MemoryStore(enabled=True)
    first = asyncio.create_task(store._ensure_redis())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = asyncio.create_task(store._ensure_redis())
        await asyncio.sleep(0)
        assert store._redis is None
        assert not second.done()
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []))
        await store.close()
    factory.assert_called_once()
    candidate.ping.assert_awaited_once()


@pytest.mark.parametrize("operation", [
    "history", "persist", "health", "profile_read", "profile_write", "profile_invalidate", "clear_session", "clear_user",
])
async def test_failed_redis_operation_closes_client_and_backs_off(monkeypatch, operation):
    candidate = redis_client()
    failure = ConnectionError("redis unavailable")
    candidate.lrange = AsyncMock(side_effect=failure)
    candidate.rpush = AsyncMock(side_effect=failure)
    candidate.get = AsyncMock(side_effect=failure if operation == "profile_read" else None, return_value=None)
    candidate.setex = AsyncMock(side_effect=failure)
    candidate.delete = AsyncMock(side_effect=failure)
    candidate.ping.side_effect = failure
    monkeypatch.setattr(memory, "aioredis", None)
    store = memory.MemoryStore(enabled=True)
    store.initialize = AsyncMock()
    store._redis = candidate
    if operation == "history":
        assert await store.get_history(user_id="owner", session_id="session") == []
    elif operation == "persist":
        await store.persist_message(user_id="owner", session_id="session", role="user", content="hello")
    elif operation == "health":
        assert (await store.health())["redis"] == "fallback"
    elif operation in {"profile_read", "profile_write"}:
        store.profile_redis_cache_enabled = True
        assert await store.get_profile("owner") == {}
    elif operation == "profile_invalidate":
        store.profile_redis_cache_enabled = False
        await store.update_profile(user_id="owner", changes={"nickname": "小明"}, reason="test", session_id="session")
    else:
        await store.clear_user(user_id="owner", session_id="session" if operation == "clear_session" else None)
    assert store._redis is None
    candidate.aclose.assert_awaited_once()
    assert store._redis_retry_after > memory.time.monotonic()


async def test_late_redis_failure_does_not_discard_replacement():
    old_client, replacement = redis_client(), redis_client()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def lrange(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        raise ConnectionError("old connection failed")

    old_client.lrange = AsyncMock(side_effect=lrange)
    store = memory.MemoryStore(enabled=False)
    store._redis = old_client
    pending = asyncio.create_task(store.get_history(user_id="owner", session_id="session"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await store.get_history(user_id="owner", session_id="session")
        store._redis = replacement
        release.set()
        await pending
        assert store._redis is replacement
        old_client.aclose.assert_awaited_once()
        replacement.aclose.assert_not_awaited()
    finally:
        release.set()
        await pending
        await store.close()


async def test_redis_recovers_after_backoff_without_leaking_failed_client(monkeypatch):
    failed, recovered = redis_client(), redis_client()
    failed.ping.side_effect = ConnectionError("redis unavailable")
    factory = Mock(side_effect=[failed, recovered])
    monkeypatch.setattr(memory, "aioredis", SimpleNamespace(from_url=factory))
    store = memory.MemoryStore(enabled=True)
    await store._ensure_redis()
    await store._ensure_redis()
    factory.assert_called_once()
    assert store._redis is None
    failed.aclose.assert_awaited_once()
    store._redis_retry_after = 0
    await store._ensure_redis()
    assert store._redis is recovered
    assert store._redis_last_error is None
    assert store._redis_retry_after == 0
    await store.close()
    recovered.aclose.assert_awaited_once()


async def test_redis_cleanup_disconnects_pool_when_client_close_fails():
    candidate = redis_client()
    candidate.aclose.side_effect = ConnectionError("close failed")
    store = memory.MemoryStore(enabled=False)
    store._redis = candidate
    await store.close()
    await store.close()
    candidate.aclose.assert_awaited_once()
    candidate.connection_pool.disconnect.assert_awaited_once()
    assert store._redis is None
