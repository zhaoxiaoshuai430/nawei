"""Failed startup must not publish or abandon partially initialized resources."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from services.knowledge.catalog import KnowledgeCatalog


@pytest.mark.parametrize("cancel", [False, True])
async def test_catalog_initialization_failure_reclaims_pool_and_can_retry(monkeypatch, cancel):
    entered, release = asyncio.Event(), asyncio.Event()

    async def initialize_schema(*args):
        entered.set()
        await release.wait()
        raise RuntimeError("schema unavailable")

    connection = SimpleNamespace(execute=AsyncMock(side_effect=initialize_schema))
    acquire = AsyncMock()
    acquire.__aenter__.return_value = connection
    pool = SimpleNamespace(acquire=Mock(return_value=acquire), terminate=Mock(), close=AsyncMock())
    create_pool = AsyncMock(return_value=pool)
    monkeypatch.setattr("services.knowledge.catalog.asyncpg", SimpleNamespace(create_pool=create_pool))
    catalog = KnowledgeCatalog("postgresql://unused")
    task = asyncio.create_task(catalog.initialize())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            release.set()
            await asyncio.wait_for(task, 1)
        assert catalog.pool is None
        pool.terminate.assert_called_once()
        if not cancel:
            assert "schema unavailable" in catalog.last_error

        connection.execute.side_effect = None
        await catalog.initialize()
        assert create_pool.await_count == 2
        assert catalog.pool is pool
        assert catalog.last_error is None
        await catalog.close()
        pool.close.assert_awaited_once()
        assert catalog.pool is None
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("failure", [RuntimeError("startup failed"), asyncio.CancelledError()])
async def test_knowledge_lifespan_closes_after_failed_startup(monkeypatch, failure):
    from services.knowledge import main

    service = SimpleNamespace(initialize=AsyncMock(side_effect=failure), close=AsyncMock())
    monkeypatch.setattr(main, "knowledge_service", service)
    with pytest.raises(type(failure)):
        async with main.lifespan(main.app):
            pytest.fail("Failed startup must never yield a ready application")
    service.close.assert_awaited_once()
