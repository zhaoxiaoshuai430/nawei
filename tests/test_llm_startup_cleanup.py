"""Startup failures and cancellation must close already initialized dependencies."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.llm import main


@pytest.mark.parametrize("failure", [RuntimeError("startup failed"), asyncio.CancelledError()])
async def test_failed_llm_startup_closes_runtime_and_http_clients(monkeypatch, failure):
    runtime = SimpleNamespace(initialize=AsyncMock(side_effect=failure), close=AsyncMock())
    client = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(main, "agent_runtime", runtime)
    monkeypatch.setattr(main, "_deepseek_clients", {1.0: client})
    with pytest.raises(type(failure)):
        async with main._llm_lifespan(main.app):
            pytest.fail("Failed startup cannot serve requests")
    runtime.close.assert_awaited_once()
    client.aclose.assert_awaited_once()
    assert main._deepseek_clients == {}


async def test_llm_shutdown_closes_http_even_if_runtime_close_fails(monkeypatch):
    runtime = SimpleNamespace(initialize=AsyncMock(), close=AsyncMock(side_effect=TimeoutError("shutdown")))
    client = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(main, "agent_runtime", runtime)
    monkeypatch.setattr(main, "_deepseek_clients", {1.0: client})
    monkeypatch.setenv("LLM_UPSTREAM_WARMUP_ENABLED", "false")
    monitor_started, monitor_stopped = asyncio.Event(), asyncio.Event()

    async def monitor(runtime):
        try:
            monitor_started.set()
            await asyncio.Event().wait()
        finally:
            monitor_stopped.set()

    monkeypatch.setattr(main.local_drain, "monitor", monitor)
    with pytest.raises(TimeoutError, match="shutdown"):
        async with main._llm_lifespan(main.app):
            assert monitor_started.is_set()
    assert monitor_stopped.is_set()
    client.aclose.assert_awaited_once()
    assert main._deepseek_clients == {}
