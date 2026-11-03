"""Summary scheduling starts only after dependencies and schema are ready."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from services.llm.agent_runtime import LLMAgentRuntime


@pytest.mark.parametrize("schema_available", [True, False])
async def test_recovered_database_is_checked_before_scheduler_start(monkeypatch, schema_available):
    monkeypatch.setenv("SUMMARY_ENABLED", "true")
    monkeypatch.setenv("LLM_MEMORY_REQUIRED", "true")
    monkeypatch.setenv("KNOWLEDGE_REQUIRED", "true")
    memory = SimpleNamespace(_pool=None, initialize=AsyncMock())
    knowledge = SimpleNamespace(health=AsyncMock(return_value={"status": "ok"}))
    runtime = LLMAgentRuntime(memory_store=memory, knowledge_client=knowledge)
    runtime.summary_worker.start = Mock()

    async def recover():
        runtime.summary_worker.start.assert_not_called()
        memory._pool = object()
        return {"postgres": "ok", "redis": "ok"}

    async def check_schema():
        assert memory._pool is not None
        knowledge.health.assert_awaited_once()
        runtime.summary_worker.start.assert_not_called()
        if not schema_available:
            raise RuntimeError("summary migration missing")

    memory.health = AsyncMock(side_effect=recover)
    runtime.summary_repository.check_schema = AsyncMock(side_effect=check_schema)
    if schema_available:
        await runtime.initialize()
        runtime.summary_worker.start.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="migration missing"):
            await runtime.initialize()
        runtime.summary_worker.start.assert_not_called()
    runtime.summary_repository.check_schema.assert_awaited_once()


async def test_summaries_cannot_start_with_only_in_memory_fallback(monkeypatch):
    monkeypatch.setenv("SUMMARY_ENABLED", "true")
    monkeypatch.setenv("LLM_MEMORY_REQUIRED", "false")
    monkeypatch.setenv("KNOWLEDGE_REQUIRED", "false")
    memory = SimpleNamespace(_pool=None, initialize=AsyncMock())
    knowledge = SimpleNamespace(health=AsyncMock(return_value={"status": "ok"}))
    runtime = LLMAgentRuntime(memory_store=memory, knowledge_client=knowledge)
    runtime.summary_worker.start = Mock()
    with pytest.raises(RuntimeError, match="Summary PostgreSQL is unavailable"):
        await runtime.initialize()
    runtime.summary_worker.start.assert_not_called()


async def test_required_knowledge_failure_does_not_leave_scheduler_running(monkeypatch):
    monkeypatch.setenv("SUMMARY_ENABLED", "true")
    monkeypatch.setenv("LLM_MEMORY_REQUIRED", "false")
    monkeypatch.setenv("KNOWLEDGE_REQUIRED", "true")
    monkeypatch.setattr("services.llm.agent_runtime.asyncio.sleep", AsyncMock())
    memory = SimpleNamespace(_pool=object(), initialize=AsyncMock())
    knowledge = SimpleNamespace(health=AsyncMock(return_value={"status": "unavailable"}))
    runtime = LLMAgentRuntime(memory_store=memory, knowledge_client=knowledge)
    runtime.summary_repository.check_schema = AsyncMock()
    runtime.summary_worker.start = Mock()
    with pytest.raises(RuntimeError, match="Required knowledge"):
        await runtime.initialize()
    runtime.summary_worker.start.assert_not_called()
