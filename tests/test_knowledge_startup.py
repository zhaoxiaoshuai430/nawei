"""Cold retrieval readiness and bounded client failures."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest

from services.knowledge.client import KnowledgeClient
from services.knowledge.config import load_settings
from services.knowledge.schemas import KnowledgeContext
from services.knowledge.service import KnowledgeService
from services.llm import main as llm_main


async def test_retrieval_waits_within_configured_total_budget(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("KNOWLEDGE_REQUIRED", "true")
    client = KnowledgeClient()

    async def handler(request):
        await asyncio.sleep(0.04)
        return httpx.Response(200, json={"knowledge_base_id": "default", "query": "cold"})

    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        context = await client.retrieve(knowledge_base_id="default", query="cold")
        assert context.query == "cold" and context.metadata["client_elapsed_ms"] >= 30
    finally:
        await client.close()


@pytest.mark.parametrize("required", [True, False])
async def test_total_timeout_is_bounded_and_diagnostic(monkeypatch, required):
    monkeypatch.setenv("KNOWLEDGE_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("KNOWLEDGE_REQUIRED", str(required))
    client = KnowledgeClient()

    async def handler(request):
        await asyncio.sleep(1)
        raise AssertionError("Should have been cancelled")

    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        if required:
            with pytest.raises(RuntimeError, match="TimeoutError: elapsed_ms=.*budget_seconds=0.01"):
                await client.retrieve(knowledge_base_id="default", query="cold")
        else:
            context = await client.retrieve(knowledge_base_id="default", query="cold")
            assert context.metadata["error_type"] == "TimeoutError"
        assert "TimeoutError" in client.last_error
    finally:
        await client.close()


async def test_upstream_failure_does_not_expose_response_or_credentials(monkeypatch, caplog):
    monkeypatch.setenv("KNOWLEDGE_REQUIRED", "true")
    client = KnowledgeClient()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(503, text="secret-in-server-error")))
    try:
        with pytest.raises(RuntimeError, match="http_status=503") as error:
            await client.retrieve(knowledge_base_id="default", query="private question")
        assert "secret-in-server-error" not in str(error.value) + caplog.text
        assert "private question" not in caplog.text
    finally:
        await client.close()


async def test_warmup_exercises_full_retrieval_with_a_fresh_cache_key(monkeypatch):
    service = KnowledgeService(replace(load_settings(), embedding_provider="hash", embedding_dimension=128))
    service.retrieve = AsyncMock(return_value=KnowledgeContext(knowledge_base_id="default", query="warm", metadata={"timings_ms": {"graph_expansion": 1}}))
    await service._warm_retrieval()
    first = service.retrieve.call_args.args[0]
    await service._warm_retrieval()
    second = service.retrieve.call_args.args[0]
    assert first.query != second.query
    assert first.graph_candidates
    assert service.startup_warmup["status"] == "ready"
    assert service.startup_warmup["timings_ms"]["graph_expansion"] == 1


async def test_warmup_timeout_prevents_false_readiness(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_STARTUP_WARMUP_TIMEOUT_SECONDS", "0.01")
    service = KnowledgeService(replace(load_settings(), embedding_provider="hash", embedding_dimension=128))

    async def slow(request):
        await asyncio.sleep(1)

    service.retrieve = slow
    with pytest.raises(RuntimeError, match="warmup failed: TimeoutError"):
        await service._warm_retrieval()
    assert service.startup_warmup == {"status": "failed", "error_type": "TimeoutError"}


async def test_disabled_warmup_does_not_query(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_STARTUP_WARMUP_ENABLED", "false")
    service = KnowledgeService(replace(load_settings(), embedding_provider="hash", embedding_dimension=128))
    service.retrieve = AsyncMock()
    await service._warm_retrieval()
    service.retrieve.assert_not_called()
    assert service.startup_warmup["status"] == "disabled"


@pytest.mark.parametrize("body", [[], None, {}, {"ready": True}, {"status": "ok", "ready": False}])
async def test_knowledge_health_rejects_invalid_readiness(monkeypatch, body):
    monkeypatch.setenv("KNOWLEDGE_ENABLED", "true")
    client = KnowledgeClient()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=body)))
    try:
        result = await client.health()
        assert result["status"] != "ok"
    finally:
        await client.close()


async def test_knowledge_health_bounds_total_wait(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_ENABLED", "true")
    client = KnowledgeClient()

    async def stalled(request):
        await asyncio.Event().wait()

    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(stalled))
    try:
        result = await asyncio.wait_for(client.health(), timeout=6)
        assert result["status"] == "error"
        assert result["error_type"] == "TimeoutError"
    finally:
        await client.close()


@pytest.mark.parametrize(("required", "knowledge_status", "expected"), [
    (True, "ok", "ok"),
    (True, "degraded", "degraded"),
    (True, "error", "degraded"),
    (True, "disabled", "degraded"),
    (False, "error", "ok"),
])
async def test_llm_health_accounts_for_required_knowledge(monkeypatch, required, knowledge_status, expected):
    monkeypatch.setenv("LLM_MEMORY_REQUIRED", "true")
    monkeypatch.setenv("KNOWLEDGE_REQUIRED", str(required))
    monkeypatch.setattr(llm_main.agent_runtime, "health", AsyncMock(return_value={
        "memory": {"postgres": "ok", "redis": "ok"},
        "knowledge": {"status": knowledge_status},
    }))
    result = await llm_main.health()
    assert result["status"] == expected
