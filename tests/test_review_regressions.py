"""Failure and concurrency boundaries discovered during architecture review."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import httpx

from services.knowledge import catalog
from services.knowledge.catalog import KnowledgeCatalog
from services.knowledge.schemas import KnowledgeDocumentRequest, KnowledgeStructuredDocumentRequest
from services.knowledge.service import KnowledgeService
from services.gateway.auth import public_metadata
from services.gateway import main as gateway
from services.llm import main as llm
from tests.test_optimization_boundaries import persistence_fixture
from app_protocols import LLMRequest, Message, MessageRole


def ingestion_fixture(structured=False):
    service = KnowledgeService.__new__(KnowledgeService)
    service.settings = SimpleNamespace(enabled=True, chunk_size=600, chunk_overlap=100, qdrant_collection="test")
    service.catalog = KnowledgeCatalog("")
    service.cache = SimpleNamespace(invalidate_base=AsyncMock())
    service.embeddings = SimpleNamespace(model_id="test", embed=AsyncMock(return_value=[[1.0]]))
    service.extractor = SimpleNamespace(extract=AsyncMock(return_value=SimpleNamespace(entities=[], relations=[])))
    service.qdrant = SimpleNamespace(upsert_chunks=AsyncMock(), mark_document_ready=AsyncMock(), delete_document=AsyncMock())
    service.neo4j = SimpleNamespace(upsert_document=AsyncMock(), delete_document=AsyncMock())
    request = (KnowledgeStructuredDocumentRequest(title="test", chunks=[{"chunk_key": "one", "text": "Test text"}])
               if structured else KnowledgeDocumentRequest(title="test", content="Test text"))
    return service, request, service.ingest_structured if structured else service.ingest


@pytest.mark.parametrize("structured", [False, True])
async def test_competing_import_and_delete_cannot_mutate_active_document(structured):
    service, request, ingest = ingestion_fixture(structured)
    entered, release = asyncio.Event(), asyncio.Event()

    async def write(*args):
        entered.set()
        await release.wait()

    service.qdrant.upsert_chunks.side_effect = write
    first = asyncio.create_task(ingest(request))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        document_id = next(iter(service.catalog.memory))
        with pytest.raises(RuntimeError, match="already in progress"):
            await ingest(request)
        with pytest.raises(RuntimeError, match="already in progress"):
            await service.delete_document(document_id)
    finally:
        release.set()
        result = await first
    assert result.status == "ready"
    assert (await ingest(request)).duplicate
    service.qdrant.delete_document.assert_not_awaited()
    assert await service.delete_document(result.document_id)


@pytest.mark.parametrize("structured", [False, True])
async def test_cancelled_import_cleans_stores_before_unlocking(structured):
    service, request, ingest = ingestion_fixture(structured)
    entered, cleanup_started, cleanup_release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def interrupted(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    async def cleanup(*args):
        cleanup_started.set()
        await cleanup_release.wait()

    service.neo4j.upsert_document.side_effect = interrupted
    service.qdrant.delete_document.side_effect = cleanup
    task = asyncio.create_task(ingest(request))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    await asyncio.wait_for(cleanup_started.wait(), 1)
    task.cancel()  # A second cancellation must not release the document guard.
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already in progress"):
        await ingest(request)
    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert next(iter(service.catalog.memory.values()))["status"] == "failed"
    service.neo4j.delete_document.assert_awaited_once()
    assert not service.catalog._active_documents
    service.neo4j.upsert_document.side_effect = None
    assert (await ingest(request)).status == "ready"


async def test_catalog_uses_separate_postgres_session_and_closes_on_contention(monkeypatch):
    connection = SimpleNamespace(fetchval=AsyncMock(return_value=False), terminate=Mock())
    monkeypatch.setattr(catalog, "asyncpg", SimpleNamespace(connect=AsyncMock(return_value=connection)))
    store = KnowledgeCatalog("test")
    store.pool = object()
    with pytest.raises(RuntimeError, match="already in progress"):
        async with store.document_operation("document"):
            pytest.fail("must not enter")
    connection.terminate.assert_called_once()
    assert not store._active_documents
    connection.fetchval.return_value = True
    async with store.document_operation("document"):
        assert "document" in store._active_documents
    assert connection.terminate.call_count == 2


async def test_history_is_committed_before_completion_while_extraction_is_busy(monkeypatch):
    monkeypatch.setenv("LLM_AGENT_ENABLED", "true")
    monkeypatch.setenv("MEMORY_POST_TURN_ASYNC", "true")
    runtime, turn = persistence_fixture()
    turn.direct_answer = None
    shared_history = []
    extraction_started, release = asyncio.Event(), asyncio.Event()

    async def persist(**kwargs):
        shared_history.append((kwargs["role"], kwargs["content"]))

    async def extract(**kwargs):
        extraction_started.set()
        await release.wait()
        return {}

    runtime.memory_store.persist_message = persist
    runtime.skill_manager.after_turn = extract
    try:
        await runtime.after_turn(turn, "first")
        assert shared_history == [("user", "hello"), ("assistant", "first")]
        await asyncio.wait_for(extraction_started.wait(), 1)
        await asyncio.wait_for(runtime.after_turn(turn, "second"), 1)
        assert shared_history[-2:] == [("user", "hello"), ("assistant", "second")]
    finally:
        release.set()
        await runtime.close()


async def test_history_timeout_does_not_report_success(monkeypatch):
    monkeypatch.setenv("LLM_AGENT_ENABLED", "true")
    monkeypatch.setenv("MEMORY_HISTORY_WRITE_TIMEOUT_SECONDS", "0.01")
    runtime, turn = persistence_fixture()

    async def stalled(**kwargs):
        await asyncio.Event().wait()

    runtime.memory_store.persist_message = stalled
    with pytest.raises(TimeoutError):
        await runtime.after_turn(turn, "answer")
    assert runtime.persistence_status()["failed"] == 1
    assert not runtime._persistence_tasks
    await runtime.close()


def test_public_metadata_cannot_supply_age_even_in_nested_profile():
    value = {"age": 17, "profile": {"identity": {"birth_date": "2000-01-01", "grade": "高一"}}, "answer_mode": "fast"}
    assert public_metadata(value) == {"profile": {"identity": {}}, "answer_mode": "fast"}
    assert value["age"] == 17


async def test_trusted_profile_age_overrides_internal_request_hint(monkeypatch):
    monkeypatch.setattr(llm.agent_runtime.memory_store, "get_profile", AsyncMock(return_value={"age": 6}))
    request = LLMRequest(request_id="age", trace_id="age", session_id="age", user_id="owner",
                         messages=[Message(role=MessageRole.USER, content="hello")], metadata={"age": 17})
    assert await llm._resolve_request_safety_age(request) == 6


@pytest.mark.parametrize("stream", [False, True])
async def test_gateway_filters_age_at_upstream_boundary(monkeypatch, stream):
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        if stream:
            event = {"event": "message.done", "request_id": "age", "trace_id": "age",
                     "session_id": "age", "data": {"text": "answer"}}
            return httpx.Response(200, text="data: " + json.dumps(event) + "\n\n")
        return httpx.Response(200, json={"request_id": "age", "text": "answer", "finish_reason": "stop",
                                        "usage": {"input_tokens": 0, "output_tokens": 0}})

    @asynccontextmanager
    async def client(timeout):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as value:
            yield value

    monkeypatch.setattr(gateway, "_gateway_http_client", client)
    request = LLMRequest(request_id="age", trace_id="age", session_id="age", user_id="owner",
                         messages=[Message(role=MessageRole.USER, content="hello")],
                         metadata={"profile": {"age": 17}, "answer_mode": "fast"})
    provider = gateway.HTTPLLMProvider()
    if stream:
        assert [event async for event in provider.stream_complete(request)]
    else:
        assert (await provider.complete(request)).text == "answer"
    assert captured[0]["metadata"] == {"profile": {}, "answer_mode": "fast"}
