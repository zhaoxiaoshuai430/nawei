"""Failed retrieval branches must finish cancelling siblings before returning."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.knowledge.config import load_settings
from services.knowledge.neo4j_store import Neo4jKnowledgeStore
from services.knowledge.schemas import GraphCandidate, KnowledgeRetrieveRequest, ResolvedKnowledgeEntity
from services.knowledge.service import KnowledgeService


@pytest.mark.parametrize("stage", ["preparation", "expansion"])
@pytest.mark.parametrize("cancel_request", [False, True])
async def test_retrieval_cleans_up_sibling_work(monkeypatch, stage, cancel_request):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    service = KnowledgeService(load_settings())
    entered, cleaned, escape = asyncio.Event(), asyncio.Event(), asyncio.Event()
    child_tasks = []
    failure = RuntimeError("database unavailable")

    async def slow(*args, **kwargs):
        child_tasks.append(asyncio.current_task())
        entered.set()
        try:
            await escape.wait()
        finally:
            cleaned.set()

    async def failing(*args, **kwargs):
        await entered.wait()
        if cancel_request:
            await escape.wait()
        raise failure

    entity = ResolvedKnowledgeEntity(entity_id="one", name="database", canonical_name="database", entity_type="Concept")
    service.cache = SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock())
    service.qdrant = SimpleNamespace(search=AsyncMock(return_value=[]))
    service.neo4j = SimpleNamespace(resolve_entities=AsyncMock(return_value=[entity]),
                                   expand_entities=AsyncMock(return_value={}), expand=AsyncMock(return_value=[]))
    service.embeddings = SimpleNamespace(model_id="test", dimension=2, embed=AsyncMock(return_value=[[1, 0]]))
    if stage == "preparation":
        service.neo4j.resolve_entities.side_effect = failing
        service.embeddings.embed.side_effect = slow
    else:
        service.neo4j.expand_entities.side_effect = failing
        service.neo4j.expand.side_effect = slow
    request = KnowledgeRetrieveRequest(query="database", graph_candidates=[GraphCandidate(name="database")])
    task = asyncio.create_task(service.retrieve(request))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if cancel_request:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel_request else RuntimeError) as caught:
            await asyncio.wait_for(task, 1)
        if not cancel_request:
            assert caught.value is failure
        assert cleaned.is_set()
        assert all(child.done() for child in child_tasks)
        service.cache.set.assert_not_awaited()
    finally:
        escape.set()
        task.cancel()
        await asyncio.gather(task, *child_tasks, return_exceptions=True)


async def test_neo4j_relation_group_failure_cancels_other_queries():
    entered, cleaned, escape = asyncio.Event(), asyncio.Event(), asyncio.Event()
    children = []

    async def execute(query, **kwargs):
        children.append(asyncio.current_task())
        if kwargs["relation_types"] == ["FAIL"]:
            await entered.wait()
            raise RuntimeError("query failed")
        entered.set()
        try:
            await escape.wait()
            return [], None, None
        finally:
            cleaned.set()

    store = Neo4jKnowledgeStore.__new__(Neo4jKnowledgeStore)
    store.driver = SimpleNamespace(execute_query=execute)
    store.database = "test"
    try:
        with pytest.raises(RuntimeError, match="query failed"):
            await store.expand_entities(entity_ids=["one"], relation_groups={"a": (["FAIL"], 2), "b": (["WAIT"], 2)}, depth=1)
        assert cleaned.is_set()
        assert all(child.done() for child in children)
    finally:
        escape.set()
        await asyncio.gather(*children, return_exceptions=True)
