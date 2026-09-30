from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import models

from services.knowledge.config import load_settings
from services.knowledge.embeddings import FastEmbedProvider, build_embedding_provider
from services.knowledge.qdrant_store import QdrantKnowledgeStore
from services.knowledge.schemas import KnowledgeContext, KnowledgeRetrieveRequest
from services.knowledge.service import KnowledgeService


@pytest.mark.asyncio
async def test_cpu_embedding_validates_dimensions_without_downloading():
    provider = FastEmbedProvider(replace(load_settings(), embedding_dimension=2))
    provider._model = SimpleNamespace(embed=lambda texts, **kwargs: [SimpleNamespace(tolist=lambda: [0.6, 0.8]) for text in texts])
    assert await provider.embed(["test"]) == [[0.6, 0.8]]
    assert await provider.embed([]) == []
    provider.dimension = 3
    with pytest.raises(ValueError, match="dimension"):
        await provider.embed(["test"])
    await provider.close()


def test_unknown_provider_does_not_silently_fall_back_to_hash():
    with pytest.raises(ValueError, match="Unknown"):
        build_embedding_provider(replace(load_settings(), embedding_provider="typo"))


@pytest.mark.asyncio
@pytest.mark.parametrize("dimension, mismatches", [(128, 0), (512, 1)])
async def test_incompatible_index_is_rejected_without_recreation(dimension, mismatches):
    store = object.__new__(QdrantKnowledgeStore)
    store.collection = "existing"
    store.dimension = 512
    store.embedding_version = "semantic-v1"
    store.client = AsyncMock()
    store.client.collection_exists.return_value = True
    store.client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
        vectors=models.VectorParams(size=dimension, distance=models.Distance.COSINE))))
    store.client.count.return_value = SimpleNamespace(count=mismatches)
    with pytest.raises(ValueError):
        await store.initialize()
    store.client.create_collection.assert_not_called()
    store.client.create_payload_index.assert_not_called()


@pytest.mark.asyncio
async def test_retrieval_cache_is_namespaced_by_index_and_model(monkeypatch):
    settings = replace(load_settings(), qdrant_collection="shadow-index", embedding_provider="hash", embedding_model="hash-test", embedding_dimension=64)
    service = KnowledgeService(settings)
    cache = AsyncMock()
    cache.get.return_value = KnowledgeContext(knowledge_base_id="default", query="test")
    monkeypatch.setattr(service, "cache", cache)
    monkeypatch.setattr(service, "_require_stores", lambda: (None, None))
    await service.retrieve(KnowledgeRetrieveRequest(query="test"))
    context = cache.get.call_args.kwargs["route_context"]
    assert context["collection"] == "shadow-index"
    assert context["embedding_model"] == "hash-test"
    assert context["embedding_dimension"] == 64
