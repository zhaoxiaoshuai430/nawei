"""Configurable embedding providers with a deterministic development fallback."""

from __future__ import annotations

import hashlib
import asyncio
import math
import re
from typing import Any, Protocol

import httpx

from services.knowledge.config import KnowledgeSettings


class EmbeddingProvider(Protocol):
    dimension: int
    model_id: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized vector per input text."""

    async def close(self) -> None:
        """Release provider resources."""


class HashEmbeddingProvider:
    """Token-hash vectors for local integration tests, not production semantics."""

    def __init__(self, *, dimension: int, model_id: str = "hash-embedding-v1") -> None:
        self.dimension = max(32, dimension)
        self.model_id = model_id

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(text, self.dimension) for text in texts]

    async def close(self) -> None:
        return None


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, settings: KnowledgeSettings) -> None:
        if not settings.embedding_api_base_url or not settings.embedding_model:
            raise RuntimeError("Embedding API base URL and model are required.")
        self.dimension = settings.embedding_dimension
        self.model_id = settings.embedding_model
        self.url = settings.embedding_api_base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if settings.embedding_api_key:
            headers["Authorization"] = f"Bearer {settings.embedding_api_key}"
        self.client = httpx.AsyncClient(timeout=60.0, headers=headers, trust_env=False)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.post(self.url, json={"model": self.model_id, "input": texts})
        response.raise_for_status()
        body = response.json()
        ordered = sorted(body.get("data") or [], key=lambda item: int(item.get("index", 0)))
        vectors = [list(map(float, item["embedding"])) for item in ordered]
        if len(vectors) != len(texts):
            raise ValueError("Embedding provider returned an unexpected vector count.")
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError("Embedding vector dimension does not match EMBEDDING_DIMENSION.")
        return vectors

    async def close(self) -> None:
        await self.client.aclose()


class FastEmbedProvider:
    """CPU-only ONNX embeddings with bounded inference off the event loop."""

    def __init__(self, settings: KnowledgeSettings) -> None:
        self.dimension = settings.embedding_dimension
        self.model_id = settings.embedding_model
        self.cache_dir = settings.embedding_cache_dir or None
        self.threads = max(1, settings.embedding_threads)
        self._model: Any = None
        self._lock = asyncio.Lock()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_id, cache_dir=self.cache_dir,
                                        threads=self.threads, providers=["CPUExecutionProvider"], local_files_only=True)
        vectors = [vector.tolist() for vector in self._model.embed(texts, batch_size=32)]
        if len(vectors) != len(texts) or any(len(vector) != self.dimension for vector in vectors):
            raise ValueError("Embedding output does not match the configured count or dimension")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("Embedding output contains non-finite values")
        return vectors

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self._lock:
            task = asyncio.create_task(asyncio.to_thread(self._embed, texts))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise

    async def close(self) -> None:
        async with self._lock:
            self._model = None


def build_embedding_provider(settings: KnowledgeSettings) -> EmbeddingProvider:
    if settings.embedding_provider in {"openai", "openai_compatible", "http"}:
        return OpenAICompatibleEmbeddingProvider(settings)
    if settings.embedding_provider == "fastembed":
        return FastEmbedProvider(settings)
    if settings.embedding_provider != "hash":
        raise ValueError("Unknown embedding provider; use hash, fastembed or openai_compatible")
    return HashEmbeddingProvider(
        dimension=settings.embedding_dimension,
        model_id=settings.embedding_model or "hash-embedding-v1",
    )


def _hash_vector(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _tokens(text: str) -> list[str]:
    normalized = text.strip().lower()
    words = re.findall(r"[a-z0-9_]+", normalized)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    grams = [chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))]
    return words + grams + list(chinese)
