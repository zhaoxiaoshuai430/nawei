"""Redis cache for repeated knowledge retrieval requests."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover - optional in lightweight tests
    aioredis = None  # type: ignore[assignment]

from services.knowledge.schemas import KnowledgeContext


_CACHE_VERSION = "v7"
logger = logging.getLogger(__name__)


class KnowledgeCache:
    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.redis: Any = None
        self._retry_after = 0.0
        self._retry_interval_seconds = float(os.getenv("KNOWLEDGE_CACHE_RETRY_INTERVAL_SECONDS", "10"))

    async def initialize(self) -> None:
        if aioredis is None or self.redis is not None or time.monotonic() < self._retry_after:
            return
        try:
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            self._retry_after = 0.0
        except Exception as exc:
            await self._disable_redis("initialize", exc)

    async def close(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()
            self.redis = None

    async def get(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        top_k: int,
        graph_depth: int,
        anchor_terms: list[str] | None = None,
        route_context: dict[str, Any] | None = None,
    ) -> KnowledgeContext | None:
        await self.initialize()
        if self.redis is None:
            return None
        try:
            raw = await self.redis.get(
                self._key(
                    knowledge_base_id,
                    query,
                    top_k,
                    graph_depth,
                    anchor_terms or [],
                    route_context or {},
                )
            )
            return KnowledgeContext(**json.loads(raw)) if raw else None
        except Exception as exc:
            await self._disable_redis("get", exc)
            return None

    async def set(
        self,
        context: KnowledgeContext,
        *,
        top_k: int,
        graph_depth: int,
        anchor_terms: list[str] | None = None,
        route_context: dict[str, Any] | None = None,
    ) -> None:
        await self.initialize()
        if self.redis is None:
            return
        try:
            await self.redis.setex(
                self._key(
                    context.knowledge_base_id,
                    context.query,
                    top_k,
                    graph_depth,
                    anchor_terms or [],
                    route_context or {},
                ),
                self.ttl_seconds,
                context.model_dump_json(),
            )
        except Exception as exc:
            await self._disable_redis("set", exc)

    async def invalidate_base(self, knowledge_base_id: str) -> None:
        await self.initialize()
        if self.redis is None:
            return
        patterns = (
            f"knowledge:search:{knowledge_base_id}:*",
            f"knowledge:search:{_CACHE_VERSION}:{knowledge_base_id}:*",
        )
        try:
            for pattern in patterns:
                async for key in self.redis.scan_iter(match=pattern):
                    await self.redis.delete(key)
        except Exception as exc:
            await self._disable_redis("invalidate", exc)

    async def _disable_redis(self, operation: str, exc: Exception) -> None:
        client = self.redis
        self.redis = None
        self._retry_after = time.monotonic() + max(self._retry_interval_seconds, 0.1)
        logger.warning("Knowledge cache Redis %s failed; continuing without cache: %s", operation, type(exc).__name__)
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass

    @staticmethod
    def _key(
        knowledge_base_id: str,
        query: str,
        top_k: int,
        graph_depth: int,
        anchor_terms: list[str],
        route_context: dict[str, Any],
    ) -> str:
        normalized_anchors = sorted(
            {term.strip().lower() for term in anchor_terms if term.strip()}
        )
        cache_input = json.dumps(
            {
                "query": query.strip().lower(),
                "anchor_terms": normalized_anchors,
                "route_context": route_context,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(cache_input.encode("utf-8")).hexdigest()
        return f"knowledge:search:{_CACHE_VERSION}:{knowledge_base_id}:{top_k}:{graph_depth}:{digest}"
