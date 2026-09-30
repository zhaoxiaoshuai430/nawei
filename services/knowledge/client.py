"""Optional internal HTTP client used by the LLM Agent."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from services.knowledge.schemas import (
    GraphCandidate,
    KnowledgeContext,
    KnowledgeExpansionOptions,
    KnowledgeRetrieveRequest,
    default_knowledge_expansion_options,
)


logger = logging.getLogger(__name__)


class KnowledgeClient:
    def __init__(self) -> None:
        self.enabled = _read_bool("KNOWLEDGE_ENABLED", True)
        self.required = _read_bool("KNOWLEDGE_REQUIRED", False)
        self.url = os.getenv(
            "KNOWLEDGE_RETRIEVE_URL",
            "http://127.0.0.1:8003/api/v1/knowledge/retrieve",
        ).strip()
        self.health_url = os.getenv("KNOWLEDGE_HEALTH_URL", "http://127.0.0.1:8003/health").strip()
        self.api_key = os.getenv("KNOWLEDGE_INTERNAL_API_KEY", "").strip()
        self.timeout_seconds = _read_float("KNOWLEDGE_TIMEOUT_SECONDS", 30.0)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds, connect=min(5.0, self.timeout_seconds)), trust_env=False,
        )
        self.last_error: str | None = None

    async def close(self) -> None:
        await self.client.aclose()

    async def retrieve(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        top_k: int = 8,
        graph_depth: int = 2,
        anchor_terms: list[str] | None = None,
        graph_candidates: list[GraphCandidate] | None = None,
        expansion: KnowledgeExpansionOptions | None = None,
    ) -> KnowledgeContext:
        if not self.enabled or not knowledge_base_id:
            return KnowledgeContext(knowledge_base_id=knowledge_base_id or "default", query=query)
        headers = {"X-Internal-API-Key": self.api_key} if self.api_key else {}
        request = KnowledgeRetrieveRequest(
            knowledge_base_id=knowledge_base_id,
            query=query,
            top_k=top_k,
            graph_depth=graph_depth,
            anchor_terms=anchor_terms or [],
            graph_candidates=graph_candidates or [],
            expansion=expansion or default_knowledge_expansion_options(),
        )
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.post(self.url, json=request.model_dump(), headers=headers)
                response.raise_for_status()
                context = KnowledgeContext(**response.json())
            self.last_error = None
            context.metadata["client_elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            return context
        except (TimeoutError, httpx.HTTPError, TypeError, ValueError) as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            self.last_error = (
                f"{type(exc).__name__}: elapsed_ms={elapsed_ms}, "
                f"budget_seconds={self.timeout_seconds:g}, http_status={status}"
            )
            logger.warning("Knowledge retrieval failed: %s", self.last_error)
            if self.required:
                raise RuntimeError(f"Required knowledge retrieval failed: {self.last_error}") from exc
            return KnowledgeContext(
                knowledge_base_id=knowledge_base_id,
                query=query,
                metadata={"error": self.last_error, "error_type": type(exc).__name__, "client_elapsed_ms": elapsed_ms},
            )

    async def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "status": "disabled"}
        headers = {"X-Internal-API-Key": self.api_key} if self.api_key else {}
        try:
            async with asyncio.timeout(5.0):
                response = await self.client.get(self.health_url, headers=headers, timeout=5.0)
                response.raise_for_status()
                body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Knowledge health response must be a JSON object")
            ready = body.get("status") == "ok" and body.get("ready") is True
            return {"enabled": True, "status": "ok" if ready else "degraded", "detail": body}
        except (TimeoutError, httpx.HTTPError, TypeError, ValueError) as exc:
            return {"enabled": True, "status": "error", "error": str(exc)[:300], "error_type": type(exc).__name__}


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default
