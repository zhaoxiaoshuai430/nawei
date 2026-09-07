"""Optional internal HTTP client used by the LLM Agent."""

from __future__ import annotations

import os
from typing import Any

import httpx

from services.knowledge.schemas import (
    GraphCandidate,
    KnowledgeContext,
    KnowledgeExpansionOptions,
    KnowledgeRetrieveRequest,
    default_knowledge_expansion_options,
)


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
        self.timeout_seconds = _read_float("KNOWLEDGE_TIMEOUT_SECONDS", 8.0)
        self.client = httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False)
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
        try:
            response = await self.client.post(self.url, json=request.model_dump(), headers=headers)
            response.raise_for_status()
            self.last_error = None
            return KnowledgeContext(**response.json())
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            self.last_error = str(exc)
            if self.required:
                raise RuntimeError(f"Required knowledge retrieval failed: {exc}") from exc
            return KnowledgeContext(
                knowledge_base_id=knowledge_base_id,
                query=query,
                metadata={"error": str(exc)[:300]},
            )

    async def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "status": "disabled"}
        headers = {"X-Internal-API-Key": self.api_key} if self.api_key else {}
        try:
            response = await self.client.get(self.health_url, headers=headers)
            response.raise_for_status()
            body = response.json()
            return {"enabled": True, "status": body.get("status", "ok"), "detail": body}
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            return {"enabled": True, "status": "error", "error": str(exc)[:300]}


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
