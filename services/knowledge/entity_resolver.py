"""Resolve LLM-proposed graph candidates against trusted Neo4j entities."""

from __future__ import annotations

from typing import Any

from services.knowledge.schemas import GraphCandidate, ResolvedKnowledgeEntity


class GraphEntityResolver:
    def __init__(self, neo4j_store: Any) -> None:
        self.neo4j_store = neo4j_store

    async def resolve(
        self,
        *,
        knowledge_base_id: str,
        candidates: list[GraphCandidate],
    ) -> list[ResolvedKnowledgeEntity]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda item: item.priority)
        return await self.neo4j_store.resolve_entities(
            knowledge_base_id=knowledge_base_id,
            candidates=ordered,
        )
