"""Assemble graph and content evidence into a bounded KnowledgeContext."""

from __future__ import annotations

from services.knowledge.schemas import (
    KnowledgeChunkHit,
    KnowledgeContext,
    KnowledgeGraphRelation,
    ResolvedKnowledgeEntity,
)


class GroundingAssembler:
    def assemble(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        chunks: list[KnowledgeChunkHit],
        resolved_entities: list[ResolvedKnowledgeEntity],
        relation_groups: dict[str, list[KnowledgeGraphRelation]],
        anchor_terms: list[str],
        retrieved_chunk_count: int,
        metadata: dict[str, object] | None = None,
    ) -> KnowledgeContext:
        normalized_groups = {
            name: _dedupe_relations(relations) for name, relations in relation_groups.items() if relations
        }
        relations = _dedupe_relations(
            [relation for group in normalized_groups.values() for relation in group]
        )
        if chunks and (resolved_entities or anchor_terms):
            grounding_status = "grounded"
        elif chunks:
            grounding_status = "content_grounded"
        elif resolved_entities:
            grounding_status = "graph_grounded"
        elif anchor_terms:
            grounding_status = "missing_anchor"
        else:
            grounding_status = "ungrounded"

        context_metadata: dict[str, object] = {
            "chunk_count": len(chunks),
            "relation_count": len(relations),
            "retrieved_chunk_count": retrieved_chunk_count,
            "anchor_terms": anchor_terms,
            "grounding_status": grounding_status,
            "content_grounding": "matched" if chunks else "missing",
            "graph_match_level": (
                resolved_entities[0].match_level if resolved_entities else "unresolved"
            ),
            "resolved_entity_names": [item.name for item in resolved_entities],
            "relation_group_counts": {
                name: len(relations) for name, relations in normalized_groups.items()
            },
        }
        context_metadata.update(metadata or {})
        return KnowledgeContext(
            knowledge_base_id=knowledge_base_id,
            query=query,
            chunks=chunks,
            relations=relations,
            resolved_entities=resolved_entities,
            relation_groups=normalized_groups,
            metadata=context_metadata,
        )


def _dedupe_relations(
    relations: list[KnowledgeGraphRelation],
) -> list[KnowledgeGraphRelation]:
    result: list[KnowledgeGraphRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in relations:
        key = (relation.source_id, relation.relation_type, relation.target_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)
    return result
